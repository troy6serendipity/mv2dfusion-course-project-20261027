"""Small-sample course benchmark of MV2DFusion and its independent expert baselines.

All metrics are custom center-distance metrics, NOT official nuScenes mAP/NDS.
No training or GT-based score tuning is performed.
"""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes.eval.detection.config import config_factory

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("MV2DFUSION_UPSTREAM",str(REPO_ROOT / "upstream/MV2DFusion")))
DATA = Path(os.environ.get("NUSCENES_ROOT",str(REPO_ROOT / "data/nuscenes")))
INDICES = [0, 7, 14, 21, 28, 35, 40, 48, 56, 64, 72, 80]
CONDITIONS = ["clean", "image_zero", "lidar_keep50", "calib_yaw2deg"]
METHODS = ["lidar_expert", "simple_projection_fusion", "mv2dfusion"]
SCORE = 0.3
DISTANCE = 2.0
IMAGE_IOU = 0.1

def strip_init(obj):
    if isinstance(obj, dict):
        for key in list(obj):
            if key == "pretrained" or (key == "init_cfg" and isinstance(obj[key], dict) and obj[key].get("type") == "Pretrained"):
                obj[key] = None
            else:
                strip_init(obj[key])
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strip_init(item)

def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()

def prediction(boxes, scores, labels):
    return {
        "boxes": boxes.clone().to("cpu"),
        "scores": scores.detach().cpu().clone(),
        "labels": labels.detach().cpu().clone(),
    }

def reset_temporal(model):
    model.prev_scene_token = None
    model.test_clip_id = 0
    model.current_seq = 0
    head = model.fusion_bbox_head
    head.reset_memory()
    for name in ["memory_embedding", "memory_reference_point", "memory_timestamp",
                 "memory_egopose", "memory_velo", "memory_query_mask", "memory_instance_inds"]:
        setattr(head, name, None)

def install_capture(model, cache, control):
    original_boxes = model.pts_backbone.bbox_head.get_bboxes

    def capture_boxes(*args, **kwargs):
        result = original_boxes(*args, **kwargs)
        boxes, scores, labels, _ = result[0]
        # Clone before convention conversion. Never mutate the network's result.
        clone_boxes, clone_labels = model.convert_to_fsd_anno(
            boxes.clone(), labels.clone(), inv=True)
        cache["lidar"] = prediction(clone_boxes, scores.clone(), clone_labels)
        return result

    model.pts_backbone.bbox_head.get_bboxes = capture_boxes
    original_roi = model.forward_roi_head

    def capture_roi(imgs, feats, metas):
        result = original_roi(imgs, feats, metas)
        cache["detections2d"] = [array(d) for d in result[1]]
        cache["camera_metas"] = [
            dict(lidar2img=array(m["lidar2img"]),
                 img_shape=list(m["img_shape"]))
            for m in metas
        ]
        return result

    model.forward_roi_head = capture_roi
    original_simple = model.simple_test

    def corrupt_and_forward(metas, **data):
        condition = control["condition"]
        data = dict(data)
        metas = [dict(m) for m in metas]
        cache["condition_details"] = {
            "condition": condition,
            "image_tensor_shape": list(data["img"].shape),
            "points_before": int(data["points"].shape[1]),
        }
        if condition == "image_zero":
            data["img"] = torch.zeros_like(data["img"])
            cache["condition_details"]["image_fill"] = "zero in normalized input space (mean-pixel fill), all six cameras"
        elif condition == "lidar_keep50":
            points = data["points"]
            generator = torch.Generator(device=points.device)
            generator.manual_seed(control["seed"])
            keep = torch.randperm(points.shape[1], generator=generator, device=points.device)[:max(1, points.shape[1] // 2)]
            keep = keep.sort().values
            data["points"] = points[:, keep, :].contiguous()
        elif condition == "calib_yaw2deg":
            ext = data["extrinsics"]
            rot = torch.eye(4, device=ext.device, dtype=ext.dtype)
            angle = math.radians(2.0)
            rot[0, 0] = math.cos(angle)
            rot[0, 1] = -math.sin(angle)
            rot[1, 0] = math.sin(angle)
            rot[1, 1] = math.cos(angle)
            data["extrinsics"] = ext @ rot
            data["lidar2img"] = data["intrinsics"] @ data["extrinsics"]
            for b, meta in enumerate(metas):
                for key in ["extrinsics", "intrinsics", "lidar2img"]:
                    meta[key] = array(data[key][b])
            cache["condition_details"]["extrinsic_yaw_bias_deg"] = 2.0
        elif condition != "clean":
            raise ValueError(condition)
        cache["condition_details"]["points_after"] = int(data["points"].shape[1])
        return original_simple(metas, **data)

    model.simple_test = corrupt_and_forward

def iou_2d(box, others):
    if len(others) == 0:
        return np.zeros(0, dtype=np.float64)
    top = np.maximum(box[:2], others[:, :2])
    bottom = np.minimum(box[2:], others[:, 2:4])
    intersection = np.maximum(bottom - top, 0).prod(-1)
    a = np.maximum(box[2:] - box[:2], 0).prod()
    b = np.maximum(others[:, 2:4] - others[:, :2], 0).prod(-1)
    return intersection / np.maximum(a + b - intersection, 1e-9)

def simple_fusion(lidar, detections, camera_metas):
    corners = lidar["boxes"].corners.numpy()
    scores = lidar["scores"].numpy().copy()
    labels = lidar["labels"].numpy()
    details = Counter()
    for i, (corner, score, label) in enumerate(zip(corners, scores.copy(), labels)):
        visible = False
        supporting_score = None
        for det, meta in zip(detections, camera_metas):
            homogeneous = np.c_[corner, np.ones(8)] @ meta["lidar2img"].T
            if (homogeneous[:, 2] <= 0.1).any():
                continue
            uv = homogeneous[:, :2] / homogeneous[:, 2:3]
            height, width = meta["img_shape"][:2]
            rect = np.r_[uv.min(0), uv.max(0)]
            rect[[0, 2]] = np.clip(rect[[0, 2]], 0, width)
            rect[[1, 3]] = np.clip(rect[[1, 3]], 0, height)
            if rect[2] <= rect[0] or rect[3] <= rect[1]:
                continue
            visible = True
            same_class = det[(det[:, 5].astype(np.int64) == label) & (det[:, 4] >= 0.1)]
            overlap = iou_2d(rect, same_class)
            supported = same_class[overlap >= IMAGE_IOU]
            if len(supported):
                value = float(supported[:, 4].max())
                supporting_score = value if supporting_score is None else max(supporting_score, value)
        if supporting_score is not None:
            scores[i] = math.sqrt(max(0.0, float(score) * supporting_score))
            details["supported"] += 1
        elif visible:
            scores[i] = 0.5 * float(score)
            details["visible_unmatched"] += 1
        else:
            details["outside_camera_visibility"] += 1
    result = prediction(lidar["boxes"], torch.as_tensor(scores), lidar["labels"])
    return result, dict(details)

def range_mask(boxes, labels, info, names, class_ranges):
    centers = boxes.gravity_center.numpy()
    rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    translation = np.asarray(info["lidar2ego_translation"])
    centers_ego = centers @ rotation.T + translation
    radii = np.linalg.norm(centers_ego[:, :2], axis=1)
    return np.array([
        0 <= int(label) < len(names) and radii[i] <= class_ranges[names[int(label)]]
        for i, label in enumerate(labels)
    ], dtype=bool)

def evaluate(pred, gt, info, names, class_ranges):
    scores = pred["scores"].numpy()
    labels = pred["labels"].numpy()
    valid_pred = np.isfinite(scores) & (scores >= SCORE)
    valid_pred &= range_mask(pred["boxes"], labels, info, names, class_ranges)
    pred_xy = pred["boxes"].gravity_center.numpy()[valid_pred, :2]
    pred_labels = labels[valid_pred]
    gt_labels = np.asarray(gt["gt_labels_3d"])
    valid_gt = range_mask(gt["gt_bboxes_3d"], gt_labels, info, names, class_ranges)
    gt_xy = gt["gt_bboxes_3d"].gravity_center.numpy()[valid_gt, :2]
    gt_labels = gt_labels[valid_gt]
    classes = {}
    matched_distances = []
    for label, name in enumerate(names):
        p = pred_xy[pred_labels == label]
        g = gt_xy[gt_labels == label]
        tp = 0
        distances = []
        if len(p) and len(g):
            dist = np.linalg.norm(p[:, None] - g[None, :], axis=-1)
            # All unmatched rows have a dummy column. One extra feasible match
            # saves 1e6, overwhelming any possible total accepted distance.
            penalty = 1e6
            cost = np.full((len(p), len(g) + len(p)), penalty)
            cost[:, :len(g)] = np.where(dist <= DISTANCE, dist, 4 * penalty)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if col < len(g) and dist[row, col] <= DISTANCE:
                    tp += 1
                    distances.append(float(dist[row, col]))
        matched_distances.extend(distances)
        classes[name] = dict(tp=tp, fp=int(len(p) - tp), fn=int(len(g) - tp),
                             gt=int(len(g)), predictions=int(len(p)),
                             matched_distance_sum_m=float(sum(distances)))
    tp = sum(x["tp"] for x in classes.values())
    fp = sum(x["fp"] for x in classes.values())
    fn = sum(x["fn"] for x in classes.values())
    return dict(tp=tp, fp=fp, fn=fn, gt=tp + fn, predictions=tp + fp,
                matched_distance_sum_m=float(sum(matched_distances)),
                matched_center_error_m=float(np.mean(matched_distances)) if matched_distances else None,
                by_class=classes)

def rates(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                matched_center_error_m=counts["matched_distance_sum_m"] / tp if tp else None)

def save_predictions(path, pred):
    np.savez_compressed(path, boxes=pred["boxes"].tensor.numpy(),
                        scores=pred["scores"].numpy(), labels=pred["labels"].numpy())

def summarize(records, names):
    rows = []
    by_class = {}
    for condition in dict.fromkeys(r["condition"] for r in records):
        selected = [r for r in records if r["condition"] == condition]
        for method in METHODS:
            count = {key: sum(r["methods"][method][key] for r in selected)
                     for key in ["tp", "fp", "fn", "gt", "predictions", "matched_distance_sum_m"]}
            row = dict(condition=condition, method=method, frames=len(selected), **count, **rates(count))
            # Instrumented full model time, shared by all three captured methods.
            row["shared_full_forward_s_mean"] = float(np.mean([r["instrumented_full_forward_seconds"] for r in selected]))
            row["shared_full_forward_s_std"] = float(np.std([r["instrumented_full_forward_seconds"] for r in selected]))
            row["peak_allocated_MiB_max"] = max(r["peak_allocated_MiB"] for r in selected)
            row["simple_fusion_postprocess_s_mean"] = float(np.mean([r["simple_fusion_postprocess_seconds"] for r in selected]))
            rows.append(row)
            entries = {}
            for name in names:
                counts = {key: sum(r["methods"][method]["by_class"][name][key] for r in selected)
                          for key in ["tp", "fp", "fn", "gt", "predictions", "matched_distance_sum_m"]}
                entries[name] = dict(**counts, **rates(counts))
            by_class[condition + "/" + method] = entries
    return rows, by_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or Path(os.environ.get("MV2D_EXPERIMENT_OUTPUT",str(REPO_ROOT / "experiment_outputs"))) / ("benchmark_smoke" if args.smoke else "benchmark_main")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    indices = INDICES[:1] if args.smoke else INDICES
    conditions = CONDITIONS[:1] if args.smoke else CONDITIONS
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    for module in ["projects.mmdet3d_plugin", "projects.fsdv2"]:
        importlib.import_module(module)
    cfg = Config.fromfile(str(REPO / "projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py"))
    strip_init(cfg.model)
    cfg.model.train_cfg = None
    cfg.data.test.update(data_root=str(DATA) + "/",
        ann_file=str(DATA / "nuscenes2d_temporal_infos_val_new.pkl"),
        test_mode=True, use_valid_flag=False)
    cfg.data.test.pop("samples_per_gpu", None)
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint_path = Path(os.environ.get("MV2DFUSION_CHECKPOINT",str(REPO_ROOT / "weights/mv2dfusion_r50_nusc_latest.pth")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Incomplete official checkpoint match: " + str(loaded))
    del checkpoint, state
    model.CLASSES = dataset.CLASSES
    model = model.cuda().eval()
    cache, control = {}, {}
    install_capture(model, cache, control)
    net = MMDataParallel(model, device_ids=[0]).eval()
    names = list(dataset.CLASSES)
    class_ranges = config_factory("detection_cvpr_2019").class_range
    manifest, gt_counts = [], Counter()
    for i in indices:
        info = dataset.data_infos[i]
        gt = dataset.get_ann_info(i)
        labels = np.asarray(gt["gt_labels_3d"])
        valid = range_mask(gt["gt_bboxes_3d"], labels, info, names, class_ranges)
        counts = Counter(names[int(x)] for x in labels[valid] if 0 <= x < len(names))
        gt_counts.update(counts)
        manifest.append(dict(index=i, token=info["token"], scene_token=info["scene_token"],
                             timestamp=int(info["timestamp"]), frame_idx=int(info["frame_idx"]),
                             gt_after_lidar_hit_and_range_filter=int(valid.sum()), gt_by_class=dict(counts)))
    json_write(output / "sample_manifest.json", manifest)
    json_write(output / "gt_statistics.json", dict(total=sum(gt_counts.values()), by_class=dict(gt_counts),
        use_valid_flag=False, criterion="num_lidar_pts > 0; valid categories; official per-class ego-distance range",
        class_range_m=class_ranges))
    protocol = dict(score_threshold=SCORE, center_distance_threshold_m=DISTANCE,
        matching="same-class BEV center distance, Hungarian maximum cardinality then minimum distance using 1e6 dummy penalty",
        gt_filter="use_valid_flag=False => num_lidar_pts > 0; official class ego-range applied to GT and predictions",
        metrics="custom micro P/R/F1 and matched center error; NOT official mAP or NDS",
        simple_fusion="same-class resized-image projection IoU >= 0.1; matched sqrt(sL*sI), visible unmatched 0.5*sL, invisible unchanged",
        image_detection_support_min_score=0.1,
        image_zero="all six normalized image inputs filled with zero (mean-pixel fill), not raw black images",
        lidar_keep50="retain floor(N/2) input points with fixed per-frame seed; preserve retained order",
        calib_yaw2deg="extrinsics right-multiplied by +2 degree LiDAR-frame yaw; recompute K@E and update metadata",
        temporal="reset all model temporal state for every frame and condition",
        shared_checkpoint=True, shared_expert_weights=True,
        timing="shared instrumented full forward for all captured methods; not separate baseline end-to-end latency",
        peak_memory="PyTorch peak allocated; not all device/third-party allocations",
        sample_indices=indices, conditions=conditions)
    json_write(output / "protocol.json", protocol)
    records = []
    (output / "per_frame_results.jsonl").write_text("", encoding="utf-8")
    for condition in conditions:
        for i in indices:
            random.seed(20261026 + i)
            np.random.seed(20261026 + i)
            torch.manual_seed(20261026 + i)
            torch.cuda.manual_seed_all(20261026 + i)
            control.update(condition=condition, seed=20261026 + i)
            cache.clear()
            reset_temporal(model)
            data = collate([dataset[i]], samples_per_gpu=1)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.no_grad():
                result = net(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            fusion_raw = result[0]["pts_bbox"]
            fused = prediction(fusion_raw["boxes_3d"], fusion_raw["scores_3d"], fusion_raw["labels_3d"])
            assert "lidar" in cache and "detections2d" in cache
            started_simple = time.perf_counter()
            simple, support = simple_fusion(cache["lidar"], cache["detections2d"], cache["camera_metas"])
            simple_elapsed = time.perf_counter() - started_simple
            preds = dict(lidar_expert=cache["lidar"], simple_projection_fusion=simple, mv2dfusion=fused)
            info, gt = dataset.data_infos[i], dataset.get_ann_info(i)
            metrics = {}
            for method, pred in preds.items():
                assert torch.isfinite(pred["boxes"].tensor).all() and torch.isfinite(pred["scores"]).all()
                metrics[method] = evaluate(pred, gt, info, names, class_ranges)
                metrics[method].update(rates(metrics[method]))
                save_predictions(output / "predictions" / ("%s_frame%03d_%s.npz" % (condition, i, method)), pred)
            record = dict(condition=condition, index=i, token=info["token"], scene_token=info["scene_token"],
                instrumented_full_forward_seconds=elapsed,
                simple_fusion_postprocess_seconds=simple_elapsed,
                peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                methods=metrics, simple_fusion_support=support,
                condition_details=cache["condition_details"])
            records.append(record)
            with (output / "per_frame_results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            json_write(output / "running_progress.json", dict(completed=len(records), expected=len(indices)*len(conditions),
                latest={key:record[key] for key in ["condition", "index", "instrumented_full_forward_seconds", "peak_allocated_MiB"]},
                metrics={m:{k:metrics[m][k] for k in ["tp","fp","fn","precision","recall","f1"]} for m in METHODS}))
            print(json.dumps(dict(condition=condition, index=i,
                methods={m:{k:metrics[m][k] for k in ["tp","fp","fn","f1"]} for m in METHODS}),
                ensure_ascii=False), flush=True)
            del result, fused, simple, preds, data
            cache.clear()
            gc.collect()
    rows, per_class = summarize(records, names)
    summary = dict(status="REAL_SMALL_SAMPLE_BENCHMARK_COMPLETED", full_benchmark_reproduced=False,
        checkpoint=str(checkpoint_path), checkpoint_fully_matched=True,
        protocol=protocol, sample_manifest=manifest, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), results=rows)
    json_write(output / "benchmark_summary.json", summary)
    json_write(output / "per_frame_results.json", records)
    json_write(output / "per_class_results.json", per_class)
    import csv
    with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("BENCHMARK COMPLETE", str(output), flush=True)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
