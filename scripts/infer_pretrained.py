"""Real MV2DFusion checkpoint inference; no synthetic detections."""
import argparse,hashlib,importlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch,cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import MMDataParallel,collate
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
ROOT=Path(__file__).resolve().parents[1]
REPO=Path(os.environ.get("MV2DFUSION_UPSTREAM",str(ROOT/"upstream/MV2DFusion")))
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
NAMES=["car","truck","construction_vehicle","bus","trailer","barrier","motorcycle","bicycle","pedestrian","traffic_cone"]
COLORS=np.array([[21,175,225],[245,120,55],[175,75,230],[50,175,100],[200,180,50],[140,140,140],[180,50,120],[35,180,190],[245,60,60],[250,180,30]])/255.

def strip_init(obj):
    if isinstance(obj,dict):
        for key in list(obj):
            if key=="pretrained" or (key=="init_cfg" and isinstance(obj[key],dict) and obj[key].get("type")=="Pretrained"): obj[key]=None
            else: strip_init(obj[key])
    elif isinstance(obj,(list,tuple)):
        for item in obj: strip_init(item)

def raw_path(path,root):
    p=Path(path)
    if p.exists(): return p
    text=str(path)
    for part in ["samples/","sweeps/","maps/"]:
        if part in text: return root/text[text.index(part):]
    return root/p

def projection(cam):
    r=np.linalg.inv(cam["sensor2lidar_rotation"])
    t=cam["sensor2lidar_translation"]@r.T
    rt=np.eye(4)
    rt[:3,:3],rt[3,:3]=r.T,-t
    k=np.eye(4)
    k[:3,:3]=cam["cam_intrinsic"]
    return k@rt.T

def render(info,pred,index,out,root,threshold):
    box=pred["boxes_3d"]
    scores,labels=pred["scores_3d"].numpy(),pred["labels_3d"].numpy()
    keep=scores>=threshold
    corners=box.corners.numpy()[keep]
    scores,labels=scores[keep],labels[keep]
    np.savez_compressed(out/("frame_%03d_predictions.npz"%index),boxes=box.tensor.numpy(),scores=pred["scores_3d"].numpy(),labels=pred["labels_3d"].numpy())
    cameras=["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]
    fig,axes=plt.subplots(2,3,figsize=(18,8.2))
    for ax,name in zip(axes.flat,cameras):
        cam=info["cams"][name]
        image=cv2.imread(str(raw_path(cam["data_path"],root)))
        assert image is not None,cam["data_path"]
        h,w=image.shape[:2]
        matrix=projection(cam)
        for corner,score,label in zip(corners,scores,labels):
            hom=np.c_[corner,np.ones(8)]@matrix.T
            if (hom[:,2]<=.1).any(): continue
            uv=hom[:,:2]/hom[:,2:3]
            if not ((uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)).any(): continue
            pixels=np.clip(uv,-10000,10000).astype(np.int32)
            color=tuple(int(x*255) for x in COLORS[label][::-1])
            for a,b in EDGES: cv2.line(image,tuple(pixels[a]),tuple(pixels[b]),color,3,cv2.LINE_AA)
            point=tuple(np.maximum(pixels.min(0),[0,25]))
            cv2.putText(image,"%s %.2f"%(NAMES[label],score),point,cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
        ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        ax.set_title(name,fontsize=10)
        ax.axis("off")
    fig.suptitle("MV2DFusion R50 | real nuScenes Mini | PREDICTIONS >= %.2f"%threshold,fontsize=15)
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.9,wspace=.02,hspace=.16)
    fig.savefig(out/("frame_%03d_cameras.png"%index),dpi=150)
    plt.close(fig)
    pts=np.fromfile(raw_path(info["lidar_path"],root),dtype=np.float32).reshape(-1,5)
    pts=pts[(np.abs(pts[:,0])<55)&(np.abs(pts[:,1])<55)]
    fig,ax=plt.subplots(figsize=(9,9))
    ax.set_facecolor("#101827")
    ax.scatter(pts[::2,0],pts[::2,1],s=.3,c=pts[::2,2],cmap="gray",vmin=-3,vmax=3,alpha=.7)
    for corner,label in zip(corners,labels):
        for a,b in [(0,3),(3,7),(7,4),(4,0)]: ax.plot(corner[[a,b],0],corner[[a,b],1],color=COLORS[label],lw=1.4)
    ax.scatter([0],[0],marker="^",c="white",s=55)
    ax.set(xlim=(-55,55),ylim=(-55,55),xlabel="LiDAR x (m)",ylabel="LiDAR y (m)",title="MV2DFusion | REAL sample | predicted boxes (BEV)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out/("frame_%03d_bev.png"%index),dpi=170)
    plt.close(fig)
    fig=plt.figure(figsize=(11,7.5))
    ax=fig.add_subplot(111,projection="3d")
    cloud=pts[::6]
    ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],s=.4,c=cloud[:,2],cmap="gray",vmin=-3,vmax=3,alpha=.35)
    for corner,label in zip(corners,labels):
        for a,b in EDGES: ax.plot(corner[[a,b],0],corner[[a,b],1],corner[[a,b],2],color=COLORS[label],lw=1.)
    ax.set(xlim=(-50,50),ylim=(-50,50),zlim=(-4,9),xlabel="x (m)",ylabel="y (m)",zlabel="z (m)",title="MV2DFusion R50 | REAL nuScenes Mini | 3D predictions")
    ax.set_box_aspect((1,1,.25))
    ax.view_init(elev=25,azim=-65)
    fig.tight_layout()
    fig.savefig(out/("frame_%03d_pointcloud3d.png"%index),dpi=170)
    plt.close(fig)
    return int(keep.sum())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--frames",type=int,default=1)
    parser.add_argument("--score",type=float,default=.3)
    args=parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0,str(REPO))
    for module in ["projects.mmdet3d_plugin","projects.fsdv2"]: importlib.import_module(module)
    cfg=Config.fromfile(str(REPO/"projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py"))
    strip_init(cfg.model)
    cfg.model.train_cfg=None
    root=Path(os.environ.get("NUSCENES_ROOT",str(ROOT/"data/nuscenes")))
    cfg.data.test.update(data_root=str(root)+"/",ann_file=str(root/"nuscenes2d_temporal_infos_val_new.pkl"),test_mode=True)
    cfg.data.test.pop("samples_per_gpu",None)
    dataset=build_dataset(cfg.data.test)
    model=build_model(cfg.model,test_cfg=cfg.get("test_cfg"))
    checkpoint_path=Path(os.environ.get("MV2DFUSION_CHECKPOINT",str(ROOT/"weights/mv2dfusion_r50_nusc_latest.pth")))
    checkpoint=torch.load(checkpoint_path,map_location="cpu")
    state=checkpoint.get("state_dict",checkpoint)
    # Preserve OrderedDict._metadata: spconv uses version=2 to interpret weights.
    if any(k.startswith("module.") for k in state):
        raise RuntimeError("Unexpected module prefix; preserve checkpoint metadata when adapting it")
    incompatible=model.load_state_dict(state,strict=False)
    missing=set(incompatible.missing_keys)&set(dict(model.named_parameters()))
    if missing: raise RuntimeError("Missing learned parameters: "+repr(sorted(missing)))
    print("Checkpoint loaded; missing buffers:",incompatible.missing_keys,"unexpected:",incompatible.unexpected_keys,flush=True)
    del checkpoint,state
    model.CLASSES=dataset.CLASSES
    model.prev_scene_token=None
    model.fusion_bbox_head.reset_memory()
    model=MMDataParallel(model.cuda(),device_ids=[0]).eval()
    out=Path(os.environ.get("MV2D_DEMO_OUTPUT",str(ROOT/"demo_outputs")))
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(min(args.frames,len(dataset))):
        data=collate([dataset[i]],samples_per_gpu=1)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.no_grad(): result=model(return_loss=False,rescale=True,**data)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        pred=result[0].get("pts_bbox",result[0])
        count=render(dataset.data_infos[i],pred,i,out,root,args.score)
        records.append(dict(index=i,token=dataset.data_infos[i]["token"],predicted_boxes=len(pred["scores_3d"]),boxes_above_display_threshold=count,display_threshold=args.score,inference_seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
        print(records[-1],flush=True)
    digest=hashlib.sha256()
    with checkpoint_path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): digest.update(chunk)
    report=dict(status="PRETRAINED_REAL_SAMPLE_INFERENCE",model="MV2DFusion R50",dataset="nuScenes v1.0-mini",checkpoint_sha256=digest.hexdigest(),missing_learned_parameters=[],real_data=True,paper_benchmark_reproduced=False,AIS_used=False,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),frames=records,compatibility="Torch1.13.1/cu117, mmcv1.7.0, PyTorch ordinal grouping")
    (out/"inference_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
if __name__=="__main__": main()
