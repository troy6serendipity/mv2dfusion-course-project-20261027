# 课程项目小样本实验协议

本实验使用真实 nuScenes Mini 验证子集，对原论文预训练模型和两种共享专家权重的基线做固定协议比较。未训练或微调模型，也未根据本批 GT 调整阈值。

## 数据与样本

固定索引为 0、7、14、21、28、35、40、48、56、64、72、80，共 12 帧，分别来自 Mini 验证集两个场景的各 6 帧。样本 token、时间戳和场景编号见 sample_manifest.json。

GT 显式使用 use_valid_flag=False，即只保留 num_lidar_pts>0 的标注。再过滤无效类别，并按 nuScenes detection_cvpr_2019 的每类 ego-distance 范围过滤 GT 与预测。范围为：汽车/卡车/公交/拖车/工程车 50m，行人/摩托/自行车 40m，锥桶/路障 30m。预测和 GT 均用 LiDAR 到 ego 的实际标定计算该范围。

每个条件均有 481 个 GT：car 271、pedestrian 158、motorcycle 24、truck 13、traffic_cone 5、bicycle 5、bus 5。trailer、construction_vehicle、barrier 在此样本中没有 GT，因此不能据其类别 recall/F1 判断性能，也没有做十类宏平均。若预测这些类别而无匹配，仍按协议计为 FP。

这些过滤规则与样本规模使结果不等于官方完整 nuScenes 评价，不能将结果写成官方 mAP、NDS 或完整数据集 ATE。

## 三种方法

1. LiDAR 专家。捕获完整 checkpoint 内 FSDv2 专家的独立 3D 输出，复制 boxes、scores、labels 后，调用官方 convert_to_fsd_anno(...,inv=True) 转成当前 nuScenes 类别、长宽和 yaw 约定。捕获函数原样返回原始 bbox_list，不改变网络后续查询生成与前向结果。
2. 简单投影融合。使用同一个 LiDAR 专家和同一次前向中独立图像专家的 2D 结果。将 LiDAR 框八角点投影到缩放裁剪后的图像坐标，使用同一处理后的 lidar2img。仅考虑八角点深度均>0.1且与图像有正交叠面积的相机。寻找同类别、2D分数>=0.1且二维包围框IoU>=0.1的检测。支持分数取最大值。匹配时 s=sqrt(sL*sI)，相机可见但无匹配时 s=0.5*sL，所有相机不可见时保持 sL。几何和类别保持 LiDAR 输出，仅改变分数。该规则可能增加或减少固定阈值下的 FP/FN，并未保证必然优于单模态。
3. MV2DFusion。使用原完整视觉、LiDAR和融合检测头的输出，使用官方 R50 checkpoint，全部参数/缓冲匹配。未添加 AIS、额外训练分支或新的学习参数。

三种方法共享 checkpoint 的专家权重。这里不是分别重新训练三个模型，也没有把相机置零条件冒充独立训练的单模态模型。

## 四种输入条件

- clean：原始预处理输入。
- image_zero：六路归一化图像张量置零，相当于模型输入空间的均值填充。不是原始 RGB 图像的纯黑填充。
- lidar_keep50：对经过官方预处理和多扫描聚合后的输入点，按每帧固定种子保留 floor(N/2) 个点，保留选中点的原顺序。
- calib_yaw2deg：外参右乘 LiDAR 坐标下 +2° yaw 矩阵，重算 lidar2img=intrinsics@extrinsics，并同步更新每帧元数据中的 intrinsics、extrinsics、lidar2img。内参值本身不加扰动。图像内容、点云位置和 GT 坐标保持原样，模拟空间标定偏差。

GT 始终来自同一份原始标注，不因点云稀疏或图像缺失重新过滤。干预写入新的输入张量/元数据副本，不修改旧项目数据文件。

每个样本和每个条件前重置 detector.prev_scene_token、test_clip_id、current_seq，并清空融合头全部时序 memory 字段。随后原网络执行一帧推理。这样不会跨样本或跨条件利用历史记忆，但也不同于论文完整时序评价。

每帧随机种子为 20261026+索引。GPU稀疏算子仍可能有极小浮点非确定性。校验发现图像缺失和标定扰动下的 LiDAR 专家与 clean 的类别、形状及评价计数一致，预测张量存在约 1e-5 量级差异，不承诺逐位一致。

## 指标与匹配

保留分数>=0.3且在相应类别范围内的预测。按类别独立计算 LiDAR 平面中心距离。距离<=2m 为有效匹配边。

匈牙利算法使用足量虚拟未匹配列：虚拟代价 1e6，非法边代价4e6，有效边代价为实际距离。该代价规模远大于一帧有效匹配距离总和，因此先最大化有效匹配数，再最小化匹配距离。没有采用可能减少匹配数量的局部最近邻贪心法。

微平均指标：
P=TP/(TP+FP)
R=TP/(TP+FN)
F1=2TP/(2TP+FP+FN)

匹配中心误差为所有 TP 的 LiDAR 平面中心距离平均值。没有匹配时记为 null。它只描述已匹配目标，不能脱离召回率单独比较。

所有逐帧计数、类别计数及聚合值均保存在 JSON/CSV。没有用 GT 选择最优显示分数或校准预测分数。

## 计时与显存

一次完整网络前向同时捕获三种方法所需的专家输出。instrumented_full_forward_seconds 包含 MMDataParallel 调用、分支捕获产生的 CPU 拷贝和 GPU 同步，不包含数据读取/组装、模型加载、简单融合后处理、评价和写盘。

shared_full_forward_s_mean 是三种方法共享的这次前向测量，不是三种方法各自独立端到端延迟。simple_fusion_postprocess_seconds 单独记录 CPU 规则后处理时间。

peak_allocated_MiB 为 PyTorch 峰值分配，不包含所有系统或第三方 GPU 分配。条件按顺序运行，时延还可能受机器并发负载、缓存和功耗状态影响，不据此把条件间时延差异全部归因于退化方式。

## 主要实测结果

clean：LiDAR专家 P82.70%、R90.44%、F1 86.40%；简单投影融合 P72.29%、R81.91%、F1 76.80%；MV2DFusion P71.27%、R95.43%、F1 81.60%。

在此协议下，MV2DFusion 比专家增加24个TP、减少24个FN，同时增加94个FP，体现固定阈值下的召回与精度权衡，不能写成全面优于单模态。

image_zero 下 MV2DFusion 的召回从95.43%降至80.87%，F1却升至84.38%，原因是 FP 同时减少。不能只展示 F1 就宣称相机缺失提高了模型能力。

点云保留50%时，MV2DFusion召回93.56%、F1 81.23%。2°标定偏差时，召回90.02%、F1 76.57%，已匹配中心误差从0.232m增至0.409m。结果仅适用于该小样本设置。

## 运行与输出

在仓库根目录执行独立烟雾验证：
~~~bash
bash scripts/run_benchmark.sh --smoke
~~~

完整固定协议：
~~~bash
bash scripts/run_benchmark.sh
~~~

默认输出在仓库 experiment_outputs/benchmark_main；--output 可指定新目录。重复运行会覆盖主汇总和同名预测；逐帧 JSONL 在新运行开始时清空。

通过 config/local.env.sh 或环境变量 MV2DFUSION_UPSTREAM、NUSCENES_ROOT、MV2DFUSION_CHECKPOINT 分别指定官方源码、数据根目录和权重；MV2D_ENV_PREFIX 指定 Python 环境，MV2D_CACHE_ROOT 指定缓存。详见 scripts/env.sh 与 config/local.env.example.sh。

本 results/ 仅保存原始实验的汇总、协议与样本清单；以下逐帧与 NPZ 输出由完整实验脚本生成，不随 Git 分发。

重要输出：
- benchmark_summary.json / benchmark_summary.csv：12行条件与方法汇总。
- per_frame_results.json / per_frame_results.jsonl：48次前向、144组方法指标。
- per_class_results.json：类别细分。
- sample_manifest.json：样本清单。
- gt_statistics.json：GT过滤与类别分布。
- protocol.json：机器可读参数。
- benchmark_validation.json：文件、数值和独立性检查。
- predictions/：所有条件、所有样本、三种方法的144个原始预测NPZ。

NPZ 的 boxes 使用标准 LiDARInstance3DBoxes.tensor 约定，底部中心原点，列为 x,y,z,dx,dy,dz,yaw,vx,vy；scores、labels 为预测分数和本次配置类别编号。可据此绘制 clean 与退化条件的对比。

## 使用限制

仅12帧、2场景、7个有GT类别，样本之间存在场景相关性；没有跨地点统计推断。固定0.3阈值未校准三个输出的分数分布，故P/R/F1不能直接替代官方AP。GT只保留有LiDAR命中的目标，评价定义也应连同结果报告。没有重新训练、没有独立camera-only三维检测器、没有AIS或真实海事三模态数据。
