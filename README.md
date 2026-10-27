# MV2DFusion 复现代码包

基于作者公开的 MV2DFusion-R50 完整预训练模型，在 Windows + WSL2 上完成真实 nuScenes Mini 推理、3D 检测可视化及固定小样本鲁棒性比较。代码包提供可移植运行入口、兼容补丁、实验协议与实测汇总；未训练或微调模型。

论文：[MV2DFusion: Leveraging Modality-Specific Object Semantics for Multi-Modal 3D Detection](https://arxiv.org/abs/2408.05945)。作者：Zitian Wang、Zehao Huang、Yulu Gao、Naiyan Wang、Si Liu。正式出版信息见 [IEEE DOI](https://doi.org/10.1109/TPAMI.2025.3609348)。

## 任务、传感器与融合层次

任务是在道路场景中预测十类目标的三维位置、尺寸、朝向和类别。nuScenes 是多传感器实采数据集，包含六路相机、LiDAR、雷达等传感器；本次模型输入使用六路相机图像和 LiDAR，点云预处理聚合当前及历史扫描，位姿与标定用于坐标变换。本复现没有接入雷达或 AIS，也没有使用海事数据。

MV2DFusion 先由图像与点云专家提取目标语义，再在目标查询层进行跨模态交互。图像查询包含深度不确定性信息，LiDAR 查询携带三维几何信息；属于目标/查询层融合，不能描述为原始数据直接拼接。方法和实现以[作者仓库](https://github.com/wangzt-halo/MV2DFusion)为准。

本包比较三种输出，均共享同一 checkpoint 的专家权重：

| 方法 | 实现 |
|---|---|
| LiDAR 专家 | 捕获模型内部 FSDv2 的三维输出；转换前复制框、分数和类别，保持后续融合前向不变 |
| 简单投影融合 | 将 LiDAR 框投影到相机，与同类二维检测做 IoU 门控，仅调整 LiDAR 分数 |
| MV2DFusion | 作者完整模型的融合检测输出 |

简单融合匹配要求二维分数 ≥0.1、投影 IoU ≥0.1；匹配后分数为 sqrt(sL × sI)，相机可见但未匹配时为 0.5 × sL，所有相机不可见时保持原分数。它没有新增训练参数，也不保证优于单模态。详细坐标约定、匹配和输入干预见 [实验协议](results/PROTOCOL.md)。

## 来源、数据与权重

- 官方源码：[wangzt-halo/MV2DFusion](https://github.com/wangzt-halo/MV2DFusion)，固定版本为 [e4bc2d93f4009ce7210188da79860730676398b6](https://github.com/wangzt-halo/MV2DFusion/tree/e4bc2d93f4009ce7210188da79860730676398b6)。本仓库只提供固定版本下载入口与兼容补丁，不附带官方完整源码。
- 数据：[nuScenes 官网](https://www.nuscenes.org/nuscenes)、[Mini 官方下载](https://www.nuscenes.org/data/v1.0-mini.tgz)。Mini 压缩包约 4.17 GB，解压约 5.1 GiB；共 404 帧，官方划分含 323 个训练样本和 81 个验证样本。本次只做验证推理。
- 数据许可：仅按官方许可开展非商业课程研究。nuScenes 数据适用 CC BY-NC-SA 4.0 及官网条款，获取和使用前请阅读[官方使用条款](https://www.nuscenes.org/terms-of-use)与 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。数据与 devkit 的软件许可不同。
- 权重：[作者公开 R50 下载目录](https://drive.google.com/drive/folders/1UEB-ynVrprZrUzY6UWzEZFPjaF-FEzpx)，也可从[作者总权重目录](https://drive.google.com/drive/folders/1gnBZdzRJbvR5-wWpNXQWo5Qhl1No6dqS)进入。Google Drive 文件 ID 为 1Sf5jeH8CMHauxJRRNuwYd7aXy2n4dlEo；保存为 mv2dfusion_r50_nusc_latest.pth。
- 已实测权重文件大小：881149879 字节；本地 SHA-256：0b78315efcd6ab9a0b6f6a76467d01e2381f3c6a9efb3b4a5879a92fe4a867b0。这是本次下载文件的指纹，不是与作者公布校验和的比对结果。

原始照片、点云、权重和运行环境均不随 Git 仓库分发。第三方代码与模型权利说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已验证环境

本次实际机器为 Windows + Ubuntu 26.04 WSL2，NVIDIA RTX 4060 Laptop GPU 8 GB。模型依赖旧版 OpenMMLab，实际环境与作者推荐环境存在以下差异：

| 组件 | 作者 README | 本次实测 |
|---|---|---|
| Python | 3.8.10 | 3.8.20 |
| PyTorch | 1.11.0 | 1.13.1+cu117 |
| CUDA | 11.3 | 编译器 11.8，PyTorch runtime 11.7 |
| MMCV | 1.6.1 | mmcv-full 1.7.0 |
| MMDetection / MMDetection3D | 2.25.1 / 1.0.0rc4 | 2.25.1 / 1.0.0rc4 |
| spconv / FlashAttention | 2.3.6 / 1.0.2 | spconv-cu117 2.3.6 / FlashAttention 1.0.2 |

其他关键组件包括 torchvision 0.14.1+cu117、gcc 11.4、mmcls 0.23.2、mmsegmentation 0.30.0、torch-scatter 2.1.1+pt113cu117、NumPy 1.23.5、SciPy 1.10.1。完整快照见 [env/](env/) 下的两份锁文件；它们记录已验证环境，不保证在任意平台上直接重建。

兼容处理包括用 PyTorch 实现组内索引算子，以及只运行 nuScenes 时跳过可选 AV2 数据集注册。没有新增可学习参数。加载完整 checkpoint 时保留 OrderedDict._metadata，避免旧版 spconv 卷积权重被错误转换。历史依赖的声明约束仍可能触发包管理器警告；本包验证范围是所列预训练推理与小样本实验。

## 在 WSL2 中运行

下列命令在 WSL 的 bash 终端执行。将整个仓库、WSL 发行版虚拟磁盘和运行目录放在容量足够的目标磁盘上。默认环境、源码、数据、权重、缓存均位于仓库目录下；Windows 已有 WSL 虚拟磁盘的存放位置需由使用者设置。

### 1. 配置路径与环境

进入仓库根目录后，首次新建环境需要 Linux Miniconda/Conda 与 git；请将 Conda 安装在目标磁盘对应的 Linux 文件系统，并按需在 config/local.env.sh 中设置 CONDA_EXE。环境首次重建脚本经过语法检查；本次验证复用了已有环境，未在另一台空白机器上重装。首次新建环境使用：

~~~bash
bash scripts/setup_wsl.sh
bash scripts/clone_upstream.sh
~~~

默认目录为 .runtime/env、upstream/MV2DFusion、data/nuscenes、weights。编译旧版 CUDA 扩展需要可用 NVIDIA 驱动、CUDA 工具链、gcc 及足够磁盘空间；首次安装会联网下载依赖。

已有环境和数据时，先创建不进入 Git 的 config/local.env.sh，指定绝对路径，随后直接复用，不必重新安装或下载：

~~~bash
mkdir -p config
cat > config/local.env.sh <<'EOF'
export MV2D_ENV_PREFIX="/path/to/existing/env"
export NUSCENES_ROOT="/path/to/data/nuscenes"
export MV2DFUSION_UPSTREAM="/path/to/MV2DFusion"
export MV2DFUSION_CHECKPOINT="/path/to/mv2dfusion_r50_nusc_latest.pth"
EOF
source scripts/env.sh
~~~

环境锁快照如包含本地构建路径，重建时应替换为当前机器路径；不要复制他人机器的绝对路径。配置文件中不要加入账号密码或访问令牌。

### 2. 下载数据与权重

~~~bash
source scripts/env.sh
python scripts/download_data.py --accept-license
python scripts/download_weights.py
bash scripts/prepare_data.sh
~~~

请先阅读上述数据许可，确认同意后再使用 --accept-license。数据脚本从官方 Mini 地址获取数据，权重脚本从作者公开资源获取 R50 checkpoint；已有有效文件应复用。若官网要求注册、登录或下载确认，请在官网完成授权并将下载文件放入配置目录，再执行后续步骤。

数据根目录需要 samples/、sweeps/、maps/ 和 v1.0-mini/。prepare_data.sh 生成推理所需 info 文件，不修改原始标注。完整训练集与测试集不属于本次下载范围。

### 3. 跑通一个真实样本并生成效果图

~~~bash
bash scripts/run_demo.sh --frames 1
~~~

新入口的本机实测记录见 [demo_validation.json](results/demo_validation.json)：单帧真实推理输出 300 个框，0.3 阈值显示 42 个，前向约 3.900 s、PyTorch 峰值约 6557 MiB；完整权重加载缺失和意外键均为空。此 Demo 的首次前向时间与批量实验计时口径、运行条件不同，不混合比较。

默认本地输出目录为 demo_outputs/。保存三维预测及可视化；仓库不预置原始照片。显存低于 8 GB 的显卡可能需要进一步缩减可视化或输入规模。该入口使用作者完整 R50 模型，不能用随机框或替代模型输出代表复现结果。

### 4. 运行固定小样本实验

先做独立接口检查，再运行完整协议：

~~~bash
bash scripts/run_benchmark.sh --smoke
bash scripts/run_benchmark.sh
~~~

完整运行包含 12 帧 × 4 个条件 = 48 次完整模型前向；每次同时取得三种方法的输出，共 144 组评价。固定验证索引为 0、7、14、21、28、35、40、48、56、64、72、80，来自两个场景的各六帧。每个样本、每个条件前重置模型时序状态，避免跨样本记忆；因此也不同于论文完整时序评价。

四个条件为 clean、image_zero、lidar_keep50、calib_yaw2deg，分别表示原始输入、六路归一化图像置零、预处理聚合点云随机保留 50%、LiDAR 到相机外参右乘 +2° yaw 并同步重算投影矩阵。GT 不随输入干预改变。脚本参数与每个输出文件含义见 [PROTOCOL.md](results/PROTOCOL.md)；可用 --output 指向新目录，避免覆盖已有运行。

## 自定义指标与真实结果

预测分数阈值固定为 0.3，同类别目标按 LiDAR 平面中心距离 ≤2 m 进行一对一匹配。带大代价虚拟列的匈牙利匹配优先最大化有效匹配数，再最小化距离。报告微平均 P、R、F1，以及已匹配目标的平均平面中心误差。

GT 使用 num_lidar_pts>0，并与预测一起按官方每类距离范围过滤。每个条件有 481 个 GT；七类有 GT，三类无 GT，不做十类宏平均；无 GT 类别的误检仍计为 FP。完整样本与统计见 [sample_manifest.json](results/sample_manifest.json)、[gt_statistics.json](results/gt_statistics.json)。

**以下数值是本机实际运行结果，既不是官方 mAP/NDS，也不是完整验证集复现分数。** 本协议只有单一距离阈值与分数阈值，未计算 AP 曲线；匹配中心误差也不能标成官方 ATE。

| 输入条件 | 方法 | P (%) | R (%) | F1 (%) | 匹配中心误差 (m) |
|---|---|---:|---:|---:|---:|
| clean | LiDAR 专家 | 82.70 | 90.44 | 86.40 | 0.222 |
| clean | 简单投影融合 | 72.29 | 81.91 | 76.80 | 0.226 |
| clean | MV2DFusion | 71.27 | 95.43 | 81.60 | 0.232 |
| image_zero | LiDAR 专家 | 82.70 | 90.44 | 86.40 | 0.222 |
| image_zero | 简单投影融合 | 98.38 | 63.20 | 76.96 | 0.157 |
| image_zero | MV2DFusion | 88.21 | 80.87 | 84.38 | 0.201 |
| lidar_keep50 | LiDAR 专家 | 84.10 | 83.58 | 83.84 | 0.244 |
| lidar_keep50 | 简单投影融合 | 70.04 | 76.30 | 73.03 | 0.254 |
| lidar_keep50 | MV2DFusion | 71.77 | 93.56 | 81.23 | 0.257 |
| calib_yaw2deg | LiDAR 专家 | 82.70 | 90.44 | 86.40 | 0.222 |
| calib_yaw2deg | 简单投影融合 | 71.66 | 81.50 | 76.26 | 0.223 |
| calib_yaw2deg | MV2DFusion | 66.62 | 90.02 | 76.57 | 0.409 |

完整机器可读数据：[JSON](results/benchmark_summary.json)、[CSV](results/benchmark_summary.csv)。

clean 下 MV2DFusion 比 LiDAR 专家增加 24 个 TP，同时增加 94 个 FP：召回提高 4.99 个百分点，F1 反而更低。因此不能据此宣称融合全面胜出。图像置零后，MV2DFusion 的召回下降约 14.6 个百分点；F1 上升来自误检同时减少，不能只用 F1 宣称缺失图像提高了能力。2° 标定偏差下中心误差由 0.232 m 增至 0.409 m，表明该设置对几何对齐较敏感。

峰值 PyTorch 分配约 6625 MiB（约 6.47 GiB）。clean 条件共享完整前向均值约 0.957 s，其余条件约 4.790、6.554、6.617 s。这个时间包含捕获专家输出的 CPU 拷贝与 GPU 同步，不包含数据读取、模型加载、规则融合、评价和写盘；三个方法共享一次前向，不代表各自独立运行时延。条件按顺序运行，机器并发负载、缓存和功耗可能混杂，不应把所有时延差异归因于退化类型。

## 复现范围与局限

本次完成的是预训练推理与小样本条件比较，不包括重新训练、全量 nuScenes mAP/NDS、独立训练的 camera-only 三维检测器、AIS 融合或海事迁移。两个场景的 12 帧存在场景相关性，不能支撑整体基准排名或统计显著性结论。固定 0.3 阈值未校准三个输出的分数分布；GT 仅保留 LiDAR 命中的目标，结果应连同此定义解读。

GPU 稀疏算子可能产生约 1e-5 量级的浮点差异；固定种子不等于逐位复现。原始数据、权重及软件依赖请从相应官方来源取得并遵守其条款。报告中请区分作者论文结果、本包实测结果与后续设想。
