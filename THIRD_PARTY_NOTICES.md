# 第三方资源与权利说明

本文件记录课程复现使用的第三方来源，不替代原始许可。仓库提供本次编写的运行入口、兼容改动、评价脚本和汇总结果；官方完整源码、模型权重、nuScenes 原始数据、原始照片和本机运行环境不随本仓库分发。

## MV2DFusion 论文与官方实现

论文作者：Zitian Wang、Zehao Huang、Yulu Gao、Naiyan Wang、Si Liu。

- [论文](https://arxiv.org/abs/2408.05945)
- [正式出版 DOI](https://doi.org/10.1109/TPAMI.2025.3609348)
- [官方源码](https://github.com/wangzt-halo/MV2DFusion)
- 本次固定提交：[e4bc2d93f4009ce7210188da79860730676398b6](https://github.com/wangzt-halo/MV2DFusion/tree/e4bc2d93f4009ce7210188da79860730676398b6)

本包通过单独克隆取得该源码，并以兼容补丁适配已验证推理环境。固定提交的仓库根目录未见独立 LICENSE 文件；不能据其公开可见就推定完整源码可按 Apache-2.0、MIT 或其他统一许可再分发。使用上游及其各源文件时，应保留原作者声明并核对相应许可，必要时联系权利人。本包没有替上游增加授权。

## 预训练权重

使用作者公开的 MV2DFusion-R50 nuScenes checkpoint，来源为[作者 R50 目录](https://drive.google.com/drive/folders/1UEB-ynVrprZrUzY6UWzEZFPjaF-FEzpx)，入口同时见[官方 README](https://github.com/wangzt-halo/MV2DFusion)。

权重由使用者自行从原来源获取，下载脚本不产生新的再分发授权。未确认模型权重具有适用于所有用途的独立宽松许可证；使用时应核对作者说明与训练数据条款。仓库中的 SHA-256 仅用于本次文件指纹核对。

## nuScenes 数据

数据来源与归属：nuScenes 数据集及其原始权利人，项目网站 [nuscenes.org](https://www.nuscenes.org/nuscenes)。本次使用 nuScenes Mini，不重新分发其图片、点云、标注压缩包或可识别原始图像。

nuScenes 数据按 CC BY-NC-SA 4.0 及[官网使用条款](https://www.nuscenes.org/terms-of-use)供非商业研究使用；许可原文见 [Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/)。使用者应自行阅读并接受官方当前条款，包括署名、非商业用途及适用的同方式共享要求；商业用途按官方另行许可办理。数据集许可与 devkit 软件许可应分别处理。

数据引用：

> Caesar, H. et al. nuScenes: A multimodal dataset for autonomous driving. CVPR, 2020.

本包的样本 token、计数与评价汇总用于说明实验可核验性，不构成对原始数据的额外授权。

## 主要软件依赖

依赖由环境安装工具单独获取，不随本包包含其完整源码或二进制分发。下列链接是相应原始项目及其许可入口；具体版本的 LICENSE/NOTICE 和源文件声明优先：

| 资源 | 来源与许可入口 |
|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| OpenMMLab MMDetection3D | [mmdetection3d LICENSE](https://github.com/open-mmlab/mmdetection3d/blob/main/LICENSE) |
| MMCV | [mmcv LICENSE](https://github.com/open-mmlab/mmcv/blob/main/LICENSE) |
| MMDetection | [mmdetection LICENSE](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE) |
| nuScenes-devkit | [nuscenes-devkit LICENSE](https://github.com/nutonomy/nuscenes-devkit/blob/master/LICENSE) |
| spconv | [traveller59/spconv](https://github.com/traveller59/spconv) |
| FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| torch-scatter | [rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) |

MV2DFusion 还依赖或借鉴 OpenMMLab、SST、Far3D 等项目，完整来源请参阅官方仓库中的致谢、导入模块与具体文件声明。本包兼容处理不改变这些资源的权利归属。

发布报告、PPT 或可视化时，应保留论文与 nuScenes 来源标注；不要将第三方模型、数据或作者论文结果写成本课程项目原创成果。
