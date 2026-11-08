# 算力与首次 GPU 实验

已在 RTX 4090 D 开发机完成 GPU 启动、短训练、4096 环境训练及双模拟器平地验证。预训练示例见[仓库首页](../README.md)；没有为本项目提交 NSCC 作业。

## 目标镜像

| 项目 | 固定目标 |
| --- | --- |
| GPU | RTX 4090 24 GB |
| 系统 | Ubuntu 22.04 |
| Isaac Sim | 4.5.0 |
| IsaacLab | Git 标签 v2.1.0，commit `21f7136325136ca3f6ca4e0a8125edffe5c24f7e` |
| Python | 3.10，使用 IsaacLab 环境的解释器 |
| 容器基底 | `nvcr.io/nvidia/isaac-sim:4.5.0` |

建议 32 GB 以上系统内存、100 GB 以上持久盘。这是工程容量建议。平台提供 2.1.0 即可，不需要 2.1.1。不要将 Isaac Gym Preview 4 当成 Isaac Sim 4.5.0。

依据：[v2.1.0 README](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/README.md)、[Docker 配置](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/docker/.env.base)。IsaacLab 扩展元数据版本为 `0.36.21`；该数字不是错误安装了 0.x 发布版。

实际 RTX 4090 D / Ubuntu 22.04.5 / Python 3.10.15 镜像已通过 preflight；显卡驱动 535.154.05，Isaac 内实际加载 torch 2.5.1+cu118。IsaacLab 的 source/apps 对照固定提交核验 807 个文件，平台仅将 `utils/assets.py` 的 Nucleus 资源根目录改为 `/root/Assets/Isaac/4.5`，无其他差异。该改动已声明保留；本项目 G1 采用独立锁定的本地 URDF/mesh 路径。

## 在平台镜像上准备

1. 确认镜像已安装 Isaac Sim 4.5.0 及 IsaacLab 2.1.0，并启用其 Python 环境。若镜像只有 Sim，用官方 Lab v2.1.0 安装流程补齐。无需为了本项目更新到 5.x。
2. 将工程和已生成的候选数据同步到持久盘。保留已有 torch、torchvision、CUDA 和 NVIDIA 驱动组合；CPU 安装命令中的 `+cpu` 包绝不能覆盖 GPU 环境。
3. 安装本工程、锁定上游并运行 `amp-lab doctor --training`。所有命令用该镜像的 Python，不要误用系统 Python。
4. 运行 `scripts/preflight_isaac.py`。实际检查 Python、Lab 扩展和 Sim 版本；4 个环境、10 步、policy 观测与部署 builder 数值对齐、rollout 有限值。
5. 32 环境 × 10 次 AMP 更新，检查 checkpoint、判别器 loss、GPU 使用和出口文件。
6. 再从 1024 环境开始训练，依据实测显存、samples/s 与收敛调整并行数。扩展到 2048/4096 之前记录对比，不能事先保证 4090 可承受任意数量。

GPU 安装使用 `constraints/isaac45.txt`，保留 IsaacLab 所要求的 ONNX 1.16.1 和镜像 torch/CUDA；`constraints/cpu.txt` 中的 ONNX 1.17.0 用于独立 CPU 环境。首次 URDF 转换可能在线补全 Isaac 扩展，已安装 Sim 不代表这一步不需要联网。

## 沿用 NSCC 的实验组织方式

训练工作参考本机 `nscc-hpc` skill：先做短实验，项目独立环境、持久化代码版本/数据清单/配置/日志/checkpoint，先量吞吐再扩大申请；SSH 断开后先检查原进程，避免重复开训练。

NSCC 历史记录中的 ASPIRE 2A A100 资源不应直接作为 IsaacSim 替代。其 GPU 图形能力、Vulkan/渲染与 Isaac 支持需单独核实；PyTorch CUDA 可用不等于 Isaac 环境可用。目前采用用户能够提供的 RTX 4090 路线。若以后使用 NSCC，再按 skill 核对当前权限、PBS 队列与 scratch，登录节点只做轻量管理，计算任务进 PBS。

实验记录见 [seed43 模型卡](experiments/g1_amp_seed43_6000.md)；不要将 SSH 密码或其他项目的认证材料写入仓库。

## 本地交接包

完成本地诊断后，可把已提交的源码、候选数据及锁定依赖整理到私人开发传输包，减少租用机器后的重复下载。脚本不会连接平台或申请实例：

```bash
python scripts/prepare_gpu_handoff.py --dataset data/processed/cmu_walk_candidate_v1/manifest.json --include-upstream --output outputs/gpu_handoff.tar.gz
```

要求主项目无未提交修改，上游 HEAD 与锁文件一致且无 tracked 修改。只收录 Git 跟踪的工程文件、指定数据目录和显式包含的上游源码/其 Git 元数据，不收录 `.venv`、现有训练权重或 `outputs/`。原始 CMU 文件未包含，转换数据及来源报告已包含；如需重新重定向，按数据文档下载哈希锁定原始文件。

在目标 Ubuntu 上用 GNU tar 解压，例如 `tar -xzf gpu_handoff_v2.tar.gz`，保留上游符号链接的精确目标。一些解压库会去掉目录链接末尾斜杠，使 Git 误判上游被修改。

包内 `GPU_HANDOFF.md` 给出安装、4 环境 preflight、32 环境 × 10 次更新的命令，`GPU_HANDOFF.json` 记录代码提交与数据指纹，`GPU_HANDOFF.sha256` 提供逐文件校验。安装 Python 依赖仍需软件源网络。这个包不是公开 Release，没有可行走权重，不代替最终许可和发布审核。
