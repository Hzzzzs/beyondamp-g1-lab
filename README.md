# G1 AMP：5 分钟运行示例

Linux、Python 3.10 或 3.12；此示例无需 GPU 或 Isaac。策略基于 beyondAMP 训练，已在 MuJoCo 和 Isaac Sim 分别完成 24 项 20 秒平地行走测试。

```bash
git clone https://github.com/Hzzzzs/beyondamp-g1-lab.git
cd beyondamp-g1-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/cpu.txt -e '.[mujoco]'
mkdir -p examples
curl -L https://github.com/Hzzzzs/beyondamp-g1-lab/releases/download/v0.1.0/g1-amp-seed43-6000.tar.gz -o examples/g1-amp.tar.gz
echo '1ffe31c6b395a06ba4dba2bc8f84564555f906e733c9df7a4cadb06febcf0b7f  examples/g1-amp.tar.gz' | sha256sum -c -
tar -xzf examples/g1-amp.tar.gz -C examples
amp-lab inspect examples/release_seed43_review_local/bundle
amp-lab deploy examples/release_seed43_review_local/bundle --headless --seconds 20 --velocity 0.6 0 0 --report outputs/result.json
```

如果直连下载遇到网络错误，可安装并使用 GitHub CLI：`gh release download v0.1.0 --repo Hzzzzs/beyondamp-g1-lab --pattern g1-amp-seed43-6000.tar.gz --dir examples`，再运行上面的校验与解压命令。

报告中的 `completed` 应为 true、`fallen` 为 false，`mean_velocity` 第一个数约为 0.6 m/s。有桌面时去掉 `--headless` 可观看；录视频时参考[快速开始](docs/quickstart.md)。训练、双仿真、数据和限制见[项目说明](docs/quickstart.md)及[模型卡](docs/experiments/g1_amp_seed43_6000.md)。仅适用于平地仿真，未验证实机和复杂地形。示例包含上游模型许可、数据来源和训练配置，beyondAMP、robotlib作为外部依赖。

## 一键工作流（开发中）

已下载示例后可用 `./scripts/deploy.sh --sim mujoco` 运行，或使用 `./scripts/evaluate.sh --sim mujoco --gait --output outputs/my_evaluation` 生成矩阵和步态报告。训练和双仿真入口见[一键工作流说明](docs/one-command-workflows.md)；新入口已通过 GPU 短训练、断点恢复和同策略双仿真测试。

## 项目详情

以固定版本的 [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP/tree/cee88cdc0958c417e316f9452f802e25a71bc289) 为算法基础，建设从公开动作数据到 IsaacLab 训练、ONNX 导出、原生 MuJoCo 推理的工程。

**真实 AMP 训练与同一 ONNX 的 Isaac／MuJoCo 平地行走已通过。** seed43 第6000次权重在两个模拟器各24项指令／朝向测试中均完成20秒未跌倒，已组装匹配权重、数据和模型的候选包。示例下载见仓库首页；结果与产物位置见 [示例候选模型卡](docs/experiments/g1_amp_seed43_6000.md)。`outputs/cpu_smoke*` 仍是早期诊断产物。

15000次权重也已通过双环境各24项测试，指令跟踪进一步改善。新增可复现的步态诊断显示，后期策略步频和滑步有所改善，但抬脚高度仍明显低于专家动作；见 [长程比较与风格诊断](docs/experiments/g1_amp_18000.md)。AMP奖励不是动作相似度百分比。

首轮20000次训练已结束，按用户要求以19800次作优化参考。19800次双环境直行通过，但高速度步态部分指标退步；慢速数据、AMP奖励和固定专家标准化等对照结果见 [风格对照记录](docs/experiments/g1_amp_2026-11-07.md)。当前 seed43 候选采用固定专家标准化和较高偏航奖励。

首个支持目标是 **Unitree G1 29 DoF / 平地、速度指令行走 / AMP**。这里对标 unitree_rl_lab 的工程完整度；当前不承诺其所有机器人、Mimic 任务或真机功能。

| 功能 | 当前状态 |
| --- | --- |
| 固定 beyondAMP / robotlib / assetslib 提交 | 已实现 |
| CMU ASF/AMC 解析、FK、G1 有界 IK、50 Hz 数据转换 | CPU 已跑通，动作仍需质量验收 |
| beyondAMP PPO + 判别器更新、断点恢复 | 真实4096环境训练及恢复已通过，原20000次长训练已完成 |
| IsaacLab G1 AMP 环境、终止状态捕获 | 真实GPU已通过 |
| 训练端导出关节顺序、增益、观测及资产指纹 | 真实GPU导出与部署已验证 |
| ONNX 导出及 PyTorch/ONNX 数值对齐 | 真实训练权重32组对齐通过 |
| 独立 MuJoCo 推理、回放、视频、评测矩阵 | CPU 已跑通 |
| 同一权重 IsaacSim / MuJoCo 稳定行走 | 平地各24项测试通过 |
| 匹配权重、数据、模型的 Release 组装 | 候选已组装；公开分发材料待整理 |

如果只想运行最终示例，先看 [初学者快速开始](docs/quickstart.md)。部署流程只需要 CPU 推理依赖和完整部署包；示例下载见仓库首页。

## 1. 本地安装与验证

Linux，Python 3.10 或 3.12。当前使用仓库内的 editable 安装；不要直接将源码 wheel 当成独立资源包发布。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/cpu.txt -e '.[mujoco,export,dev]'
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
amp-lab fetch-upstream --install
make check test
amp-lab doctor
amp-lab model-audit --output outputs/source_model_audit.json
```

训练依赖按提交下载到 `.cache/upstream/`，不会把上游工程整份复制进本仓库。首次下载 assetslib 包含多种机器人，可能较大。CPU 推理只需要 `.[mujoco]` 和一个完整部署包，不需要 PyTorch、Isaac 或训练数据。

## 2. 准备可追溯的公开动作样本

```bash
python scripts/prepare_cmu_sample.py --output data/processed/cmu07_01
amp-lab motion validate data/processed/cmu07_01/candidate_amp.npz
amp-lab motion audit data/processed/cmu07_01/candidate_amp.npz
amp-lab motion playback data/processed/cmu07_01/candidate_amp.npz
```

无显示器时，可用 EGL 输出参考视频：

```bash
MUJOCO_GL=egl amp-lab motion playback data/processed/cmu07_01/candidate_amp.npz --video outputs/reference.mp4
```

这段视频是重定向数据的**运动学回放**，不代表策略已经学会行走。初始样本只有 2.62 秒，用于接通流程；正式训练需扩展动作数量、分离验证集并检查脚部接触和关节速度。实验性脚锁定可用 `--experimental-contact-projection` 生成对比，默认不启用。

CMU 数据保留自己的使用条款，详见 [数据说明](docs/data.md)。自有命名关节动作可经 `amp-lab motion convert` 转为 AMP 数据，输入格式见同一文档。

准备小型训练/验证片段划分：

```bash
python scripts/prepare_cmu_dataset.py --output data/processed/cmu_walk
```

训练集为 `07_01/02/03`，验证片段为 `07_06`。它们来自同一个人，规模仍小；`manifest.json` 保存每段质量报告与输出哈希。正式训练入口可传 `--motions data/processed/cmu_walk/train/*/candidate_amp.npz`，不要把 validation 目录混进专家训练集。

## 3. CPU 端到端诊断

```bash
python scripts/smoke_cpu.py --motion data/processed/cmu07_01/candidate_amp.npz --output outputs/cpu_smoke
```

这个命令用真实 G1 MuJoCo 模型和真实 CMU 动作做 3 次小网络更新，覆盖数据加载、AMP 更新、恢复、导出、部署包、推理。`report.json` 明确记录失败和跌倒。其用途是验证接口，**不是训练合格的行走策略**。

## 4. GPU 环境及正式训练

锁定组合：**Ubuntu 22.04 / RTX 4090 / Isaac Sim 4.5.0 / IsaacLab v2.1.0 / Python 3.10**。官方容器基底为 `nvcr.io/nvidia/isaac-sim:4.5.0`。IsaacLab 的 Python 扩展版本是 `0.36.21`，与发布标签 `v2.1.0` 是两种版本号。

此组合已在 RTX 4090 D 开发机完成 preflight、4096环境训练、断点恢复和同策略双环境行走验证。平台镜像准备与短实验步骤见 [算力与 GPU 启动](docs/compute.md)。

在已有 IsaacLab 2.1.0 环境中，通过其 `isaaclab.sh -p` 使用匹配的 Python 与动态库，保留镜像提供的 torch/CUDA。下面路径对应实测镜像，其他机器请改为实际 IsaacLab 安装目录：

```bash
AMP_ISAACLAB=/workspace/isaaclab
"$AMP_ISAACLAB/isaaclab.sh" -p -m pip install -c constraints/isaac45.txt -e '.[mujoco,export]'
"$AMP_ISAACLAB/isaaclab.sh" -p -m amp_lab.cli fetch-upstream --install
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/preflight_isaac.py --headless --report outputs/gpu_preflight.json
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/train.py --headless --num-envs 32 --iterations 10 --motions data/processed/cmu_walk_candidate_v1/train/*/candidate_amp.npz --output outputs/gpu_smoke
```

候选数据改进和一条命令的复现预设见 [运动学实验记录](docs/experiments/motion_2026-10-22.md)，仍需动力学验证。迁移到开发机可使用 [本地交接包](docs/compute.md#本地交接包)。

短实验通过后，使用已通过行走验证的平地预设训练。实测4096环境约占用5–6GiB显存；下面显式指定配置，避免使用仅供早期对照的默认奖励：

```bash
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/train.py --headless --num-envs 4096 --iterations 10000 --seed 42 --config configs/train_g1_rsi_speed.json --motions data/processed/cmu_walk_candidate_v1/train/*/candidate_amp.npz --output outputs/g1_amp_seed42
```

`--iterations` 是要达到的**总迭代数**。恢复时保留原配置、数据和 seed，并使用同一输出目录：

```bash
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/train.py --headless --num-envs 4096 --iterations 20000 --seed 42 --config configs/train_g1_rsi_speed.json --motions data/processed/cmu_walk_candidate_v1/train/*/candidate_amp.npz --output outputs/g1_amp_seed42 --resume outputs/g1_amp_seed42/model_10000.pt
```

恢复优化器、判别器、replay buffer、学习率与随机数状态，环境从新 episode 开始。不能把恢复后的轨迹视为连续仿真复现。

若机器断电后日志已超过最近保存的 checkpoint，保留旧目录，改用新的 `--output` 目录，并用 `--resume` 指向最近的完整 checkpoint。继续使用原数据、配置、seed 和环境数。

已验证配置保留20%默认站立起步、80%参考状态初始化，并加入根部速度AMP特征及速度误差奖励。其他配置用于开发对照，纯RSI并未解决早期站立收敛。可用以下命令检查已验证预设的参考状态初始化：

```bash
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/preflight_isaac.py --headless --config configs/train_g1_rsi_speed.json --motions data/processed/cmu_walk_candidate_v1/train/*/candidate_amp.npz --report outputs/walk_preflight.json
```

训练的 `metrics.jsonl` 记录奖励和实际 rollout 指标。绘图需安装 `.[analysis]`；查看实际前进速度和完成回合时长：

```bash
python scripts/plot_metrics.py outputs/g1_amp_seed42/metrics.jsonl --metrics rollout_forward_velocity rollout_forward_command mean_completed_episode_seconds --output outputs/training_motion.png
```

奖励提高不一定表示已向前走。用评测报告的 `mean_velocity` 对照指令，最终再用同一 ONNX 完成双环境验证。

要在训练继续时自动评测保存的检查点，在第二个终端运行：

```bash
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/evaluate_training.py outputs/g1_amp_seed42 --until 10000 --timeout 43200
```

该进程在 CPU 导出并评测每个已保存的 checkpoint，固定指令 `0.6 0 0`、每次 20 秒，报告写入训练目录的 `evaluations/model_*/mujoco.json`；不会暂停 GPU 训练。无需为中间权重反复复制完整机器人模型。`--until` 对应训练总迭代数，默认最多等待 1 小时，可用 `--timeout` 调整。

配置演变和未通过的早期对照见 [GPU 实验记录](docs/experiments/gpu_2026-10-23.md)。自动持久备份可在第二个终端运行：

```bash
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/backup_training.py outputs/g1_amp_seed42 --output /personal/amp-backups --every 500 --timeout 43200
```

备份脚本会校验读回SHA；瞬时失败会重试并记录，单份失败不会让后续备份停止。`/personal` 换成平台提供的持久挂载路径。

## 5. 导出、打包、同策略双环境验证

必须使用训练自动保存的 `deployment.yaml`；`amp-lab profile` 只生成开发用初始配置，不能替代实际训练配置。

```bash
amp-lab export outputs/g1_amp_seed42/model_10000.pt --spec outputs/g1_amp_seed42/deployment.yaml --output outputs/g1_amp_seed42/policy.onnx
amp-lab package --policy outputs/g1_amp_seed42/policy.onnx --spec outputs/g1_amp_seed42/deployment.yaml --output outputs/g1_amp_bundle
amp-lab inspect outputs/g1_amp_bundle
amp-lab check-bundle outputs/g1_amp_bundle
amp-lab deploy outputs/g1_amp_bundle --headless --seconds 20 --velocity 0.6 0 0 --report outputs/mujoco_eval.json
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/play_isaac.py outputs/g1_amp_bundle --headless --seconds 20 --velocity 0.6 0 0 --report outputs/isaac_eval.json
```

MuJoCo 可去掉 `--headless` 打开窗口，也可加 `--video outputs/policy_mujoco.mp4` 输出视频；无显示器时设置 `MUJOCO_GL=egl`。Isaac 的 `--video` 会启用渲染，已在开发机录制完整20秒行走视频；相机会跟随机器人。

```bash
python scripts/evaluate_mujoco.py outputs/g1_amp_bundle --seconds 20 --output outputs/mujoco_matrix.json
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/evaluate_isaac.py outputs/g1_amp_bundle --headless --seconds 20 --output outputs/isaac_matrix.json
```

两个模拟器使用相同矩阵，覆盖8组指令 × 3个初始朝向。每份评测带策略和配置指纹，失败不会被过滤。当前 URDF 与 MJCF 使用同一 G1 资产来源，控制语义做了显式对齐；已通过实际双环境平地评测，但不据此假定两引擎动力学完全一致。

## 6. 发布与项目状态

[发布说明](docs/release.md)列出验收与组装命令。最终示例由本项目重新训练；不会拿上游权重冒充训练结果。Release 应包含 checkpoint、ONNX、部署配置、模型/mesh、匹配动作、来源说明、评测报告和两个仿真视频。

更多实现说明见 [设计](docs/design.md)、[第三方来源](THIRD_PARTY.md) 与 [当前状态](PROJECT_STATUS.md)。
