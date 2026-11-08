# beyondAMP G1 Lab

面向初学者的 Unitree G1 强化学习实验仓库：使用公开动作数据和 [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP) 训练 AMP 行走策略，并将同一个策略部署到 MuJoCo 和 Isaac Sim。

**第一次接触？先运行预训练示例。** 只需 Linux 和 Python，不需要 GPU、Isaac 或重新训练。想自己训练时，再准备 GPU 环境。

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 看机器人行走 | [运行预训练示例](#1-运行预训练示例无需-gpu) |
| 自己训练、续训和部署 | [训练自己的策略](#2-训练自己的策略需要-gpu) |
| 评估行走效果 | [评测与结果解读](#3-评测与结果解读) |
| 排查训练或部署问题 | [训练与调试经验](#4-训练与调试经验) |
| 替换参考动作 | [动作数据与重定向](docs/data.md) |

## 已验证的效果和适用范围

公开示例为本项目从头训练的 G1 29 自由度平地速度控制策略。相同 ONNX 在 MuJoCo 和 Isaac Sim 中各完成 **24 个指令与初始朝向组合，每项 20 秒均未跌倒**；最大平面速度 RMSE 分别为 0.05954 / 0.05807 m/s，最大偏航角速度 RMSE 分别为 0.12158 / 0.07242 rad/s。详见[模型卡](docs/experiments/g1_amp_seed43_6000.md)。

目前支持平地行走研究，**尚未验证真机和复杂地形**；本仓库也不承诺覆盖 unitree_rl_lab 的全部机器人和 Mimic 功能。动作数据、模型及上游代码各有自己的来源与许可，参见[第三方说明](THIRD_PARTY.md)和[分发审查](docs/distribution-review.md)。

## 1. 运行预训练示例（无需 GPU）

在 Linux 上准备 Git、Python 3.10 或 3.12，然后运行：

```bash
git clone https://github.com/Hzzzzs/beyondamp-g1-lab.git
cd beyondamp-g1-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/cpu.txt -e '.[mujoco]'

mkdir -p examples
curl -fL https://github.com/Hzzzzs/beyondamp-g1-lab/releases/download/v0.1.0/g1-amp-seed43-6000.tar.gz -o examples/g1-amp.tar.gz
echo '1ffe31c6b395a06ba4dba2bc8f84564555f906e733c9df7a4cadb06febcf0b7f  examples/g1-amp.tar.gz' | sha256sum -c -
tar -xzf examples/g1-amp.tar.gz -C examples

./scripts/doctor.sh
./scripts/deploy.sh --sim mujoco --seconds 20 --velocity 0.6 0 0 --report outputs/demo.json
```

成功时，`outputs/demo.json` 中 `completed` 为 `true`、`fallen` 为 `false`，平均前进速度约为 0.6 m/s。速度参数依次为前进速度、左向速度（m/s）和左转角速度（rad/s）。有桌面环境时加入 `--show` 可以打开窗口；默认无窗口运行。视频和更多选项见[快速开始](docs/quickstart.md)。

若 `curl` 下载失败，可以先安装并登录 GitHub CLI，然后用下面两行替代上面的下载命令，再执行 SHA256 校验和解压：

```bash
gh release download v0.1.0 --repo Hzzzzs/beyondamp-g1-lab --pattern g1-amp-seed43-6000.tar.gz --dir examples
mv examples/g1-amp-seed43-6000.tar.gz examples/g1-amp.tar.gz
```

示例包自带策略、模型、数据和配置；只运行示例不需要下载训练用的上游依赖。`doctor.sh` 中训练资产未找到不影响使用自带机器人模型的示例。

## 2. 训练自己的策略（需要 GPU）

已验证组合：**Ubuntu 22.04、RTX 4090 D、Isaac Sim 4.5.0、IsaacLab 2.1.0、Python 3.10**。不同机器的安装路径可能不同；保留镜像自带的 PyTorch/CUDA，不要在 Isaac 环境中安装 CPU 版 PyTorch。

先完成上面的示例下载。以下假设 IsaacLab 位于 `/workspace/isaaclab`，请改为自己的路径：

```bash
export AMP_ISAACLAB=/workspace/isaaclab
"$AMP_ISAACLAB/isaaclab.sh" -p -m pip install -c constraints/isaac45.txt -e '.[mujoco,export]'
"$AMP_ISAACLAB/isaaclab.sh" -p -m amp_lab.cli fetch-upstream --install
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/preflight_isaac.py --headless --report outputs/gpu_preflight.json
```

训练使用 IsaacLab 的 Python。导出 checkpoint 时，工作流也需要能导入 PyTorch 和锁定的 `rsl_rl_amp`；可将 `AMP_PYTHON` 指向 Isaac Sim 的 Python 启动脚本（请按机器实际路径修改）：

```bash
export AMP_PYTHON=/isaac-sim/python.sh
./scripts/doctor.sh --training

# 先做集成检查：32 个环境，训练至第 10 次迭代
./scripts/train.sh --robot g1 --motion cmu_walk --num-envs 32 --iterations 10 --output outputs/smoke

# 正式训练：默认 4096 个环境、seed 43、训练至第 6000 次迭代
./scripts/train.sh --robot g1 --motion cmu_walk
```

10 次短训练用于确认流程，不能当作学会行走的策略。正式训练默认使用示例包中的 CMU `07_01/02/03` 数据及 `configs/train_g1_expert_fixed_yaw1.json`，日志为 `logs/g1_amp.log`，权重和训练配置默认保存在 `outputs/g1_amp/`。正常结束后自动导出包含 checkpoint、ONNX 和配置的部署 artifact，位于 `outputs/artifacts/`。

中断后从同一输出目录的最新 checkpoint 继续；`--iterations` 是目标**总迭代数**：

```bash
./scripts/train.sh --robot g1 --motion cmu_walk --resume latest --iterations 6000 --output outputs/g1_amp
```

续训应保持相同的训练配置、数据、seed 和环境数。优化器等状态从 checkpoint 恢复，仿真环境从新 episode 开始。路径、导出依赖和其他选项见[一键工作流](docs/one-command-workflows.md)和[GPU 环境说明](docs/compute.md)。

## 3. 评测与结果解读

```bash
# 评测预训练示例，生成矩阵与步态报告
./scripts/evaluate.sh --sim mujoco --gait --output outputs/example_eval

# 评测刚训练的最新 checkpoint；两个模拟器使用同一策略
./scripts/evaluate.sh --checkpoint latest --run-dir outputs/g1_amp --sim both --gait --output outputs/my_eval

# 部署训练结果；将 mujoco 改为 isaacsim 可在 Isaac 中运行
./scripts/deploy.sh --sim mujoco --checkpoint latest --run-dir outputs/g1_amp --show
```

`latest` 只在指定的 `--run-dir` 中按迭代数选择 checkpoint。每次评测使用新的输出目录；`summary.json` 汇总完成情况及速度、偏航误差，步态报告单独保存。默认每项 20 秒。`--sim both` 需要已配置的 Isaac 环境。

看结果时，应同时检查是否跌倒、实际速度与指令的误差、转向，以及视频中的脚部动作。当前脚部支撑与滑动指标是**高度和运动学代理**，不是接触力测量；AMP 奖励也不是动作相似度百分比。参考动作回放只是数据质量检查，不代表策略已经学会行走。

## 4. 训练与调试经验

以下经验来自本项目的实验，具体效果仍应以自己的数据和评测为准：

- **先检查数据，再长训。** 查看重定向动作是否穿地、滑步、关节越界或速度突变；参考数据质量会影响训练。示例使用同一被试的三段 CMU 行走数据，规模较小，不能据此声称跨人泛化。[数据处理与审计](docs/data.md)
- **先跑短实验，再扩大环境数。** 先检查 GPU、模型加载、checkpoint 与导出是否正常；32 环境 × 10 次只是接线检查。训练时再根据显存和吞吐调整环境数，避免一开始投入大量机时。
- **不要只盯奖励或“没摔倒”。** 机器人原地站稳也可能走满 20 秒。对比实际与目标速度、转向误差和视频，再看抬脚、滑步等步态代理。
- **每次只改少量因素并保留基线。** 我们对固定专家统计和偏航奖励做过对照；例如 seed42 的 3000 次实验中，将 yaw 权重从 0.2 调到 1.0 后，两个模拟器的最大偏航 RMSE 从 0.46085 / 0.39135 降至 0.12211 / 0.09831 rad/s，但部分抬脚高度下降。这是特定实验结果，不保证其他数据也一样。[实验记录](docs/experiments/g1_amp_2026-11-07.md)
- **以已保存的 checkpoint 为准。** 日志迭代数可能高于最后一次保存的权重；断电后未保存的更新无法恢复。把 checkpoint、训练配置、数据、日志一起备份到持久存储。
- **部署使用配套资产。** 训练生成的 `deployment.yaml` 记录观测、动作、控制参数和关节约定；权重、配置和机器人模型要匹配，不要手动重排关节来绕过检查。用同一 ONNX 分别验证两个模拟器。

| 现象 | 建议先检查 |
| --- | --- |
| 下载或校验失败 | 网络连接、Release 文件是否完整；重新下载后核对 SHA256。 |
| 找不到 `amp-lab` 或 Python 依赖 | 是否激活了正确的 `.venv`；训练和导出是否使用包含相应依赖的解释器。 |
| 找不到 IsaacLab / `rsl_rl_amp` | 检查 `AMP_ISAACLAB`、`AMP_PYTHON` 和上游依赖安装位置，运行 `./scripts/doctor.sh --training`。 |
| 导出时模型或关节不匹配 | 使用该训练的 `deployment.yaml` 和匹配的机器人资产；详见[一键工作流](docs/one-command-workflows.md)。 |
| 无法打开窗口 | 先使用默认无窗口模式；录制视频需要正确配置渲染环境。 |
| 训练奖励上涨但不前进 | 对照 rollout 实际速度、目标指令和评测视频；不要只看奖励。 |

## 更多资料

- [一键训练、续训、导出与评测](docs/one-command-workflows.md) · [GPU 环境与计算资源](docs/compute.md)
- [动作数据、重定向与质量检查](docs/data.md) · [示例模型卡和评测](docs/experiments/g1_amp_seed43_6000.md)
- [项目设计](docs/design.md) · [发布说明](docs/release.md) · [第三方来源与许可](THIRD_PARTY.md)

代码使用 MIT 许可；第三方代码、机器人资产和数据按各自的条款使用。
