# 初学者：先运行示例，再考虑训练

首次运行请先按[仓库首页](../README.md)下载并解压示例；下文的 `examples/g1_amp` 可替换为 `examples/release_seed43_review_local/bundle`。

目标体验是：安装 CPU 推理依赖 → 解压示例包 → 检查 → 打开 MuJoCo。

**平地行走候选已通过双环境测试，示例下载见仓库首页。** 结果与本地位置见 [示例模型卡](experiments/g1_amp_9700.md)。早期开发诊断包仍不能当作当前行走示例。

## 运行部署包

在本仓库根目录执行（Linux，Python 3.10 或 3.12）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/cpu.txt -e '.[mujoco]'
```

把下载的部署包解压到 `examples/g1_amp`。这个目录内应直接包含 `deployment.yaml`、`policy.onnx`、`robot/`、`metadata.json`、`checksums.sha256`，避免多套一层目录。如果拿到的是完整训练 Release，部署包在其 `bundle/` 子目录，下面的路径换成该目录即可。

在本项目开发工作区，已验证的9700次候选已解压到 `outputs/release_candidate_9700/`，可以直接运行下面两条命令：

```bash
amp-lab inspect outputs/release_candidate_9700/bundle
amp-lab deploy outputs/release_candidate_9700/bundle --headless --seconds 20 --velocity 0.6 0 0 --report outputs/quickstart_mujoco.json
```

这份候选由真实训练产生，已通过平地双模拟器测试；该本地路径是开发产物，示例下载见仓库首页。若使用其他完整候选包，请把路径改成该包内的 `bundle/`。

另一份训练到 6000 次的 seed43 候选也已通过同一 ONNX 的双模拟器平地测试；如果在本工作区使用，请先确保候选归档已解压，然后运行：

```bash
tar -xzf outputs/progress_20261107/yaw1_seed43_6000_candidate_20261108.tar.gz
amp-lab inspect outputs/gpu_expert_fixed_yaw1_seed43_6000_20261108/gait_evaluations/model_6000/bundle
amp-lab deploy outputs/gpu_expert_fixed_yaw1_seed43_6000_20261108/gait_evaluations/model_6000/bundle --headless --seconds 20 --velocity 0.6 0 0 --report outputs/seed43_mujoco.json
```

这份包的元数据仍标记为开发状态；它确实来自训练，但许可和公开分发审核尚未完成。测试条件、指标和数据来源见 [seed43 候选记录](experiments/g1_amp_seed43_6000.md)。

```bash
amp-lab inspect examples/g1_amp
amp-lab deploy examples/g1_amp --seconds 20 --velocity 0.6 0 0
```

`inspect` 会校验全部文件、加载包内机器人并实际做一次 ONNX 推理，输出观测维度、动作维度、控制频率和包的状态。检查通过表示接口可用；策略能力由随包评测报告说明。此流程不需要安装 PyTorch、Isaac、训练数据或执行 `fetch-upstream`。

速度依次是前进速度（m/s）、左向速度（m/s）、左转角速度（rad/s）。最终允许范围会由正式示例模型卡说明。

## 无显示器运行

只评测，不打开窗口，不需要渲染后端：

```bash
amp-lab deploy examples/g1_amp --headless --seconds 20 --report outputs/my_run.json
```

有可用 EGL 驱动时可以录制视频：

```bash
MUJOCO_GL=egl amp-lab deploy examples/g1_amp --headless --seconds 20 --video outputs/my_run.mp4
```

`my_run.json` 中查看 `completed`、`fallen`、`velocity_rmse` 和 `mean_velocity`。
`mean_velocity` 依次为实测平均前进速度、左向速度和左转角速度，与 `--velocity` 命令对照。
例如命令为 `0.6 0 0`、实测前进速度接近零，表示策略仍在原地站立；完成 20 秒且未跌倒不能单独证明学会行走。

## 遇到问题

| 现象 | 处理 |
| --- | --- |
| `amp-lab: command not found` | 激活 `.venv`，或使用 `.venv/bin/amp-lab` |
| 找不到 `deployment.yaml` | 检查参数是否指向包含该文件的部署包目录 |
| `Checksum failure` / `unindexed files` | 重新解压原包；日志和视频写到包外的 `outputs/` |
| ONNX 维度或模型关节不匹配 | 使用同一完整包中的权重、配置、模型；重新解压原包 |
| 无法打开窗口 | 先加 `--headless` 验证推理；录视频还需要渲染驱动 |
| 想训练自己的策略 | 返回 README 的数据准备和 GPU 训练章节 |

开发者本地可用 `outputs/cpu_smoke_v2/bundle` 替换示例路径检查接口，但它是明确标记的诊断产物，不能用于评估行走效果。
