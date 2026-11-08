# G1 一键训练、评测和部署

先按[首页](../README.md)安装依赖并下载示例包，解压后目录为 `examples/release_seed43_review_local/`。示例策略的 MuJoCo 推理无需 Isaac 或 GPU。

运行 `./scripts/doctor.sh` 检查 CPU 依赖；其中 `g1_xml_exists` 检查的是训练用的上游资产缓存，示例 bundle 自带机器人文件，运行示例不要求该项为 true。训练前再执行下面的 `--training` 检查，并按训练教程准备上游依赖。

```bash
# 直接运行示例策略；加 --show 可打开窗口
./scripts/deploy.sh --sim mujoco --report outputs/demo.json

# 24 项速度与初始朝向矩阵；--gait 同时评估参考步态
./scripts/evaluate.sh --sim mujoco --gait --output outputs/my_evaluation
```

评测目录每次使用新名称，避免覆盖旧结果。`summary.json` 汇总各模拟器完成率与最大速度、偏航误差；`gait.json` 和轨迹单独保存。脚部支撑及滑动是高度/运动学代理，不是接触力测量。`--seconds 20` 是默认的每项仿真时间，不能将极短调试结果作为完成验收。

## 训练和使用自己的 checkpoint

在已经配置 Isaac Sim 4.5.0、IsaacLab 2.1.0 和训练依赖的环境中，设置 IsaacLab 路径。将 checkpoint 转成 ONNX 时，本地 `.venv` 还需要 `.[export]`、PyTorch 和锁定版本的 `rsl_rl_amp`；仅运行已有 bundle 不需要它们。训练环境的安装步骤见[项目首页](../README.md)。

```bash
export AMP_ISAACLAB=/path/to/IsaacLab
./scripts/doctor.sh --training
./scripts/train.sh --robot g1 --motion cmu_walk
./scripts/evaluate.sh --checkpoint latest --run-dir outputs/g1_amp --sim both --gait --output outputs/my_both
./scripts/deploy.sh --sim mujoco --checkpoint latest --run-dir outputs/g1_amp
./scripts/deploy.sh --sim isaacsim --checkpoint latest --run-dir outputs/g1_amp
```

训练默认使用 Release 中三段数据、seed 43、4096 环境、6000 次迭代和 `configs/train_g1_expert_fixed_yaw1.json`，日志记录在 `logs/g1_amp.log`。用 `--num-envs 32 --iterations 10 --output outputs/smoke` 可执行短训练集成检查，不能把短训练结果视为已学会行走。`--resume latest --output outputs/smoke --iterations 20` 从该目录最新完整 checkpoint 恢复到总迭代 20；恢复不会保留中断时的仿真物理状态。

训练正常结束后，入口会自动导出最近的 checkpoint 并组装 `outputs/artifacts/` 下的 artifact，包含 `checkpoint.pt`、ONNX 数值对齐报告、`artifact.json` 和 `bundle/`；`bundle/deployment.yaml` 是训练时记录的观测、动作、关节顺序和控制参数。训练统计保存在 checkpoint 中，不额外伪造运行时归一化文件。只需训练而暂不导出时使用 `--no-export`；如果导出依赖缺失，训练权重仍保存在原输出目录，可安装依赖后单独使用部署入口导出。

`latest` 只在显式 `--run-dir`（默认 `outputs/g1_amp`）中按迭代数选 `model_<N>.pt`，不会跨实验猜测。也可传 checkpoint 文件路径；它必须携带或指定实际训练时的 `deployment.yaml`（`--spec`），不能用未经训练的 profile 替代。首次运行会校验 checkpoint 与配置、做 ONNX 数值对齐，再使用指定或示例机器人资产构建 bundle，保存在 `outputs/artifacts/`。`--robot-assets` 可以指定匹配的机器人资产目录；匹配失败应使用正确的训练资产，不要重排关节绕过检查。

使用自己的数据时，先按[数据说明](data.md)完成重定向与转换，再通过 `--motions path/a.npz path/b.npz` 指定带关节名的 50 Hz AMP 数据。`--config` 可指定训练 JSON。

这组新入口的 MuJoCo 示例、checkpoint 导出和 CPU 检查已在本地验证；Isaac 训练、恢复和 Isaac 推理入口仍需在目标 GPU 环境验证。仓库中既有底层训练和双仿真结果不等同于新入口已完成 GPU 测试。
