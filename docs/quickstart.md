# 快速开始：运行 G1 行走示例

先按[首页第 1 节](../README.md#1-运行预训练示例无需-gpu)安装 CPU 依赖、下载并校验 v0.1.0 示例。下列命令均在仓库根目录执行，并已激活 `.venv`。

```bash
# 无窗口运行 20 秒，保存报告
./scripts/deploy.sh --sim mujoco --seconds 20 --velocity 0.6 0 0 --report outputs/my_run.json

# 有桌面环境时打开窗口
./scripts/deploy.sh --sim mujoco --seconds 20 --velocity 0.6 0 0 --show
```

速度依次是前进速度、左向速度（m/s）及左转角速度（rad/s）。报告中 `completed` 应为 true、`fallen` 为 false；还要检查 `mean_velocity` 是否接近指令。只站立不跌倒并不代表学会行走。

## 检查示例和录制视频

```bash
amp-lab inspect examples/release_seed43_review_local/bundle
MUJOCO_GL=egl ./scripts/deploy.sh --sim mujoco --seconds 20 --video outputs/my_run.mp4
```

`inspect` 检查部署包文件、机器人及 ONNX 接口。录制视频需要可用的 EGL 渲染驱动；只做无窗口推理无需渲染。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `amp-lab: command not found` | 激活 `.venv`，或使用 `.venv/bin/amp-lab`。 |
| 找不到 bundle 或 `deployment.yaml` | 确认示例归档已解压，目录为 `examples/release_seed43_review_local/bundle`。 |
| 校验失败或出现 `unindexed files` | 重新校验下载包并解压；将日志和视频写到 bundle 外的 `outputs/`。 |
| 关节或 ONNX 维度不匹配 | 使用同一部署包中的策略、配置和模型，不混用不同训练的文件。 |
| 窗口打不开 | 先不加 `--show` 验证无窗口推理，再检查桌面和图形驱动。 |

想训练、恢复或评测自己的策略，请看[一键工作流](one-command-workflows.md)；示例能力与限制见[模型卡](experiments/g1_amp_seed43_6000.md)。
