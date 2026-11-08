# Seed43 行走示例：发布审阅材料

状态：**v0.1.0 示例已公开**。同一策略已通过平地双模拟器评测；上游授权范围和资产逐文件来源仍需核实，详见分发审核。

## 用户要获得什么

- 29 自由度 G1 模型、ONNX 推理策略、命名关节部署配置和完整校验清单。
- 真实训练 checkpoint、训练配置，以及 3 段与训练记录的 SHA256 一致的 CMU 重定向数据。
- MuJoCo 与 Isaac 平地行走结果、对应演示视频，以及可复现的依赖版本与命令。

## 当前材料

- [模型卡及数据指纹](experiments/g1_amp_seed43_6000.md)：训练设定、评测、限制和数据 SHA256。
- [数据来源及重定向说明](data.md)：CMU 来源、转换方法、命名数据格式和质量限制。
- [初学者运行说明](quickstart.md)：环境、检查、运行及排错。
- [分发审核表](distribution-review.md)：可确认的上游声明、须保留的信息，以及仍需核实的部分。
- [发布验收说明](release.md)：候选组装和人工复核。

开发工作区的视频为 `outputs/progress_20261107/seed43_mujoco.mp4` 和 `outputs/progress_20261107/seed43_isaac.mp4`；各为 20 秒，使用同一 ONNX。原始参考动作回放示例保存在 `outputs/release_candidate_9700/videos/reference_07_01.mp4`，是训练数据回放，不是策略输出。

已校验的本地部署与评测归档为 `outputs/progress_20261107/yaw1_seed43_6000_candidate_20261108.tar.gz`，它不含 checkpoint 和完整三段训练数据。完整候选另外在开发机 `outputs/release_seed43_review_20261108/` 组装，状态为 `assembled_pending_license_and_documentation_review`；其可用性和备份以项目状态的最新记录为准。

## 发布前待确认

1. 核对资产逐文件上游来源、版权标注、许可与变更说明；完成 beyondAMP 和 robotlib 的许可范围确认。
2. 确认训练 checkpoint、三段数据、部署模型、两环境报告和演示视频在最终分发包内完整且与模型卡指纹一致。
3. 确定公开托管位置与用户侧下载方式，审阅模型卡、许可说明和视频后再发布。

现有结果针对平地和有限指令，两个种子的实验不能证明跨人、复杂地形或真实机器人安全性；未经实机验证，不能将该包当作实机控制器。
