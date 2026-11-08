# 示例发布要求

最终发布目标是一份能独立使用的 G1 平地 AMP 示例：自己的训练 checkpoint / ONNX、训练所用动作数据、训练和部署模型、配置、原始出处、复现命令、IsaacSim 和 MuJoCo 评测视频。

seed43第6000次权重已通过同一ONNX双环境平地行走，各24项矩阵测试均完成20秒未跌倒，当前作为主要示例候选，见 [模型卡](experiments/g1_amp_seed43_6000.md)。seed42第9700次候选继续保留。示例下载见仓库首页，第三方资产与代码许可仍需按[分发审核](distribution-review.md)逐项确认。

## 最小验收

- 目标 Isaac 镜像完成 preflight，训练正常保存、恢复，学习曲线记录真实实验。
- 使用同一 ONNX，在 IsaacSim 和 MuJoCo 各完成至少 20 秒前进行走：不跌倒、完成要求时长、速度 RMSE 不高于 0.3 m/s，实测平均前进速度至少为指令的一半且不低于 0.15 m/s。实测值来自评测报告的 `mean_velocity`；慢速命令下仅靠原地站稳也可能满足 RMSE，因此必须检查实际前进。该门槛是首版工程目标，不是课程官方评分线。
- 偏航角速度 RMSE 不高于 0.3rad/s，避免把持续转圈或左右大幅摆动当成指令跟踪；平均角速度接近零仍可能有明显振荡。
- 后续补充多速度、转向、初始朝向和重复运行；单次前进通过不代表已完成全面泛化评测。
- 权重必须与 checkpoint 导出报告、部署契约、模型资产树和训练数据 SHA256 一致。
- 发布动作数据前检查接触、滑动、速度、片段质量与条款，明确候选/验收状态。
- 补齐视频、模型卡、数据卡、依赖许可和固定版本。尤其是 beyondAMP 顶层许可尚需澄清，不把 setup 元数据视为全部上游文件统一授权。

代码中检查前 4 类机器可检验的证据；许可、视频和报告内容仍需最终审阅。

## 组装候选发布目录

```bash
python scripts/build_release.py --checkpoint outputs/g1_amp_seed42/model_5000.pt --spec outputs/g1_amp_seed42/deployment.yaml --policy outputs/g1_amp_seed42/policy.onnx --motions data/processed/cmu07_01/candidate_amp.npz --reports outputs/mujoco_eval.json outputs/isaac_eval.json --output outputs/release_candidate
```

该命令只在本地组装，不发布到 GitHub。它拒绝 CPU 诊断或测试 checkpoint，校验实际训练数据、ONNX 导出证据和双环境运行结果。产物：

```text
release_candidate/
  bundle/
    policy.onnx
    deployment.yaml
    robot/                  # XML/URDF/mesh/USD，保留原始目录结构
    model_manifest.json
    metadata.json
    checksums.sha256
  training/model.pt
  data/*.npz
  data/*.json
  evaluations.json
  release.json
  checksums.sha256
```

组装后继续补充训练日志/曲线、参考回放、策略视频、模型卡、数据卡和许可说明，再进行公开发布。外层状态保留为 `assembled_pending_license_and_documentation_review`，不自动宣称完成公开发布审查。

示例权重可放 GitHub Release 或适合大文件的模型托管服务；代码仓库只保存下载清单、SHA256、版本关联和小型说明文件，不把大权重、mesh 和缓存提交进 Git 历史。当前没有建立外部仓库或上传文件。
