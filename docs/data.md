# 动作数据与来源

首个公开样本来自 [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/)，subject 07 的 `07_01`，官方标注 walk，原始频率 120 Hz。下载地址和 SHA256 固定在 `configs/cmu_sample.lock.json`。

[FAQ](http://mocap.cs.cmu.edu/faqs.php)允许复制、修改和再分发；[主页](http://mocap.cs.cmu.edu/)说明不能直接转售原始或转换格式的数据。发布时保留 CMU 来源与 NSF EIA-0196217 资助说明，附原始条款链接；本项目的 MIT 许可不覆盖动作数据。原站样本通过 HTTP 下载，以固定 SHA256 验证，不默认关闭 TLS 验证。

## 数据处理

1. ASF 骨架解析 bone axis、DOF 顺序、层次和长度，AMC 解析连续帧。原始长度换算为米：`(1/0.45) * 2.54 / 100`。
2. FK 在人体骨架上计算世界位置和旋转。CMU Y-up / Z-forward 转成机器人 Z-up / X-forward；左方向保持右手系。
3. G1 有界 IK 跟踪膝、踝、肘、腕目标，加入较小的平足、平滑和静止姿态约束。单一人体缩放是初始基线，不是完成质量验收的重定向器。
4. 命名关节 qpos 保存为不依赖 pickle 的 NPZ。按目标控制频率生成 50 Hz 轨迹，再由机器人模型 FK 计算 body 状态、速度和相邻帧。
5. 审计关节限位、速度、脚底碰撞球高度与近地脚部水平速度。v2 报告另计算支撑覆盖率、最长无支撑时段和所选脚底点的水平速度：从刚体质心线速度加上角速度贡献，避免把质心速度当作脚底速度。支撑假设只使用高度和 1 cm 滞回，不按速度筛选，以免隐去滑动。报告只描述运动学，不能替代动力学可行性验证。

当前 `07_01` 只有 132 个输出帧、2.62 秒，适合工程 smoke。`prepare_cmu_dataset.py` 另提供固定哈希的 `07_01/02/03` 训练片段与 `07_06` 独立验证片段，共约 12.2 秒；每段都有单独审计报告，规模仍小。原站描述均为 walk、120 Hz。同一人的不同片段不能宣称跨人泛化，后续还需更大范围的运动和形态验证。

脚锁定为显式的可选实验。初步实验出现较大根部修正，因此不将其作为默认推荐数据，也不将“脚贴地”自动视为更好数据。

## 命名 qpos NPZ

```text
fps                  scalar positive float
joint_names          [J] Unicode strings, unique
joint_pos            [T,J] radians
root_pos             [T,3] metres
root_quat_wxyz       [T,4] WXYZ unit quaternions
```

名字必须与目标模型全部 hinge joints 匹配。`amp-lab motion convert input.npz --output output.npz` 插值位置和关节角，旋转用 SLERP，输出包括：

```text
fps, joint_names, body_names
joint_pos, joint_vel                       [T,J]
body_pos_w, body_lin_vel_w, body_ang_vel_w  [T,B,3]
body_quat_w                                [T,B,4]
```

body pose 为 link actor frame 世界位姿，body 线速度参考刚体质心，与所选 IsaacLab API 约定一致。转换报告记录输入、输出、模型树指纹和上一步重定向报告。NPZ 均用 `allow_pickle=False` 读取。

训练加载时按实际 joint names 对齐；每段内部采样 `(t,t+1)`，不跨片段末尾采样，专家 FPS 必须等于控制频率。不接收只有 `[T,29]` 却没有关节名称的文件。

## 重新审计与实验性地面约束

已生成的数据可以直接重新审计，不改动数据或旧 manifest：

```bash
python scripts/audit_dataset.py data/processed/cmu_walk/manifest.json --output outputs/contact_audit_v2.json
```

工具先核对每段数据的 SHA256。`review_warnings` 的 90% 支撑覆盖率和 0.3 m/s 脚底速度 p95 是工程排查阈值，没有动力学验收效力。`near_ground_foot_fraction` 的分母是帧数 × 两只脚，`support_hypothesis.coverage_fraction` 的分母是帧数，两者不能混用。支撑假设不测量接触力，也不识别双支撑力分配。

默认 IK 之外，可生成一套独立实验数据：

```bash
python scripts/prepare_cmu_dataset.py --experimental-floor-ik --output data/processed/cmu_walk_floor_ik
```

该实验默认保持人体根部轨迹，通过关节 IK 加入脚底碰撞球高度残差和更强的平足先验。显式 `--root-height-offset` 会改变根部高度，并在来源报告中记录。假设平地行走没有腾空期，用人体双踝高度与 1 cm 滞回选择支撑脚，固定该支撑段的踝关节 XY 目标，并根据相对踝高度定义摆动脚离地目标；它没有使用测得的接触标签或动力学。必须比较审计、回放、关节速度和实际训练效果，不能仅依据贴地率将其设为正式训练集。此选项与改变根部轨迹的 `--experimental-contact-projection` 是两个不同实验。

本地对比（2026-10-22）：四段数据的支撑覆盖率由约 8%–13% 提高至 98%–99%，无关节越界或超过 1 cm 的穿地；但所选脚底水平速度 p95 为约 0.39–0.99 m/s，且最大关节速度提高。因此当前实验保留为候选，**没有替换默认数据**。后续优先检查支撑切换平滑性和滑动，再做物理验证。详细本地报告在 `outputs/cmu_floor_ik_comparison.json`（大文件与实验产物不入 Git）。

后续平滑摆动和机器人脚底高度映射已整理为 [运动学实验记录](experiments/motion_2026-10-22.md)。首轮 GPU 短实验可用候选预设生成数据：

```bash
python scripts/prepare_cmu_dataset.py --config configs/cmu_walk_candidate.json --output data/processed/cmu_walk_candidate_v1
```

该预设仍标记为待动力学验证。初学者运行最终示例时无需自行选择这些重定向超参数；正式 Release 会提供与权重完全匹配的数据。

## 调整动作速度

参考动作的速度应与目标任务接近。当前候选三段动作的机体系平均前进速度约为
1.09 / 1.06 / 0.81m/s，高于主要测试目标 0.6m/s。可以保留每个源姿态，统一放慢时间轴，
再按控制频率重新计算 FK 与全部速度：

```bash
.venv/bin/python -m amp_lab.cli motion retime \
  data/processed/cmu_walk_candidate_v1/train/07_01/candidate_qpos.npz \
  --speed 0.65 --output data/processed/slow_example/qpos.npz
.venv/bin/python -m amp_lab.cli motion convert data/processed/slow_example/qpos.npz \
  --fps 50 --output data/processed/slow_example/amp.npz
.venv/bin/python -m amp_lab.cli motion audit data/processed/slow_example/amp.npz
```

`speed=0.65` 表示播放速度为原来的65%，并非将输出控制频率改为32.5Hz。
第一步只更改源 qpos 的时间标记，第二步仍生成50Hz训练数据；不能只修改最终 AMP 文件的 fps。
来源报告保留父数据哈希与速度因子。放慢时间轴不能证明动作动力学可行，需要单独训练验证。
