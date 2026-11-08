# 第三方代码、模型与数据

本仓库新增代码采用 MIT。该许可不覆盖下载的第三方代码、机器人资产或动作数据。

| 来源 | 固定版本 / 用途 | 许可与处理 |
| --- | --- | --- |
| [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP/tree/cee88cdc0958c417e316f9452f802e25a71bc289) | `cee88cdc0958c417e316f9452f802e25a71bc289`；AMPPPO、ActorCritic、判别器 | 包元数据标记 BSD-3-Clause，部分 rsl_rl 文件含完整 BSD 声明；顶层未找到统一 LICENSE，公开再分发前需澄清。当前按固定提交外部安装 |
| [robotlib](https://github.com/Renforce-Dynamics/robotlib/tree/1380fe7f0f60720d40c6d2b57cf21f1d9a16fad6) | `1380fe7f0f60720d40c6d2b57cf21f1d9a16fad6`；G1 URDF 配置 | 保留上游出处；当前外部安装，不整体复制 |
| [assetslib](https://github.com/Renforce-Dynamics/assetslib/tree/9df792784e3642163f03390bf6ae16cf88b222a5) | `9df792784e3642163f03390bf6ae16cf88b222a5`；G1 模型与 mesh | 仓库含 Apache-2.0 LICENSE；G1 来源及 Unitree 版权说明需随最终资产包保留。当前只在忽略的缓存目录下载 |
| [IsaacLab](https://github.com/isaac-sim/IsaacLab/tree/v2.1.0) | `v2.1.0`；环境与 explicit PD | BSD-3-Clause，作为外部安装 |
| NVIDIA Isaac Sim | `4.5.0` | NVIDIA 自身条款；本仓库不再分发模拟器 |
| [CMU Graphics Lab Mocap](http://mocap.cs.cmu.edu/) | 07.asf / 07_01.amc；公开动作样本 | 免费复制、修改、再分发；不直接转售原始/转换数据。注明 CMU 与 NSF EIA-0196217，保留 [FAQ](http://mocap.cs.cmu.edu/faqs.php) 和主页条款 |

`configs/*lock.json` 记录来源与提交。打包模型前生成完整资产指纹，公开前补齐实际被打包文件的版权与许可，不用一个仓库的总许可覆盖来源不明的子资产。
