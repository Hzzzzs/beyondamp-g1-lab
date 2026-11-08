# 第三方来源和分发审核（待发布前复核）

此表区分上游声明、当前证据和本项目的发布决定；不把本仓库 MIT 许可扩展到第三方内容或自动推断权重许可。

| 内容 | 原始证据 | 当前处理 |
| --- | --- | --- |
| CMU 动作数据 07_01/02/03 | [CMU FAQ](https://mocap.cs.cmu.edu/faqs.php)允许复制、修改和再分发；[CMU 主页](https://mocap.cs.cmu.edu/)注明数据资助 NSF #0196217。 | 可按上游声明准备数据分发，保留来源、资助信息和转换记录；FAQ的声明不自动解决重定向时使用的机器人模型权利。当前 3 段各自哈希见候选模型卡和数据 manifest。 |
| G1 模型资产 | 固定的 [assetslib](https://github.com/Renforce-Dynamics/assetslib/tree/9df792784e3642163f03390bf6ae16cf88b222a5) 仓库顶层包含 Apache-2.0；其 [README](https://github.com/Renforce-Dynamics/assetslib)注明资产来自多个上游，包括 Menagerie 和 Unitree。上游 [Menagerie G1](https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_g1/LICENSE) 与 [unitree_model](https://github.com/unitreerobotics/unitree_model)分别列出 BSD-3-Clause。 | 按文件追溯实际包内 URDF、MJCF、USD、网格的来源与修改，确认覆盖范围；分发时携带适用许可、版权、NOTICE 与修改说明。不能仅凭仓库顶层 Apache 声明认定每份资产均可按 Apache 单独再分发；未完成逐文件追溯前不公开该模型包。 |
| beyondAMP 算法代码 | 固定 [beyondAMP 版本](https://github.com/Renforce-Dynamics/beyondAMP/tree/cee88cdc0958c417e316f9452f802e25a71bc289)的多个 setup 元数据标记 BSD-3-Clause；本地固定检出未发现仓库顶层 LICENSE。 | 包级元数据不证明整个仓库及其 third_party 子树均获同一许可。保持外部固定依赖，不将其代码打包进本仓库；公开分发前向权利人确认覆盖与适用条件。自行下载外部仓库也不自动赋予再分发或修改权限。 |
| robotlib | 固定 [robotlib 版本](https://github.com/Renforce-Dynamics/robotlib/tree/1380fe7f0f60720d40c6d2b57cf21f1d9a16fad6) 的 setup 元数据标记 BSD-3-Clause，检出中未见顶层 LICENSE。 | 保持外部固定依赖；确认权利及适用范围后再决定分发方式。 |
| 训练权重、转换数据及演示视频 | 本项目生成，来源包括上述数据、模型、依赖和仿真工具。 | 保存来源和指纹，不将生成产物自动视为可任意分发；权利人及相应上游条款审核完成前只作为本地候选。 |

截至 2026-11-08，资产逐文件来源、beyondAMP 和 robotlib 的许可范围仍未核实完成；上述内容不构成公开发布许可结论。未发送任何询问或公开上传。
