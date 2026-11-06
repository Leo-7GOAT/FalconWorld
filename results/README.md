# 冻结实验与预训练策略

这里发布已经完成的合成仿真实验，不是执行复现命令后生成的新结果。重训默认写入根目录的 `outputs/`，不会覆盖本目录。

| 发布目录 | 原始实验目录 | 内容 |
|---|---|---|
| `sb3_standard/` | `outputs/rl_sb3/standard/` | 3 seeds × 7 方法、21 个最终策略、1680 测试回合 |
| `ppo_reference_204800/` | `outputs/rl_sb3/reference_204800/` | 3 seeds 纯 PPO、240 测试回合、每 seed 204800 在线步 |
| `ackermann_rl/` | `outputs/ackermann_rl/tracking_v2_quick/` | 新增横纵向联合控制，3 seeds × 3方法、9策略、288回合，独立审计通过 |
| `ackermann_rl_v1/` | `outputs/ackermann_rl/quick/` | 奖励修正前完整失败轮；不能与v2混合排名或作为当前奖励的复现结果 |
| `ackermann_gradient_selection/` | `outputs/ackermann_rl/model_gradient_selection/` | 仅验证集选择两个多步模型梯度候选；不访问新test |
| `ackermann_gradient_final/` | `outputs/ackermann_rl/model_gradient_final/` | 新seed3/4/5，五方法、65536步、600新测试回合，含额外更新和平滑控制 |
| `ackermann_gradient_efficiency/` | `outputs/ackermann_rl/model_gradient_efficiency/` | 新seed6/7/8，五方法、短预算、600新测试回合，含更充分总交互纯PPO控制 |
| `ackermann_mismatch_v2/` | `outputs/ackermann_rl/friction_mismatch_v2/` | 单调死区失配，四预测模型、五PPO方法、600闭环回合，含完整机制物理强对照 |
| `ackermann_mismatch_warm/` | `outputs/ackermann_rl/friction_mismatch_warm/` | 探索性补充，0/2048在线步各480次配对闭环运行，两预算都保留 |
| `ackermann_mismatch_excluded_v1/` | `outputs/ackermann_rl/friction_mismatch/` | 被排除的非单调近似，仅作开发归档，不支持物理失配结论 |

历史报告与 JSON 中记录的 `outputs/...` 按此表映射。Windows 相对路径分隔符保持原样；审计脚本会处理该记法。没有将训练、模型或测试轨迹按表现筛选；所有正式种子和方法完整保留。

## 正式实验结构

- `registration.json`、`source_snapshot/`：冻结的协议、代码字节与版本指纹。
- `common_data/`：相互独立的 train / val 合成数据 NPZ、episode 种子和 SHA256 清单；无需外部数据下载。
- `seed0/`、`seed1/`、`seed2/`：各方法 `final_policy.zip`、学习曲线、验证及测试逐回合日志、逐步轨迹；standard 另有每个种子的 `world_models/`。
- `independent_audit/`：已完成独立审计记录；`report/` 为指标、配对区间、成本及学习曲线汇总。
- `model_diagnosis/`：当前世界模型的一步/递归误差和 bootstrap 机制诊断，仅 standard 存在。

Checkpoint 以官方 Stable-Baselines3 格式保存，仅用于冻结策略评价；没有模型分支 RNG 的精确续训能力。已发布策略来自本项目自身实验，加载入口见主 README。

`SHA256SUMS.json` 为发布文件逐个校验表，涵盖代码、文档和冻结实验；它不包含自身与本次检查的 `publication_verification.json`。源实验已有注册 SHA256 保留，模型、合成数据、训练源快照和测试轨迹未经修改。两处历史 JSON 的本地路径经脱敏，详情见 [发布说明](../docs/PUBLICATION.md)。

完整正式训练没有在发布时重复进行；本次重新执行的是测试、原始日志审计和发布策略加载评价。具体命令、结果和检查范围见 `publication_verification.json`。
