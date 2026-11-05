# 世界模型辅助PPO：目标完成性审查

审查日期：2026-11-05（Asia/Shanghai）。目标为本地仿真研究、可复现实现与收益/代价/边界验证；按用户要求，车辆部署和Pi实时验收推迟。以下结论不把工程通过当作安全可靠性或稳定正收益证明。

| 目标要求 | 权威证据 | 审查结论 |
| --- | --- | --- |
| 真实强化学习基线 | `falconworld/ppo_agent.py`、`sb3_worldmodel_ppo.py`；无模型分支与官方SB3同seed训练后权重逐位一致测试；实际zip/Adam状态 | PPO策略/价值网络确实更新；使用成熟SB3默认优化主体，保留初期自研失败，不把MPC称RL |
| 基线训练充分性与重复性 | 官方默认204800单seed纯val交叉验证、65k三seed正式结果、204800三seed参考及其独立审计 | 已从弱配置升级并检查较充分预算；仍有seed失败。这里验收可核查的研究基线，不声称已得到满足实车可靠性门槛的控制器 |
| 训练世界模型与信息边界 | common_data manifest与train/val NPZ、24个冻结模型、`models.py` | 48个train与12个val完整episode，seed集合无交集；拟合/尺度仅train，选择仅val；无test/未来脚本作为输入，明确actuator为特权状态 |
| 模型真正改善/影响RL的通路 | `WorldModelPPO`更新前分支、独立GAE、行为概率核验、super.train及实测Adam次数 | 模型经验进入policy/critic优化；不是只画预测曲线；h1即时reward相同而next_obs/bootstrap不同，h3影响后续reward |
| 公平数据和计算预算 | 七方法ledger、zip optimizer状态；主组每seed65536真值仿真步、10240更新，model替换32768；总交互72311控制与20epoch额外更新对照 | 主组更新数相同；模型生成/预训练额外成本未隐藏；纯PPO不虚计offline消费；不同预算参考不冒称等预算 |
| 多seed冻结测试与统计 | 主21训练任务、1680唯一回合、参考3任务240唯一回合；按seed→family配对bootstrap、per_seed/family表 | 所有seed/失败纳入，ID/OOD分开，未把帧当独立样本；3seed区间和未校正多重比较限制保留 |
| 学习效率与最终驾驶表现 | 7方法val曲线、native/online/shared三横轴、最终固定zip、逐步轨迹、return/碰撞/完成/进度/gap/jerk/回退/耗时表 | 已量化；不据test选择best，不从4个曲线点推断精确学会时间或必然节省样本 |
| 收益来源、消融与失效边界 | no-history/h3/physics/额外更新对照，`SB3_MODEL_DIAGNOSIS.md`当前池纯val诊断，逐seed退化与较充分预算参考 | 区分学习组件/通用短目标/采样效应；递归预测更差、自车残差近零，不能把RL现象归为更准动力学；因果解释仍有限，不编造正结论 |
| 可执行复现及环境 | `setup_rl.ps1`与36项固定依赖；smoke/standard一键退出0；最终130 tests与31 subtests | 本机Windows/Python3.11.9 CPU验证；不同设备性能与干净机器速度不推断 |
| 可用模型加载 | 21个主策略zip、3个参考zip、24个主预测模型；逐job加载一致性、独立加载CLI实际跑val | 策略zip可直接官方PPO.load；在线策略不依赖世界模型；不声称partial环境/优化状态精确resume |
| 原始指标独立审计 | `scripts/audit_sb3_rl.py`主1680与`audit_sb3_reference.py`参考240均0错误；每步rawreward/实际速度jerk等复算 | SHA、精确case、真实Adam/预算一致。无逐步完整训练trace，因此仅交叉核验训练计数，未夸为独立执行溯源 |
| 中文报告、图与追溯 | `SB3_RL_REPORT.md`、`SB3_MODEL_DIAGNOSIS.md`、各root/report/FINAL_REPORT.md与acceptance、旧轮归档、`PPO_PROJECT_EN.md` | 原始/负结果保留；新旧不同配置不混合统计；图已目视检查 |

## 研究判定

工程和预注册研究交付已完成。相对纯PPO的主return区间在ID/OOD均跨零，稳定总体收益尚未成立；相对physics辅助PPO的OOD对照有有限收益支持。ID jerk代价、seed0 OOD退化、额外训练/数据成本、额外真实更新控制和纯PPO较充分预算均披露。

更长预算纯PPO仍有seed未过诊断门槛，该失败作为边界保留，未把门槛降低、挑seed或无限加预算来换成功标签。当前成果可用于后续研究复现，不可据此称满足实车安全/控制可靠性或Pi实时要求。用户推迟的上车工作未在本轮执行。
