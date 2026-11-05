# FalconWorld 世界模型辅助PPO交付进度

日期：2026-11-05（Asia/Shanghai）。本地研究工程与预注册对照已验收；稳定普遍收益仍未证明。用户要求的部署和Pi实时性延期，本轮无设备连接或车辆运动。

## 已核验完成

- 强基线：Stable-Baselines3 2.9.2 PPO，Gymnasium1.4.0、CPU Torch2.15.0；策略/critic确实训练，未把MPC或世界模型预测冒称强化学习。
- 模型辅助：从当轮真实仿真状态、用同一冻结behavior策略生成h1/h3经验，独立GAE后替换半数buffer；原SB3优化主体不增加主比较梯度次数。
- 正式标准：3训练seed×7方法=21任务，主组每seed65536真值仿真步/10240实际Adam更新；固定最终策略1680唯一测试回合（每族每分布30独立case）。
- 控制/消融：纯PPO、学习模型、丰富物理辅助、无历史、三步、双倍更新、同原生总采集成本PPO；数据、生成量、优化量和墙钟独立记账。
- 补充较充分预算：204800步×3seed纯PPO，32000实际Adam/seed，240固定最终测试回合。非等预算模型优势对照，seed失败未挑掉。
- 独立审计：主1680、参考240逐步rawreward/运动学jerk/碰撞完成/回退/进度等复算、源/模型/zip摘要、实际Adam和精确预算均0错误。
- 模型：21个主PPO策略zip、3个参考zip、24个主预测模型；训练48episode/验证12episode的seed集合无交集。部署观测内actuator明确是仿真特权状态。
- 当前全测试130 passed、31 subtests。smoke与standard一键实际退出0。36项运行时依赖closure固定；setup_rl.ps1安装和VerifyOnly均实际通过。独立加载zip的评价CLI实际执行通过。
- 中文主报告、成本/学习曲线/逐seed与场景族表、原始轨迹、机制诊断、英文介绍、旧轮失败与修复均保存。当前研究为纵向PPO；旧二维/CARLA成果保留为MPC，不写成二维/CARLA PPO已完成。

## 核心科研结论

- 主WM vs纯PPO：ID碰撞2/120 vs21/120，OOD11/120 vs28/120；但return的训练seed分层95%区间均跨0。均值改善不构成稳定总体优势，seed0 OOD还存在退化。
- WM vs物理辅助PPO：OOD return差+39.66[18.51,66.99]、碰撞率差-21.67个百分点，支持有限条件差异；不推广所有场景/seed。
- ID jerk相对纯PPO增加约0.672m/s³，平均训练墙钟约145.38s vs98.63s。相对同总交互和额外更新控制，return优势不稳。
- 当前val模型池history长递归四族均劣于物理；自车残差近零。h1即时rawreward相同，差异通过next_obs/critic bootstrap；短目标与根重采样等是解释边界，不把现象包装成更准确动力学的因果证明。
- 较充分预算纯PPO：ID碰撞21→13、return42.77→52.63；OOD未一致改善，seed1仍未过诊断门槛。没有降低门槛、挑最好checkpoint或无限搜索。

## 入口和证据

README_RL.md；docs/SB3_RL_REPORT.md；docs/SB3_MODEL_DIAGNOSIS.md；docs/RL_COMPLETION_AUDIT.md。

主输出outputs/rl_sb3/standard；补充outputs/rl_sb3/reference_204800；主报告及验收快照各root/report。旧MPC与自研PPO首轮独立保留，不混成同一实验。完整历史过程在outputs/rl_sb3/goal_progress_history.md。

这里完成的是仿真研究、可核查基线和交付，不是可靠上车控制或Pi实时验收，也不是世界模型普遍优于PPO的证明。
