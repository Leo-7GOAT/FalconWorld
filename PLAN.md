# FalconWorld 目标验收计划

当前目标/协议：RL_SPEC.md、docs/SB3_RL_PROTOCOL.md及docs/SB3_REFERENCE_PROTOCOL.md。用户选择PPO，部署/实时性推迟。

- [x] 实现并核验真实PPO actor-critic、GAE/TimeLimit、仿真观测动作和安全监督契约。
- [x] 训练低维世界模型，隔离完整episode train/val，无未来脚本或测试输入泄漏。
- [x] 世界模型当前策略短分支进入PPO更新，冻结行为概率、分支GAE与真实优化器预算检查通过。
- [x] 3seed×7方法正式公平对照：21任务/1680最终测试回合；数据/梯度/实际计算与纯PPO总预算控制。
- [x] h3/nohistory/physics/额外更新消融、三轴学习效率、最终驾驶指标、逐seed/场景族分层区间。
- [x] 独立较充分预算纯PPO参考：3seed×204800步/240测试，失败与门槛未达如实保留。
- [x] 主/参考原始日志及checkpoint实际计数审计0错；全测试130+31subtests。
- [x] 环境固定/准备、一键smoke/standard、加载训练策略CLI、中文报告与英文介绍。
- [x] 原始数据/模型/图表/来源摘要/归档/设计与完成性审查落盘，区分工程完成和优势证据。
- [ ] 用户延期：实际车辆部署、动态标定、ARM64/Pi实时性能及实车安全接管。不是本轮执行范围。

旧MPC纵向/二维/CARLA工作保留完整，详见原REPORT。旧自研PPO失败和验证修复不删除。所有历史过程见outputs/rl_sb3/goal_plan_history.md与goal_progress_history.md。任何后续新假设需新登记输出，不能根据冻结test挑参数或覆盖失败。
