# 世界模型辅助强化学习：预登记协议

登记时间：2026-11-04（Asia/Shanghai），在本轮RL正式测试前冻结。此实验学习策略Q值，区别于原世界模型MPC。只使用本地仿真，不连接设备、上线或部署R550。

## 主要问题与比较方法

问题是：相同真实环境交互与离线数据预算、相同策略梯度更新次数下，加入历史世界模型的一步Dyna数据是否改善最终贪心策略的回报、碰撞和任务完成；收益是否超过丰富辨识物理模型与重复使用真实经验的控制。

主比较为model_free Double-DQN与dyna_history。dyna_history使用已训练history世界模型，在训练回放起点生成一步合成转移，以0.5比例混入策略minibatch。physics_dyna使用最强丰富辨识物理模型，使用相同合成步数、混入比例和总策略更新预算。可选replay_update_control提高真实经验复用率，需明确其额外更新成本，不能混入同更新数主比较。不得根据测试胜负添加/删减方法。

所有方法都能获得同一份真实离线训练数据，且这些数据对Double-DQN可见；世界模型预训练使用的真实转移也计入共同离线数据成本。Dyna不能无记账地免费增加预训练数据。模型拟合只使用train；超参数和训练曲线只使用val。只评估固定最终policy，不以test最优checkpoint替代最终policy，不反复调测试集直到获胜。

## 状态、动作、环境与信息边界

每次policy输入为六帧[gap m, ego speed m/s, lead speed m/s, actuator m/s²]加上上一次实际applied加速度命令，共25维。固定尺度[50,20,20,6]、previous_u/6，无训练统计拟合、无OOD裁剪。actuator是特权仿真状态；这不是实车传感器可用性声明。

动作是5个命令增量[-.8,-.4,0,.4,.8]m/s²，再按[-6,2]m/s²限幅，经共同SafetySupervisor。所有策略使用同一奖励、限幅、回退、dt=.1s与120步时限。环境内部读取前车脚本，policy及合成模型不得读取family、condition、未来脚本、隐藏参数或标签真值作为输入。一步模型输出只能递推当前可用状态。

奖励版本记录在每次评估manifest，当前为longitudinal-rl-reward-v2-bounded-survival：非碰撞奖励1-min(2,dense_cost/10)，碰撞-100。dense包含公开间距、速度、动作、变化率与危险间距成本；此为仿真假设。碰撞terminated不bootstrap，时间上限truncated仍bootstrap。不得把time-limit当物理终止。

## 固定清单与预算

训练seed默认0、1、2，测试seed命名空间8,000,000+training_seed×100,000+condition_index×10,000+family_index×1000+i；验证独立命名空间7,000,000，训练必须避开这两个命名空间。condition顺序id/ood，family顺序constant/mild/brake/stopgo。测试清单由evaluation_cases固定生成并写manifest。

- smoke：1个训练seed，每族每分布2个配对episode，即每方法16回合，每回合120步；只验证实现，不支持训练随机性结论。训练交互/更新/模型生成预算由runner在运行前登记并保留。
- standard：3个训练seed，每seed每族每分布10个独立case，即每方法240回合、每族ID30/OOD30。训练seed间场景实例不重复。真实交互、offline数据、更新数和synthetic生成数在runner登记后冻结，禁止按测试结果增加预算。

每方法同training_seed使用相同场景seed，外生随机性可比。不同策略改变自身状态后无需强行复用相同完整轨迹。train/val/test按整个episode隔离，不能按相邻帧拆分。

## 指标与统计

逐episode记录return、碰撞/回合数、完成率、progress m、gap RMSE m、jerk RMS m/s³、回退次数/原因、policy与完整仿真控制周期p50/p95 ms、超100ms比例。完成要求无碰撞跑满120步且自车进度≥max(5m,前车进度)的一半，不能一直停车取得完成。jerk由实际速度差分加速度再次差分，包含首周期基于可见历史的加速度，不拿执行器内部值替代停车边界运动学。

周期计时包括输入解码/归一化、贪心policy、独立安全层和真实仿真step；policy计时不含仿真step。没有设备I/O，不能当车端实时性。报告pool全部周期p50/p95，同时保留episode分布。原始逐轨迹CSV保留所有失败，碰撞早停需报告长度；早停较短而误差低不代表驾驶更好。

主统计单位是独立episode及训练seed，不是相邻帧。对照差值定义为method−reference；return/progress/completed正值有利，collision/gap error/jerk/fallback/latency负值有利，但必须联合看安全与效率。以training_seed先重采样，再在seed内按family分层重采样配对episode，固定bootstrap seed9401、2000次，报告效应大小与95%区间。ID/OOD与各场景族分别报告，只有3个训练seed时区间仍不稳定；smoke单seed区间不能包含训练随机性。

## 样本效率与成本账本

训练曲线横轴同时给新增true-simulator steps以及共同offline transitions+新增steps；只有从同一offline基数出发的曲线才能比较样本效率。横轴不得把synthetic transitions混称real env steps。曲线纵轴仅val固定清单，测试只评估最终policy。

每method/seed需记录：共同offline真实转移及unique episode数、worldmodel预训练真实转移与耗时（共同数据仍需透明报告）、新增true env steps、成功policy updates、实际real/synthetic minibatch抽样数、生成/接受/拒绝synthetic转移数、worldmodel调用数、训练墙钟和评估墙钟，主机及线程条件。评估开销不算训练交互。缺账本的成本标未验证，不能从回放容量反推交互量。

解释证据链：模型预测能力→合成数据误差/覆盖→策略学习曲线→最终闭环→模型/训练/运行成本。即使负结果也报告：模型偏差可能被Q学习放大、强物理已充分解决、数据/更新预算不足、分布外失败。预测更好不自动证明RL更好，零碰撞不代表通用安全保证。

## 可复用入口

主runner调用falconworld.rl_evaluation.evaluate_policy(agent,eval_seeds,output,method=...,training_seed=...)，返回episode rows；summarize汇总严格配对清单并生成分层bootstrap。输出目录必须空以保留历史。scripts/evaluate_rl.py独立加载可信本地checkpoint，执行相同评价；--summarize接受显式episodes.jsonl路径列表。只有实际运行后才把具体命令列为已验证。
