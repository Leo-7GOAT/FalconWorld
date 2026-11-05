# 世界模型辅助PPO：正式预登记协议

2026-11-04（Asia/Shanghai）。主任务依据用户明确要求采用PPO。先前Double-DQN仅历史原型，协议保留于RL_PROTOCOL_DQN_PROTOTYPE.md，不能作为PPO结果或主比较。此协议在正式PPO测试前登记；只进行本地仿真，不连接设备、上线或部署R550。

## 核心问题与方法

问题是：相同真实在线交互预算和总梯度minibatch数下，用世界模型生成的短轨迹替换部分PPO学习样本，是否改善样本效率和最终闭环表现，收益是否超过强物理模型和额外真实数据复用。

- ppo：纯真实在线rollout的PPO；无离线actor warmstart。
- wm_ppo：history世界模型，训练时从本轮真实PPO rollout的可观测状态与历史分支，在策略更新之前由同一冻结behavior策略生成一步模型轨迹，模型混合比例0.5。
- physics_ppo：最强丰富辨识物理模型，相同分支起点规则、模型步数、混合比例、真实交互及总梯度预算。
- real_reuse_control：只用真实rollout、可额外2倍梯度更新，单独记账为计算控制，不混称相同更新数比较。
- 可选wm_no_history及wm_long（horizon3）只作为最终采用组件的消融；不得按测试胜负挑选保留方法。

模型分支动作来自当前冻结behavior策略，需保留old_log_prob、value及结束bootstrap value。真实与模型轨迹分别计算GAE，不能跨不连续分支串接，也不能把模型轨迹插入真实episode造成伪时序。PPO更新使用对应采样时策略old_log_prob，不能使用更新后的策略重算旧概率。时间上限truncation bootstrap，物理碰撞termination不bootstrap。

不使用off-policy replay或旧DQNbuffer；离线数据不能直接塞入PPO actor更新，不假造behavior log probability把离线转移变成PPO数据。策略与价值函数均实际学习，不能把原MPC结果冒称RL结果。

## 固定训练预算及公平性

smoke为1个训练seed、每method/seed新增真实online env steps4096；standard为3个训练seed、各32768新增真实online steps。rollout512，batch128，每轮4epochs等效16个梯度minibatch；ppo、wm_ppo、physics_ppo具有相同总actor+critic minibatch更新次数。wm_ppo模型比例0.5、horizon1；horizon3为消融。实际更新次数、抽取样本数、生成/接受/拒绝模型步数全部记录，不以配置值替代actual。real_reuse_control额外更新单独列成本。

共同真实离线数据仅用于世界模型/物理模型拟合，按完整episode train/val/test隔离。所有对照都可以获得相同离线采集成本预算；纯PPO并不需要这些数据，也不得因此虚称使用了离线actor训练。报告同时给出共同数据预算视角，以及原生纯PPO zero-offline视角，后者可能具有更低总数据成本。世界模型预训练数据、归一化、超参数选择不能读测试集；预训练墙钟与实际转移数单列。

样本效率曲线仅在固定val清单评估，画两个横轴版本：(1)新增真实online env steps；(2)offline真实转移+新增online steps。不可隐藏世界模型预训练数据/计算开销，synthetic steps不能混称real steps。模型预测误差改进不自动代表PPO改善。

所有方法固定最终policy做测试，不以test最优checkpoint替代最终policy。验证集可用于既定超参数/模型选择，测试集不用于增加预算、翻转负结果或反复挑checkpoint。合理迭代仍无优势时保留负结果。

## 环境、观测和动作

每次policy输入为六帧[gap m, ego speed m/s, lead speed m/s, actuator m/s²]加上上一次实际applied加速度命令，共25维。固定尺度[50,20,20,6]、previous_u/6，无训练统计拟合、无OOD裁剪。actuator为特权仿真状态，不代表实车天然可测。

5个离散动作增量[-.8,-.4,0,.4,.8]m/s²，再限幅[-6,2]并经过统一独立SafetySupervisor。所有方法使用同一奖励、dt=.1s、120步时限、动作限制及回退。family、condition、隐藏环境参数、未来前车脚本只由真实环境使用，不能进入policy或模型输入；当前观测历史和候选动作可用。

奖励版本记录在每次manifest，当前longitudinal-rl-reward-v2-bounded-survival：非碰撞奖励1-min(2,dense_cost/10)，碰撞-100。dense包括公开间距、速度、动作、变化率与危险间距成本；为明确仿真假设。真实和模型使用同一奖励定义。

## 固定测试清单与统计单位

训练seed0/1/2；测试seed为8,000,000+training_seed×100,000+condition_index×10,000+family_index×1000+i。验证用独立7,000,000命名空间；训练避开这两个命名空间。condition顺序id/ood，family顺序constant/mild/brake/stopgo。每method同training_seed使用同清单，evaluation_cases写manifest保留。

smoke每族每分布2个配对episode，即每方法16回合；只验证实现，不支持训练随机性结论。standard每训练seed每族每分布10个独立case，即每方法240回合、每族ID30/OOD30；每回合120步。训练seed间场景实例不重复。train/val/test按完整episode隔离，不按相邻帧拆分。

不同策略改变自身轨迹后，不强行复用相同完整状态轨迹。主统计单位是episode和训练seed，不是frames。效应为method−reference；return/progress/completed正值有利，碰撞/误差/jerk/回退/耗时负值有利，必须联合看安全与效率。

采用training_seed先重采样、seed内family分层配对episode重采样；固定bootstrap seed9401、2000次，报告差值与95%区间。分别报告各family及ID/OOD，只有3训练seed时区间仍不稳定；smoke单seed区间不含训练随机性。

## 指标、原始证据与计算成本

逐episode记录return、碰撞次数/总回合、完成率、progress m、gap RMSE m、jerk RMS m/s³、回退次数/原因、policy与完整仿真控制周期p50/p95 ms及超100ms率。完成为无碰撞跑满120步且进度≥max(5m,前车参考进度)的一半；不能靠一直停车取得完成。

jerk根据实际速度差分得到加速度再差分，首周期由可见历史估计前一加速度，停车时不拿执行器状态替代实际运动学。碰撞早停必须保留长度和失败，不因样本更短而误判误差更低。

周期包含输入解码/固定缩放、policy、安全层和真实仿真step；policy时延不含仿真step。提供全部周期pooled p50/p95及episode分布。没有车辆IO，不能当Pi实时性或已部署声明。

成本账本每method/seed记录：offline真实转移/独立episode、预训练数据和墙钟、新增真实online steps、成功actor+critic梯度minibatches、实际真实/模型样本数、模型生成/接受/拒绝步数与调用数、训练和评估墙钟、主机/线程条件。使用runner actual字段；缺项标未验证，不从buffer长度或配置猜测。

证据链应为：世界模型学到的能力→合成轨迹偏差/覆盖→val学习曲线→最终PPO闭环→总数据/计算代价。没有优势、物理已充分、模型偏差被策略利用或OOD失败均需报告。零碰撞不等于通用安全保证。

## 可复用评价入口

falconworld.rl_evaluation.evaluate_policy(agent,eval_seeds,output,method=...,training_seed=...) 接收兼容act(encoded,explore=False)的PPOAgent。输出逐轨迹CSV、episode JSONL/CSV、manifest与完整性；summarize默认reference='ppo'，给全局/各family指标和严格配对分层bootstrap。输出目录必须空以保留证据。

scripts/evaluate_rl.py加载可信本地PPOAgent checkpoint；--summarize接受显式episodes.jsonl列表。--cost-ledger保留实际成本账本；只有实际跑过的具体命令才可标记已验证。旧DQN contract检查产物保留历史，不计正式PPO测试结果。

## ppo_total_budget：同总真实交互控制（首次正式PPO smoke前追加）

增加第七方法ppo_total_budget：纯PPO额外采集online真实环境步数，额外量等于共同offline train+val转移数；这些转移不作为离线policy更新，其原生offline消耗为0。这样native_total_training_interactions与wm_ppo的offline+online真实训练交互相等。该方法允许相应更多rollout与梯度更新，实际数值单独记账，不混入ppo vs wm_ppo相同online/相同更新数的主比较。部分尾rollout的更新数以actual为准，不能统一写死16。

样本效率主图第二横轴改为native_total_training_interactions（实际用到的offline+online），ppo、real_reuse_control、ppo_total_budget均为zero-offline；shared_data_budget_interactions可作第三种预算视角，明确只是共同预算记账，不能冒充算法实际消耗。纯PPO本身无需离线采集，可能有更低原生总成本；同总交互控制用于检验把这些真实数据预算直接花在online探索是否更划算。

smoke只检验七方法112个测试回合及端到端证据链，1训练seed不建立效能结论；standard32768online步、3seed登记尚需真实运行，不可拿smoke替代。

本地入口说明（供主README引用）：`python run_rl.py --preset smoke --workers 3`；训练完成后 `python scripts/report_rl.py --input outputs/rl_ppo/smoke` 生成 `docs/RL_REPORT.md` 与 `outputs/rl_ppo/smoke/report`。standard使用 `--preset standard` 与新注册输出；只有命令真实退出成功才标记完成。模型辅助PPO不是旧MPC，不连接R550、不上线部署。

## 训练／验证优化校准与归档（2026-11-04）

一次仅train/val诊断发现实际Adam优化步长需要校准，依据见docs/PPO_VALIDATION_DIAGNOSIS.md。统一训练learning_reward_scale=0.01，真实与模型的GAE学习reward均乘该值；环境raw reward、评价return、120步任务与测试清单保持原定义。此为优化尺度校准，不根据test结果选择scale，不因某方法输赢修改评价。

未缩放的完整smoke和standard运行分别保留于outputs/rl_ppo/smoke_unscaled_validation_trial、outputs/rl_ppo/standard_unscaled_validation_trial，保留所有checkpoint、metrics与源快照。新轮使用相同common_data与worldmodels重训PPO，各方法统一缩放；registration.json明确learning_reward_scale。不能把旧轮优胜方法或旧checkpoint替换新轮固定最终policy。

最终正式报告应在新outputs/rl_ppo/standard完整后执行 `python scripts/report_rl.py --input outputs/rl_ppo/standard`，生成docs/RL_REPORT.md。生成器默认输入已改standard，报告明确scale与raw评价区别；smoke只作为工程检查，不能替代standard效能证据。归档轮保留为历史证据，绝不以胜负选择展示轮次。
