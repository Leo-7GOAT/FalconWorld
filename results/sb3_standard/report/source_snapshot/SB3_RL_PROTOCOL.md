# 强PPO基线：SB3世界模型辅助训练预登记

登记时间2026-11-04（Asia/Shanghai）；在新SB3对照训练及新测试结果产生前登记。此协议为新一轮实验，不覆盖或改称旧32k自研PPO首轮已成功。旧outputs/rl_ppo/standard及131k预算诊断、未缩放归档和报告全部保留。

## 换用SB3的依据与边界

旧自研PPO32k及固定131k训练预算在ID验证仍未达到可用门槛，不能用弱基线声称世界模型有效或普遍无用。固定204800步的SB3默认配置交叉检查只读取训练和8个固定ID验证场景；51200首个验证点8/8完成、0碰撞、raw return58.78。此为单seed小验证集的配置/实现证据，不是新一轮三seed正式测试，不按中间验证挑最终checkpoint或赢家。

新轮基于本机已核验Stable-Baselines3 2.9.2、Gymnasium1.4.0的官方PPO实现与默认配置：MlpPolicy、learning_rate3e-4、n_steps2048、batch_size64、n_epochs10、gamma.99、gae_lambda.95、clip_range.2、ent_coef0、vf_coef.5、max_grad_norm.5、CPU。网络等其余参数保留官方默认，构造/版本/源摘要在registration冻结。训练reward统一*.01，评价仍raw环境reward。该方案是SB3 PPO的模型辅助更新实验，不是原版MBPO。

## 七方法与数据来源

1. ppo：原生真实在线SB3 PPO；无需offline policy warmstart。
2. wm_ppo：history世界模型，更新前从当轮真实rollout状态与历史分支，由同一未更新behavior策略采样动作，生成horizon1短模型轨迹；替换一半当轮真实buffer学习样本，再调用super().train()。
3. physics_ppo：丰富辨识物理模型，同起点/采样规则、模型比例、步数及更新预算。
4. real_reuse_control：纯真实rollout，n_epochs20；额外梯度次数单列，是计算控制。
5. wm_no_history：无历史模型组件消融，其他预算同wm_ppo。
6. wm_long：history模型horizon3分支消融；生成、有效接受、实际学习样本数量分别记账，不能把h3更多生成量隐藏。
7. ppo_total_budget：把共同offline train+val转移数改花为额外online真实步数，原生offline0，使native_total_training_interactions与wm_ppo相等。它有更多policy更新，不混称同online/同梯度主比较。

真实/模型GAE分别计算。模型动作、old_log_prob和value来自同一冻结behavior策略，全部生成在更新之前；不能在已更新policy下重算old概率，不能把不同分支直接串接。碰撞terminated不bootstrap；TimeLimit与有限模型horizon需保留bootstrap，分支边界停止GAE链。

主比较ppo/wm_ppo/physics_ppo不增加总梯度次数：相同当轮buffer学习样本数、batch64及10epochs，只替换一半样本来源。记录真实/模型样本数和policy实际更新数；不根据测试表现改变替换比例、horizon或PPO默认超参数。没有off-policy replay和离线behavior概率伪装。

## 预算、停止与参数冻结

新入口run_sb3_rl.py，输出outputs/rl_sb3/{smoke,standard}。standard为3训练seed（0/1/2）、每method/seed65536新增online步；n_steps2048、batch64、10epochs对应完整rollout320个梯度minibatch，32个完整rollout为10240次实际梯度更新。主方法更新数必须核验相等，不以SB3内部按epoch统计的_n_updates冒充梯度minibatch总数。

ppo_total_budget额外采集共同offline转移数；最后partial rollout精确补齐请求的真实步数，不能默认向上补到2048倍数后声称相同总交互。partial的实际样本数、minibatch尾批与梯度更新量单独记账。smoke只验证七方法链路，具体较小步数由runner登记，不冒充standard达标。

最终policy由固定预算结束的checkpoint决定；训练曲线只看固定val，不能用test挑checkpoint或运行到某个测试胜出点。新test尚未产生，不读取新test来决定本协议。后续遇到实现错误先隔离证据、记录修复、新输出目录重跑，保留旧轮，不按胜负删除。

## 信息边界与固定评价

仍使用已有rl_env的25维固定尺度观测、5离散增量动作、dt=.1s和120步episode：六帧[gap m,ego m/s,lead m/s,actuator m/s²]+previous applied command，尺度[50,20,20,6]及u/6；actuator明确为特权仿真状态。安全Supervisor、环境机制、终止/碰撞、raw reward与评价指标不变。

history模型与物理拟合使用共同真实offline train/val，完整episode划分，不读test。offline不用于PPO actor/critic policy update；纯PPO无需offline成本需诚实报告。模型/策略不能读取未来前车脚本、family、condition、隐藏参数或真实未来作为在线输入。相同training_seed各方法使用相同冻结评价场景清单与外生随机性。

测试沿用evaluation_cases：8m命名空间、ID/OOD、constant/mild/brake/stopgo，每seed每族每分布10个独立case；3seed每方法240回合，7方法1680回合，每族ID30/OOD30。验证使用独立7m命名空间，训练避开验证/测试seed。若新runner使用独立命名空间，须在训练前registration显式登记并保持方法配对。相邻帧不跨train/val/test。

报告return、碰撞计数/回合、完成、进度、gap RMSE、实际速度差分jerk、安全回退及原因、policy/完整仿真周期p50/p95与超时率。完成不能靠一直停车：需无碰撞跑满120步且进度达到参考阈值。评价固定最终贪心policy；旧greedy/stochastic诊断不授权在test挑评价模式。

统计采用training_seed→family内配对episode分层bootstrap（2000次、seed9401），分别报告ID/OOD及各族。主wm_ppo对ppo与physics_ppo；再与同总交互ppo_total_budget比较。报告差值和区间，只有3seed时训练随机性区间仍不稳定，不把frames当独立实验。

## 样本效率与计算账本

曲线三个视角：(1)新增真实online步；(2)native实际消耗offline+online，纯PPO/real_reuse/ppo_total原生offline0；(3)共同shared预算仅记账，不冒充真实消费。model预训练数据、整套8模型预训耗时上界单列，不能归为单模型成本。评估交互与墙钟、模型生成墙钟、合成生成/接受/使用数、实际梯度次数、SB3版本和主机线程条件保留。

若模型改善只是额外更新、更多真实数据、更保守驾驶或多用计算，不能记为模型独立收益。预测提升不自动证明PPO提升；OOD/急刹失败、无优势及负结果如实报告。没有闭环收益证据时区分工程完成与优势未支持，不把通用安全或实车部署当作结论。

## 报告交付边界

新报告独立scripts/report_sb3_rl.py、docs/SB3_RL_REPORT.md，旧docs/RL_REPORT.md保留原32k首轮含失败诊断。只有新运行完整且审计通过才写已验证；partial只列完整性与已产生数据，不做优势判定。当前协议只是预登记，不能声称新SB3世界模型实验已经完成。

整个任务只进行本地仿真和文件报告；不连接设备、上线或部署R550。

## runner落盘后契约核验（新测试结果产生前）

run_sb3_rl.py实际smoke预算8192、standard65536；输出seed{seed}/{method}/complete.json与learning_curve.json，最终checkpoint为final_policy.zip。worldmodel使用共同数据按seed训练，manifest冻结模型NPZ SHA。评价仍独立test/episodes.jsonl。

report_sb3_rl.py独立输出docs/SB3_RL_REPORT.md与新root/report，保留旧RL_REPORT.md。报告审计精确real env步数和optimizer hook更新数：完整2048步每10epochs=320，partial按ceil(n/64)×epochs，reuse20epochs单列。model_usage.csv同时列generated与replaced，policy_configuration.json保留实际默认配置与metadata；partial或预算异常不生成优势区间。
