# 世界模型辅助PPO：证据、收益、代价与适用边界

运行目录：`outputs\rl_ppo\standard`；preset=standard；算法为categorical clipped PPO，模型分支辅助on-policy更新，不能称为原版MBPO。

训练学习奖励缩放learning_reward_scale=0.01。评估始终使用未缩放的环境原始reward/return，不能将学习目标的缩放误读为评价指标变更。

完整性：已验证齐全；任务21/21，episode 1680/1680，缺失0，重复0。

固定预算、最终checkpoint测试，val仅监控、不挑test-best。只运行本地仿真，没有R550连接、上线或部署。

## 当前交付边界与PPO基线质量

本报告是每方法每seed 32k在线步首轮正式对照（ppo_total_budget另计），并非整个世界模型改善RL目标已经完成。完整性通过只能证明这轮登记实验齐全，不能证明PPO已学到可用驾驶策略。

仅train/val的后续固定预算诊断把scaled纯PPO seed0从32768步继续到131072步；最终8个固定ID验证场景完成率0.50、碰撞率0.50、raw return 1.93，仍未达到预登记完成率≥0.9、碰撞率≤0.1、return≥30的联合门槛。65k/98k中间checkpoint全部保留，未择优替代最终131k。单seed、8场景不能证明更大预算普遍无效，但足以说明当前PPO基线质量尚不足，不能利用弱基线宣称世界模型有效，也不能据负结果断言世界模型普遍无用。

诊断来源：docs/PPO_VALIDATION_DIAGNOSIS.md与outputs/rl_ppo_budget_validation/seed0_131072/DIAGNOSIS.md；源摘要另存report/diagnosis_sources.json。SB3默认设置固定204800步的仅验证集交叉检查尚在进行中（pending），不读取test，不挑赢家，不能代替本轮正式测试。

## 仅训练／验证集的优化校准

校准轮将真实和模型GAE学习目标统一使用reward×0.01，环境、评价raw reward与测试清单不变。缩放依据仅train/val的实际Adam步长诊断：fresh Adam一步实测actor参数步长由0.006999变为0.021098（约3.01倍），不能按clip系数声称历史Adam每步被压小82倍；不读取测试结果选择scale。仅验证greedy/stochastic诊断分别4/8和32/64碰撞，两者仅8个独立场景，不能归咎于greedy单一原因。未缩放smoke/standard已有产物保留在outputs/rl_ppo/smoke_unscaled_validation_trial与standard_unscaled_validation_trial，全部已产生checkpoint/metrics归档，不按胜负丢弃旧轮，不宣称旧轮全部登记job都完整。校准重训复用相同common_data/worldmodels，旧轮不替代新轮正式结果。

## 实验约定

ppo使用真实在线rollout；wm_ppo使用history模型一步分支、0.5比例；physics_ppo使用丰富辨识物理同样分支；wm_no_history和wm_long检验历史与分支长度；real_reuse_control使用额外2倍更新，作为独立计算控制。ppo_total_budget把共同offline转移数改花为额外online步，使其原生总真实训练交互与wm_ppo相同，允许更多更新且单独记账。离线数据仅拟合预测模型，不进入PPO actor/critic梯度。真实与模型轨迹分别GAE，分支old_log_prob来自更新前同一冻结behavior策略。

主比较每method/seed新增online预算为32768真实步，rollout512，batch128，完整rollout每轮16个梯度minibatch（real_reuse_control32）。ppo_total_budget另加offline对应online步数，尾段与更新数按实际账本报告，不能归入同online/同更新数主比较。

## 最终策略闭环

|方法|分布|回合|碰撞|完成|return|进度m|gap RMSE m|jerk RMS m/s³|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|physics_ppo|id|120|60|60|-3.499|103.783|12.737|2.354|
|physics_ppo|ood|120|59|61|-25.270|165.602|11.362|1.639|
|ppo|id|120|60|60|-4.574|102.004|12.820|2.288|
|ppo|ood|120|57|63|-21.888|163.972|11.237|1.616|
|ppo_total_budget|id|120|60|60|-4.474|102.002|12.798|2.282|
|ppo_total_budget|ood|120|54|66|-16.420|164.440|10.841|1.574|
|real_reuse_control|id|120|53|67|4.631|100.968|14.017|1.702|
|real_reuse_control|ood|120|48|72|-5.039|166.164|10.994|1.368|
|wm_long|id|120|60|60|-3.494|103.760|12.809|2.354|
|wm_long|ood|120|59|61|-24.313|164.738|11.284|1.621|
|wm_no_history|id|120|57|63|3.589|104.095|12.660|2.211|
|wm_no_history|ood|120|50|70|-9.679|165.586|11.109|1.450|
|wm_ppo|id|120|60|60|-2.754|103.773|12.690|2.322|
|wm_ppo|ood|120|59|61|-25.189|165.583|11.357|1.639|

未齐全表格仅描述已完成episode，不能据此排名；碰撞早停长度与全部轨迹均保留。完成要求无碰撞跑满120步且进度达到参考阈值，不能仅凭停车的低碰撞率宣称优势。

## 世界模型相对纯PPO、强物理与同总交互控制的配对效应

差值为wm_ppo−对照；return/progress/completed越高越好，碰撞/误差越低越好。训练seed先重采样，再在seed内按family分层配对episode重采样，95%区间。

|对照|分布|指标|差值|95%区间|配对数|
|---|---|---|---:|---|---:|
|ppo|id|return|1.8196|[0.5173,3.1981]|120|
|ppo|id|collision|0.0000|[0.0000,0.0000]|120|
|ppo|id|completed|0.0000|[0.0000,0.0000]|120|
|ppo|id|progress_m|1.7694|[-0.1700,5.2844]|120|
|ppo|id|gap_rmse_m|-0.1305|[-0.2712,-0.0223]|120|
|ppo|id|jerk_rms_mps3|0.0338|[-0.1124,0.2167]|120|
|ppo|id|fallback_count|1.0333|[-1.2417,4.1252]|120|
|ppo|id|cycle_p95_ms|0.0780|[-0.1317,0.3576]|120|
|ppo|ood|return|-3.3012|[-8.0802,0.2932]|120|
|ppo|ood|collision|0.0167|[0.0000,0.0583]|120|
|ppo|ood|completed|-0.0167|[-0.0583,0.0000]|120|
|ppo|ood|progress_m|1.6112|[-0.8373,5.5378]|120|
|ppo|ood|gap_rmse_m|0.1198|[-0.1539,0.4602]|120|
|ppo|ood|jerk_rms_mps3|0.0223|[-0.0701,0.0950]|120|
|ppo|ood|fallback_count|1.4250|[-0.1167,3.3008]|120|
|ppo|ood|cycle_p95_ms|0.1325|[-0.1669,0.4471]|120|
|physics_ppo|id|return|0.7449|[-0.0506,2.1825]|120|
|physics_ppo|id|collision|0.0000|[0.0000,0.0000]|120|
|physics_ppo|id|completed|0.0000|[0.0000,0.0000]|120|
|physics_ppo|id|progress_m|-0.0093|[-0.0931,0.0968]|120|
|physics_ppo|id|gap_rmse_m|-0.0471|[-0.1325,0.0027]|120|
|physics_ppo|id|jerk_rms_mps3|-0.0319|[-0.0871,0.0000]|120|
|physics_ppo|id|fallback_count|-0.2333|[-0.5919,0.0083]|120|
|physics_ppo|id|cycle_p95_ms|0.2375|[0.0670,0.4062]|120|
|physics_ppo|ood|return|0.0816|[0.0000,0.2885]|120|
|physics_ppo|ood|collision|0.0000|[0.0000,0.0000]|120|
|physics_ppo|ood|completed|0.0000|[0.0000,0.0000]|120|
|physics_ppo|ood|progress_m|-0.0190|[-0.0736,0.0105]|120|
|physics_ppo|ood|gap_rmse_m|-0.0050|[-0.0169,0.0004]|120|
|physics_ppo|ood|jerk_rms_mps3|-0.0005|[-0.0018,0.0001]|120|
|physics_ppo|ood|fallback_count|-0.0250|[-0.1000,0.0083]|120|
|physics_ppo|ood|cycle_p95_ms|0.2831|[0.1065,0.5231]|120|
|ppo_total_budget|id|return|1.7203|[0.2272,3.1981]|120|
|ppo_total_budget|id|collision|0.0000|[0.0000,0.0000]|120|
|ppo_total_budget|id|completed|0.0000|[0.0000,0.0000]|120|
|ppo_total_budget|id|progress_m|1.7708|[-0.1700,5.2844]|120|
|ppo_total_budget|id|gap_rmse_m|-0.1088|[-0.2693,0.0400]|120|
|ppo_total_budget|id|jerk_rms_mps3|0.0399|[-0.1121,0.2167]|120|
|ppo_total_budget|id|fallback_count|1.0417|[-1.2417,4.1252]|120|
|ppo_total_budget|id|cycle_p95_ms|0.1846|[-0.1045,0.3819]|120|
|ppo_total_budget|ood|return|-8.7686|[-22.2896,0.2932]|120|
|ppo_total_budget|ood|collision|0.0417|[0.0000,0.1250]|120|
|ppo_total_budget|ood|completed|-0.0417|[-0.1250,0.0000]|120|
|ppo_total_budget|ood|progress_m|1.1434|[-2.0843,5.5378]|120|
|ppo_total_budget|ood|gap_rmse_m|0.5161|[-0.1539,1.5129]|120|
|ppo_total_budget|ood|jerk_rms_mps3|0.0645|[-0.0323,0.2025]|120|
|ppo_total_budget|ood|fallback_count|2.5417|[-0.1083,5.9667]|120|
|ppo_total_budget|ood|cycle_p95_ms|0.2100|[-0.2186,0.4807]|120|

smoke只有1个训练seed，区间仅反映场景抽样，不构成训练随机性的优势证据。standard只有3seed，区间仍不稳定。须同时检查碰撞、完成、进度和回报；区间跨零意味着当前证据未辨别差异，不应写成等效或通用优越。

## 成本与样本效率

worldmodel_pretrain_wall_s是整套8种预测模型预训练墙钟上界，给物理与神经模型同样标记；不能称单个history/physics模型的训练成本。主训练墙钟另计model_generation_wall_s，验证/测试真实步单列，不混入训练预算。

|方法|seed|新增online步|共同offline转移|native总训练交互|shared数据预算交互|梯度minibatches|synthetic生成步|训练墙钟s|模型生成s|预训整套上界s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ppo|0|32768|6775|32768|39543|1024|0|26.857|0.000|0.000|
|wm_ppo|0|32768|6775|39543|39543|1024|32768|68.244|34.302|3.388|
|physics_ppo|0|32768|6775|39543|39543|1024|32768|59.786|29.339|3.388|
|real_reuse_control|0|32768|6775|32768|39543|2048|0|38.164|0.000|0.000|
|wm_no_history|0|32768|6775|39543|39543|1024|32768|67.644|33.891|3.388|
|wm_long|0|32768|6775|39543|39543|1024|97846|131.599|98.293|3.388|
|ppo_total_budget|0|39543|6775|39543|46318|1236|0|32.409|0.000|0.000|
|ppo|1|32768|6775|32768|39543|1024|0|33.901|0.000|0.000|
|wm_ppo|1|32768|6775|39543|39543|1024|32768|72.394|36.879|3.710|
|physics_ppo|1|32768|6775|39543|39543|1024|32768|65.321|32.084|3.710|
|real_reuse_control|1|32768|6775|32768|39543|2048|0|35.965|0.000|0.000|
|wm_no_history|1|32768|6775|39543|39543|1024|32768|60.774|30.311|3.710|
|wm_long|1|32768|6775|39543|39543|1024|97805|129.641|96.940|3.710|
|ppo_total_budget|1|39543|6775|39543|46318|1236|0|39.685|0.000|0.000|
|ppo|2|32768|6775|32768|39543|1024|0|35.543|0.000|0.000|
|wm_ppo|2|32768|6775|39543|39543|1024|32768|54.735|27.477|3.027|
|physics_ppo|2|32768|6775|39543|39543|1024|32768|67.276|33.030|3.027|
|real_reuse_control|2|32768|6775|32768|39543|2048|0|35.705|0.000|0.000|
|wm_no_history|2|32768|6775|39543|39543|1024|32768|66.429|33.252|3.027|
|wm_long|2|32768|6775|39543|39543|1024|97778|123.497|92.571|3.027|
|ppo_total_budget|2|39543|6775|39543|46318|1236|0|42.264|0.000|0.000|

纯PPO原生不需要offline数据，因此native总成本可低于shared预算视角；提供相同采集成本预算不等于PPO使用了这些数据。synthetic步不是real env steps，额外复用梯度不是免费收益。缺失成本字段一律未验证，不用配置反推actual。

## 验证学习曲线

val仅ID、每族smoke1/standard2case；四分之一预算监控，不据此挑test-best。左图新增online，中图原生实际消耗的offline+online真实交互；纯PPO、real_reuse_control和ppo_total_budget均原生零offline。右图仅共同预算记账，不能当作真实数据消耗；原生成本推算采用该方法是否使用预测模型与已记录offline转移数。细淡线为独立训练seed，带标记粗线为同一步数跨seed均值，不是置信区间；不能把曲线点当独立训练重复。

![样本效率](../outputs/rl_ppo/standard/report/sample_efficiency.png)

## 证据链与失效边界

需要连接预测能力、分支误差/覆盖、PPO学习曲线、最终闭环与总成本。当前登记评价可以检验policy闭环、成本与消融；若未另记录分支预测误差，不能声称完整机理已经证明。

预测误差降低不保证PPO改善；模型偏差可能被策略放大。OOD、不可预知急刹、有限数据和短分支覆盖仍是边界。零碰撞不构成通用安全保证；主机周期包含仿真step，不代表Pi性能。全部原始日志、成本、按family指标和配对区间在运行目录保留。

正式最终报告必须来自校准后的outputs/rl_ppo/standard且完整性通过；smoke只证明工程路径，未缩放归档只作保留的历史试验。
