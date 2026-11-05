# 当前SB3模型池的纯验证机制诊断

数据来自 `outputs\rl_sb3\standard/common_data/val.npz`，common SHA `185db49b399d7a3b1ff9c7f74a7888de699301dd70ad8a7ba87bb35289274633`，val文件SHA `760939c93d49aab614686283b495b9bfc9d8324c5a59a682c4f009de5905e1dd`。只读当前3seed冻结world_models和固定主wm_ppo最终critic进行val响应probe；不加载任何test标签、日志或评价数据，不训练，不重新选择策略/模型。

12个独立val episode，按完整episode内非重叠30步窗口，先平均窗口，再按family和seed汇总。3seed共享同一组val场景，不把重复评估当新增独立场景。逐seed的0.1/0.5/1/3秒结果见family_seed_errors.csv。此数据不是旧MPC dataset，禁止套用旧0.0034→1.646m/s数字。

## 三秒递归误差（当前模型池）

|族|模型|gap MAE m|ego MAE m/s|leader MAE m/s|actuator MAE m/s²|
|---|---|---|---|---|---|
|brake|history|3.82391|3.70074e-16|3.14216|2.31296e-17|
|brake|learned_lead|3.82391|3.70074e-16|3.14216|2.31296e-17|
|brake|physical_enriched|3.21572|3.70074e-16|2.78649|2.31296e-17|
|brake|residual|3.21283|3.70074e-16|2.55228|2.31296e-17|
|constant|history|2.58214|1.4803e-16|2.55033|1.02927e-16|
|constant|learned_lead|2.58214|1.4803e-16|2.55033|1.03505e-16|
|constant|physical_enriched|4.44089e-16|1.4803e-16|0|1.03505e-16|
|constant|residual|2.40826|1.4803e-16|1.78484|1.02927e-16|
|mild|history|2.06652|2.96059e-16|1.82302|6.56304e-17|
|mild|learned_lead|2.06652|2.96059e-16|1.82302|6.12936e-17|
|mild|physical_enriched|1.70099|2.96059e-16|1.53957|6.12936e-17|
|mild|residual|2.69497|2.96059e-16|1.90618|6.56304e-17|
|stopgo|history|2.39562|6.56882e-16|1.7436|8.78927e-17|
|stopgo|learned_lead|2.39562|6.56882e-16|1.7436|8.78927e-17|
|stopgo|physical_enriched|1.4033|6.56882e-16|1.65387|8.78927e-17|
|stopgo|residual|4.9599|6.56882e-16|3.3423|8.78927e-17|

## 单步到多步的变化

|族|时域s|history leader MAE m/s|history gap MAE m|physics leader MAE m/s|
|---|---|---|---|---|
|brake|0.1|0.0104335|0|0|
|brake|0.5|0.235718|0.0274772|0.117462|
|brake|1.0|0.746157|0.242074|0.508146|
|brake|3.0|3.14216|3.82391|2.78649|
|constant|0.1|0.00338526|0|0|
|constant|0.5|0.0682029|0.00813454|0|
|constant|1.0|0.306833|0.0823336|0|
|constant|3.0|2.55033|2.58214|0|
|mild|0.1|0.0183576|0|0.0189467|
|mild|0.5|0.12433|0.02143|0.121364|
|mild|1.0|0.355291|0.12342|0.300497|
|mild|3.0|1.82302|2.06652|1.53957|
|stopgo|0.1|0.00381718|0|0|
|stopgo|0.5|0.151177|0.0168568|0.0625|
|stopgo|1.0|0.545169|0.163056|0.166667|
|stopgo|3.0|1.7436|2.39562|1.65387|

## 自车残差与仅学习前车的对照

|seed|model|validation_rows|raw_actuator_residual_mae_mps2|raw_actuator_residual_max_mps2|effective_next_actuator_delta_mae_mps2|effective_next_actuator_delta_max_mps2|
|---|---|---|---|---|---|---|
|0|history|1390|4.837816802479237e-18|4.837816802479238e-18|2.05842412459658e-19|6.938893903907228e-18|
|0|residual|1390|4.837816802479237e-18|4.837816802479238e-18|2.05842412459658e-19|6.938893903907228e-18|
|1|history|1390|4.837816802479237e-18|4.837816802479238e-18|2.05842412459658e-19|6.938893903907228e-18|
|1|residual|1390|4.837816802479237e-18|4.837816802479238e-18|2.05842412459658e-19|6.938893903907228e-18|
|2|history|1390|4.837816802479237e-18|4.837816802479238e-18|2.05842412459658e-19|6.938893903907228e-18|
|2|residual|1390|4.837816802479237e-18|4.837816802479238e-18|2.05842412459658e-19|6.938893903907228e-18|

history与learned_lead在全部登记窗口、四时域和3seed中的预测最大差分别为 `{'gap_m': 0.0, 'ego_mps': 0.0, 'lead_mps': 0.0, 'actuator_mps2': 2.7755575615628914e-17}`，单位由字段标明。此比较定位自车残差是否带来可见变化；完整逐seed记录见history_vs_learned_lead.csv。

## h1即时奖励与bootstrap分量

同一val根+同一实际连续动作、相同previous_u，各模型analytic raw reward相对丰富物理的最大差为0。显式Euler的下一gap与ego速度由旧状态决定，因此h1的学习前车/actuator预测不会改变该步即时奖励。它们可以改变next_obs中lead/actuator，以及固定critic的bootstrap。

|seed|模型|族|reward最大差|nextobs最大差(编码)|同critic nextV平均绝对差|
|---|---|---|---|---|---|
|0|history|constant|0|0.000217468|0.000902127|
|0|history|mild|0|0.00189039|0.00106905|
|0|history|brake|0|0.00345176|0.001723|
|0|history|stopgo|0|0.00286962|0.0022161|
|1|history|constant|0|0.000168741|0.000564728|
|1|history|mild|0|0.00113574|0.000507792|
|1|history|brake|0|0.00317621|0.0012213|
|1|history|stopgo|0|0.00320646|0.00148385|
|2|history|constant|0|0.000287831|0.000676849|
|2|history|mild|0|0.000698715|0.000233415|
|2|history|brake|0|0.00295801|0.000620277|
|2|history|stopgo|0|0.00236526|0.00120282|

各预测模型相对物理的同critic nextV最大差为 `{'physical_enriched': 0.0, 'history': 0.017782092094421387, 'residual': 0.118771493434906, 'learned_lead': 0.017782092094421387}`，单位为训练reward*.01的价值单位；全部模型最大值0.118771包含无历史residual，不能误称都是history的变化。critic跨预测模型共享；没有比较不同critic的绝对标尺，也没有用此probe选择策略。不能将nextV变化本身称为正确性或收益。

## 解释范围

这些val诊断只限定当前PPO研究的能力归因：h1主要通过模型next_obs和value bootstrap起作用，h3还会影响后续奖励。真实数据的长GAE与模型短分支目标时域不同，分支根重采样也改变样本权重；丰富物理辅助控制共享这些机制，更适合区分学习组件和通用分支效应。主研究已有h3/nohistory正式消融，此处不追加实验或调参。

物理模型可精确匹配ID自车结构；无先兆外生急刹无法由6帧历史可靠预知。预测MAE或value响应不是PPO收益的替代证据；闭环正式结果、3seed不确定性、代价和失败必须按独立报告阅读。该分析不支持将旧MPC数据集的数值直接移植为新RL结果原因。
