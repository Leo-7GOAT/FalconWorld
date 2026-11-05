# PPO验证诊断与一次统一奖励量纲校准

本诊断只使用train与8个登记的ID val场景，不读取test日志、不按test挑选reward scale或策略。原未缩放运行保存在`outputs/rl_ppo/standard_unscaled_validation_trial`，不被新结果覆盖。主策略与旧greedy原始结果保留；不再搜索entropy、网络架构或其他优化器超参数。

可执行脚本为`scripts/diagnose_ppo.py`。已验证的原始产物在`outputs/ppo_validation_diagnostics_verified`：逐episode指标、逐时刻动作分布、reward-scale梯度CSV与summary.json。该目录同时保存未缩放final PPO seed0 checkpoint以及实际运行脚本快照。首次诊断目录`outputs/ppo_validation_diagnostics`也保留；verified版本修正了无裁剪数学对照行的诊断clip系数标注，实际参数步长和验证结果不变。

## Greedy是否掩盖了学到的随机策略

固定未缩放final PPO seed0，使用场景`7000000 + family_index*1000 + [0,1]`，每个case先greedy一次，再在8个预先确定sampling RNG下各运行一次stochastic。greedy共8回合，stochastic共64回合，但两者都只有8个独立场景；不能把64次采样重复当成64个独立道路场景。没有额外训练或策略选择。

|模式|回合数|独立场景|平均raw return|碰撞|平均进度m|平均entropy|+2指令饱和占比|
|---|---:|---:|---:|---:|---:|---:|---:|
|greedy|8|8|-14.1957|4/8|105.056|1.52664|65.88%|
|stochastic|64|8|-19.9498|32/64|102.512|1.51662|27.96%|

greedy访问的同一组观测上，分布的平均期望加速度增量约+0.17275m/s²，argmax动作增量约+0.77907m/s²。将高entropy随机策略转成argmax确实放大了正向指令漂移，且对相对上一指令的增量动作容易积累到+2。但stochastic虽然降低饱和占比，碰撞率仍为50%，平均return还更低。因此当前验证不支持“PPO只是因为用了greedy评价才显得失败”的单一解释。

|val场景族|greedy平均return|stochastic平均return|greedy碰撞率|stochastic碰撞率|
|---|---:|---:|---:|---:|
|constant|74.83|60.13|0|0|
|mild|38.17|28.90|0|0|
|brake|-82.41|-81.20|1|1|
|stopgo|-87.36|-87.63|1|1|

此处是单个已训练seed的有限val描述，不含统计显著性声明。随机策略评价诊断保留，但由于没有观察到验证收益，不为此追加test采样或改选一个更好的策略。

## 奖励量纲与梯度裁剪的数学诊断

在同一未缩放策略下采集固定1024步train rollout，包含4次真正终止、10个episode边界，raw reward范围[-100,0.99339]。原碰撞惩罚-100与dense reward约[-1,1]相差明显。对同一物理transition与behavior动作/logprob，分别重算scale1与scale0.01的GAE；为使旧critic进入正确量纲，诊断中仅把其最后一层weight/bias乘scale，next_values同样乘scale。这个转换只用于数学诊断，不保存为新策略，更不使用旧Adam状态声称跨量纲resume。

critic值与reward同时缩放时，GAE近似线性缩放；minibatch advantage标准化后，actor原始梯度几乎不变。critic梯度量级则显著改变，从而改变同一全网clip因子。表中无有效裁剪对照仅用于分离实现机制，没有被训练、验证成候选方法。

|训练reward scale|clip上限|value MSE|actor原始梯度范数|critic梯度范数|全网clip系数|实际actor参数步长L2|动作概率变化平均L1|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|0.5|280.210|0.107544|41.3139|0.0121024|0.00699904|0.00635698|
|1|1e9（数学无裁剪对照）|280.210|0.107544|41.3139|1|0.0210983|0.00996912|
|0.01|0.5|0.028021|0.107544|0.295850|1|0.0210983|0.00996912|
|0.01|1e9（数学无裁剪对照）|0.028021|0.107544|0.295850|1|0.0210983|0.00996912|

实测默认clip下，scale0.01的actor参数步长约为scale1的**3.01倍**，动作概率变化约1.57倍。不能把原clip系数约1/82直接说成“actor的Adam更新被压小82倍”：Adam的二阶矩归一化对共同梯度缩放具有近似抵消作用，epsilon与各坐标梯度大小仍影响实际步长。这里采用fresh Adam的一步控制诊断，不能据此推断历史Adam状态下每一步都恰好受压3倍，更不能称它证明了全部闭环失败原因。

## 一次允许的校准及边界

该train/val证据提供了统一奖励量纲校准的数值理由，尚未证明新训练会取得驾驶收益。只实施一次：所有PPO方法的真实和模型训练reward统一乘0.01；critic和GAE都使用缩放后的量纲；评价继续使用相同raw reward。保持原动作、物理任务、PPO架构、entropy、梯度裁剪、学习率、真实步数和各方法梯度预算。共享世界模型数据与动力学拟合不依赖训练reward尺度，可复用并保留摘要。

新一轮PPO从相同登记seed重新初始化，不从旧raw-unit optimizer恢复。原未缩放运行及已产生的完整job产物均归档，避免修改或隐去失败；这不表示旧试跑的全部登记job均已完成。新训练结果必须另行完整运行并报告，不能把本梯度诊断当成校准后的性能结论。该决定没有读取test反馈；不继续搜索其他scale、entropy或网络结构。

复跑诊断时使用新的空输出目录，例如：

```powershell
.venv/Scripts/python.exe scripts/diagnose_ppo.py --output outputs/ppo_diagnostic_recheck
```

脚本默认读取诊断目录中冻结的未缩放checkpoint，避免新标准目录已被替换为scaled训练后误用不同策略。checkpoint、原始脚本摘要保存在verified目录的summary.json，保全路径及逐CSV摘要见checkpoint_preservation.json；任何复跑都保留之前结果。
