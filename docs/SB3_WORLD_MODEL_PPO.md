# 官方SB3 PPO中的固定预算世界模型分支

`falconworld/sb3_worldmodel_ppo.py`继承已安装的Stable-Baselines3 `2.9.2` PPO，使用其原始`train()`进行全部策略优化。`world_model=None`时直接调用父类；已用相同seed实测，两者训练后全部policy权重逐位一致。旧自研PPO与其负结果保留，不能用新实现改写旧实验结论。

接口：`WorldModelPPO(policy,env,*,world_model=None,model_fraction=.5,model_horizon=1,model_seed=None,**PPO_kwargs)`。官方默认PPO参数不被类改写；主runner统一环境步数和参数。模型分支目前限已验证CPU、25维编码观测与5离散动作，并要求`target_kl=None`避免各方法梯度预算因KL早停发生变化。

## 每轮更新过程

1. 在父类训练前复制当前真实rollout的整个observations池。只使用编码观测，按固定尺度还原6×4历史和上一加速度指令；不读取场景身份、环境脚本、未来状态或其他真实transition字段。
2. 使用独立numpy RNG选择根，同一个尚未更新的当前SB3 policy采样动作、value与log probability。CPU动作采样也采用独立保存的Torch随机状态，生成分支不会消耗之后真实采样的Torch RNG流。
3. 每个分支只通过world model和公共`synthetic_step()`前进，raw reward乘统一训练尺度0.01。真正碰撞终止，不再继续生成；短分支普通切段保留末端当前critic的bootstrap。
4. 对每条完整分支单独做GAE，随后合并。生成数达到待替换数量后，最后一条分支先完整计算GAE再截取多余记录，不把相邻不同根的分支当成连续轨迹。
5. 随机替换当轮buffer一半对应的`observations/actions/values/log_probs/advantages/returns`，检查替换后的旧log概率和value仍匹配冻结behavior policy。原`episode_starts`保持不变。
6. 调用原始`super().train()`。本机SB3 `2.9.2`的train只读取预计算目标，不重算GAE；不得再对混合buffer调用`compute_returns_and_advantage()`。下一轮收集按SB3正常reset buffer。

这里没有历史replay，没有额外梯度更新，也不把合成观测再次当作本轮根池。原真实buffer尺寸保持不变；2048条、10 epoch、batch64时，各方法均为320次Adam更新。SB3的`_n_updates`按epoch计数，类额外读取Adam状态的真实step前后差进行预算核验，避免把10 epoch误报成10次梯度。

`model_horizon=3`可因碰撞提前停止。生成总数包含最后完整分支超出替换量的记录；替换量只计进入优化buffer的样本。它是对官方PPO进行模型经验替换的研究变体，不是原版MBPO。

## 计数与保存

`model_metadata()`返回累计`model_generated`、`model_replaced`、`model_train_calls`、`model_branch_roots`、`model_optimizer_steps`、`model_generation_wall_s`及最近一轮`last_model_metadata`。后者包含真实根池尺寸、生成/替换数、behavior概率误差、value误差和预期/实际梯度次数。生成耗时覆盖模型分支与目标准备及一致性检查，不含父类梯度优化。

`save()`沿用SB3格式，排除外部world model、根RNG和模型采样RNG状态。保存的zip可以通过官方`PPO.load()`评价策略；不需要加载世界模型。此保存范围用于冻结策略评价，不支持宣称模型分支训练可partial resume。模型数据版本和独立NPZ应由runner另行登记，不能通过pickle把模型对象偷偷放进策略文件。

## 实际验证

```powershell
.venv/Scripts/python.exe -m pytest tests/test_sb3_worldmodel_ppo.py -q
```

2026-11-04实际4项通过（6.46秒），包括：no-model与官方policy逐位一致；h3分支GAE/不重算buffer目标/实际梯度预算；64步后重建32步尾段buffer；不可pickle的外部模型被排除且官方PPO.load贪心动作一致；碰撞分支停止并使用训练尺度-1的终止reward。

默认2048/10epoch/batch64完整工作量也已另行实际运行：生成并替换1024条h1样本，Adam实际320步，最大behavior logprob差1.19e-7、value差2.98e-7，生成与目标准备1.402秒。该主机计时不是设备实时性结论。

单步显式Euler的gap_next与ego_speed_next由旧状态决定，所以同根同动作的h1即时reward在这些模型间应相同。学习前车/执行器分量在h1中主要通过next_obs的critic bootstrap影响合成优势；h3可进一步改变递归状态及后续reward，不能把h1结果解释为学习到了不同的即时奖励。

模块验证不构成世界模型改善PPO的证据；新研究必须保留强官方PPO、丰富物理模型辅助、仅学习组件与计算预算对照，并按冻结最终策略完成登记测试。
