# PPO主算法与世界模型辅助策略训练

用户选定PPO作为当前强化学习主算法。`falconworld/ppo_agent.py` 是categorical PPO actor-critic；旧 `rl_agent.py` 的Double DQN仅为历史原型，不是本轮主对比。该模块学习策略与价值函数，不涉及实车部署或实时控制验收；模型辅助PPO也不冒称原版MBPO。

实现参考 [PPO论文](https://arxiv.org/abs/1707.06347) 的裁剪代理目标及 [GAE论文](https://arxiv.org/abs/1506.02438) 的优势估计。策略与价值分别采用25输入、两层64单元tanh；策略输出5类logits，价值输出标量。隐藏层正交初始化，策略末层增益0.01、价值末层1。PyTorch仅使用CPU、单线程。

## 固定配置与接口

`PPOConfig(seed=0,...)` 默认：学习率3e-4、ratio裁剪0.2、value系数0.5、entropy系数0.01、梯度范数上限0.5、gamma0.99、GAE lambda0.95。value loss采用未裁剪的预测值与return均方误差；总损失为 `clipped_policy_loss + 0.5*value_MSE - 0.01*entropy`。每个minibatch统一标准化advantages，不独立缩放真实与模型样本。

```python
from falconworld.ppo_agent import PPOConfig, PPOAgent, compute_gae, ACTION_DELTAS

agent = PPOAgent(PPOConfig(seed=0))
action, log_prob, value = agent.sample(encoded)
# 候选离散动作相对上一加速度：[-.8,-.4,0,.4,.8] m/s²。
# runner统一执行clip(previous_u + ACTION_DELTAS[action], -6, 2)。
greedy_action = agent.act(encoded, explore=False)
estimated_value = agent.value(encoded)
advantages, returns = compute_gae(rewards, values, next_values,
                                  terminated, episode_end, .99, .95)
```

`sample()` 单条返回 `(int,float,float)`；批次输入 `(B,25)` 返回三个numpy数组。`act(...,explore=False)` 固定argmax、平手固定顺序，不消耗探索RNG；`explore=True`从categorical分布采样。`value()`单条返回float、批量返回数组。动作增量与观测编码/奖励定义由runner统一，策略不能接收隐藏场景脚本或真实未来。

## 当轮behavior数据与公平更新预算

更新字典必须包含 `obs, actions, old_log_prob, advantages, returns, old_values`，除obs为 `(N,25)` 外其余均 `(N,)`。可带整数 `policy_version`；该版本必须匹配agent当前版本。agent每完成一轮update后版本加1。

```python
metrics = agent.update(real_dataset,
    total_minibatches=16, batch_size=128,
    synthetic_fraction=0.5, model_dataset=model_dataset)
```

真实rollout与模型短分支均须在本轮任何梯度更新前，由同一冻结behavior策略采样动作、记录log概率和value。agent首次更新前验证两类数据的old_log_prob和old_values与当前网络一致；过时数据会失败，不接受历史replay。`compute_gae`不自行生成经验。模型数据本身会有动力学偏差，PPO比率仅修正策略变化，不能消除模型误差。

每个梯度批次固定128样本，模型样本数为 `round(128*synthetic_fraction)`；各方法必须同时核对真实环境步数、梯度批次数、总样本数和batch大小。每批从当轮数据均匀采样；单个来源不足指定数量时明确采用有放回抽样，数量足够时批内不放回，批次之间允许重复。此设计是固定minibatch预算PPO，不保证每条rollout样本在每个epoch恰好使用一次。没有KL早停或隐含额外梯度更新；KL和clip fraction只记录诊断。

`update()`返回实际minibatches、batch_size、real_samples、synthetic_samples、累计gradient_updates、policy_version及损失、entropy、近似KL、clip fraction和裁剪前梯度范数。增加真实经验重复更新的控制组应明确计数，不能冒充与主对比相同计算预算。只加世界模型短经验的组保持主对比同样总梯度更新数和batch，避免把额外优化次数说成模型收益。

## GAE边界

`compute_gae(rewards,values,next_values,terminated,episode_end,gamma,gae_lambda)` 支持 `(T,)` 或时间优先 `(T,N)`，返回float32的 `(advantages,returns)`。

TD残差为 `r + gamma*(1-terminated)*V(next_obs) - V(obs)`；GAE递推使用 `gamma*lambda*(1-episode_end)`，真正terminated也强制切断递推。因此：

- 真正终止不bootstrap，且不传播到下一个episode。
- 时间上限截断保留最终观测value bootstrap，但切断lambda递推，不能混入reset后episode。
- 普通rollout切段即使未结束episode，最后一步仍从next_value bootstrap；段后没有收集的TD残差不会凭空加入。

runner提供的next_values必须来自reset前的真实最终观测，合成短分支结束也不能把重置状态作为下一状态。终止mask与episode边界mask是两个不同语义。

## Checkpoint和实际验证

`save_checkpoint(path,metadata=None)` 保存actor-critic、Adam、numpy采样RNG、独立Torch动作采样RNG、gradient_updates与policy_version。先写完整同卷临时文件，再原子发布；目标存在则拒绝覆盖。`PPOAgent.load_checkpoint(path)`返回agent，metadata在 `checkpoint_metadata`。只读取可信本地checkpoint（Torch pickle容器）。

runner、环境、场景采样器和部分收集rollout没有保存；不支持partial实验resume，不用checkpoint存在就宣称可完整恢复一轮训练。加载后评价动作与下一次agent更新可以一致，不等于整个外部训练任务可以中断恢复。

2026-11-04在工程`.venv`、PyTorch `2.15.0+cpu`实际执行：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_ppo_agent.py -q
```

5项全部通过（7.12秒），覆盖正负优势的ratio裁剪、sample logprob/value与behavior一致、终止/截断/rolloutcut GAE边界、真实梯度改变策略与固定混合预算、过期rollout拒绝、checkpoint评价/随机采样/下一次Adam更新一致。测试通过只证明组件契约；世界模型是否改善PPO，仍需runner完成固定预算、多训练seed和未见场景上的对照实验。
