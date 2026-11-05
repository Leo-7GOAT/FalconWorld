# 紧凑 Double DQN 与真实/模型经验混合

`falconworld/rl_agent.py` 实现CPU上的25维输入、64×64 ReLU网络、5个离散动作的Double DQN。该模块实际优化Q值并产生策略；与此前仅使用MPC的项目不同，这条新增研究路线属于强化学习。它不是SAC，也不冒称复现原版MBPO。

Double DQN使用在线网络选择下一动作，再由目标网络评价该动作。目标为
`r + gamma * (1 - terminated) * Q_target(next_obs, argmax Q_online(next_obs))`。
这一动作选择/评价分离参考[van Hasselt 等人的 Double DQN 论文](https://arxiv.org/abs/1509.06461)。本实现采用Huber损失、Adam、梯度范数裁剪和周期硬更新目标网络；目标网络不接受梯度。CPU PyTorch线程固定为1。

## 策略与动作接口

`AgentConfig` 默认：25维输入、5动作、两层64单元、batch128、gamma0.99、学习率3e-4、每250次梯度更新同步目标、梯度范数上限10。epsilon按训练动作调用次数，从1线性降至0.05，默认20000步。runner负责统一观测编码、奖励、终止和实验总更新预算；本模块不读取场景脚本、真值未来或模型标签。

```python
from falconworld.rl_agent import AgentConfig, DoubleDQNAgent, ReplayBuffer, ACTION_DELTAS

agent = DoubleDQNAgent(AgentConfig(seed=0))
real = ReplayBuffer(100000, seed=100)
model = ReplayBuffer(100000, seed=200)
# obs/next_obs: runner统一生成的25维float32编码；previous_u: 上一步m/s²指令。
index = agent.act(obs, explore=True)
command = float(__import__('numpy').clip(previous_u + ACTION_DELTAS[index], -6, 2))
real.add(obs, index, reward, next_obs, terminated, truncated=truncated)
info = agent.update(real, model, synthetic_fraction=0.5)
```

五动作增量为 `[-0.8,-0.4,0,0.4,0.8] m/s²`，相对上一指令；runner在相同动作范围内限幅。`act()`返回离散index，不直接发送车辆控制量。二维输入批量调用返回index数组；单个输入返回int。`explore=False`始终greedy，平手使用固定argmax顺序，不消耗epsilon步数或探索RNG。随机策略/经典控制基线由runner显式实现和评估，不将训练epsilon策略当作确定性DQN评估。

## Replay与终止

`ReplayBuffer.add(obs,action,reward,next_obs,terminated,*,truncated=False,...)` 保存拷贝；uniform sample不放回，返回numpy字典。真正终止与时间上限分开记录。只有`terminated`阻断bootstrap；`truncated=True, terminated=False`继续bootstrap。截断transition的next_obs必须是reset前的最终观测，不能用新episode的初始帧替代。

可选metadata必须同时提供`raw_history`、`next_raw_history`、`previous_action`、`next_previous_action`；前两者默认形状6×4，后两者为实际连续加速度指令。它们仅供模型分支起点与日志使用，不额外输入Q网络；`raw_history_valid`显式标记可用性。未提供metadata的槽位会清零，ring覆盖不会遗留上一条transition的原始状态。模型分支生成器只应从valid的真实经验起步，不向策略泄漏隐藏环境参数。

`update(real_buffer,model_buffer=None,synthetic_fraction=0,batch_size=None)` 每次最多执行一次梯度更新，总batch固定。模型样本数为`round(batch_size*fraction)`，其余来自真实buffer；不会由于存在模型经验而增加更新次数。指定任一来源不足时返回`updated=False, skip_reason='insufficient_replay'`，不偷偷降低比例或使用更小batch。runner必须以实际成功的`gradient_updates`核对各方法预算。随机采样源各有独立RNG，合成数据生成开销应另记。

## Checkpoint与恢复范围

`agent.save_checkpoint(path,real_buffer=None,model_buffer=None,metadata=None)` 保存在线/目标网络、Adam状态、探索RNG、Torch RNG、动作步数、梯度更新数和配置。replay快照可显式加入，不默认将大型buffer写盘。先完整写同卷临时文件，再以硬链接原子发布；目标已存在则失败，绝不覆盖已有正式checkpoint。

`DoubleDQNAgent.load_checkpoint(path,load_replay=True)` 返回agent；可用快照恢复为`restored_real_buffer`与`restored_model_buffer`，metadata在`checkpoint_metadata`。checkpoint仅从可信本地来源加载，因为优化器和numpy replay状态使用Torch的pickle容器。未保存replay时不能声称可完整继续训练；即使保存了两个buffer，也仍未保存runner、环境、场景采样器和外部世界模型生成状态，所以这里只保证agent/replay恢复一致性，不保证整个实验中断恢复。新实验目录与源/数据指纹由runner管理。

测试入口（需要当前`.venv`内PyTorch CPU和pytest）：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_rl_agent.py -q
```

测试覆盖在线选动作/目标评价的Double DQN目标、terminated与truncated区别、真实梯度变化及裁剪、目标同步、同总batch的真实/合成混合、greedy不消耗探索、raw metadata覆盖边界、checkpoint动作/RNG/优化器/replay及下一次更新一致性。训练效果和世界模型是否改善策略，必须由独立runner的固定预算、多seed、未见测试场景闭环结果检验，不能由本模块测试通过推断。

2026-11-04已在本工程`.venv`、PyTorch `2.15.0+cpu`实际执行上述命令，5项通过（6.65秒）。系统Python未安装Torch，不能把两个解释器混用。当前验证仅针对CPU agent/replay组件，不表示已完成强化学习实验或证明世界模型改善策略。
