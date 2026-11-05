# 纵向 RL 环境契约

`falconworld/rl_env.py` 提供无 Gym 依赖的 NumPy 环境。它与已有世界模型共用动力学、场景和独立安全监督，属于特权状态合成实验，不能视为车端观测方案。既有 `data.py`、`models.py`、`planning.py` 未作修改。

## 接口与单位

```python
from falconworld.rl_env import LongitudinalRLEnv, encode, decode_observation, synthetic_step

env = LongitudinalRLEnv()
raw_obs, info = env.reset(seed=0, family='constant', condition='id')
features = encode(env.history, env.previous_action)
raw_next, reward, terminated, truncated, info = env.step(2)
```

`reset` 的 seed/family/condition 必须显式提供，family 为 constant/mild/brake/stopgo，condition 为 id/ood。状态每帧为 `[gap m, ego m/s, leader m/s, actuator m/s²]`，六帧按时间从旧到新展平为24维，再附上一实际执行加速度指令[m/s²]，共25维。reset重复初态六次；每次step丢弃最旧帧、追加真实下一帧。`history`/`state` 返回副本，另有 `previous_action` 和 `tick` 属性。兼容名称 `raw_history`、`previous_u` 可用。

`encode(history,previous_u)` 单独把每帧除以 `[50,20,20,6]`、指令除以6，输出float32。支持前置batch维度。尺度是预先声明的物理量级，不用训练/验证/测试统计，不裁剪，保留OOD超尺度值。`decode_observation` 仅解包原始25维观测，不能用于已编码特征。

动作索引0–4对应delta `[-0.8,-0.4,0,+0.4,+0.8]` m/s²；先对上一实际执行指令加delta，clip到[-6,2]，再交原有 `SafetySupervisor`。监督可以直接覆盖为-6，不受普通delta限幅；这次覆盖必须记录，不能把模型/策略本体的安全性与监督混为一谈。下一步previous指令使用实际执行值。所有方法共享相同映射与监督。

## 精确 reward（longitudinal-rl-reward-v2-bounded-survival）

以下状态量来自下一状态，u为监督后实际执行指令，p为上一实际执行指令：

- `gap_cost = .01 * clip(gap-(4+1.2*v),-50,50)^2`
- `speed_cost = .1 * clip(v-12,-20,20)^2`
- `action_cost = .01*u^2`
- `slew_cost = .1*(u-p)^2`
- `collision_cost = 100 * (gap<=0)`
- `reserve_cost = 10 * clip(max(0,2+.5*v-gap),0,20)^2`

`dense_cost` 为六项之和；非碰撞 `reward = 1 - min(2,dense_cost/10)`，范围[-1,1]；碰撞直接reward=-100。保留各cost分项及dense_cost，便于检查饱和。误差边界为预设仿真工程选择，并非真实车辆标定。奖励不是MPC原cost，也不是交通安全保证。

采用gamma=.99时，120步持续-1的折扣总和约-70.06；单次碰撞-100避免策略仅通过早碰撞终止来节省后续负reward。这个设计不能证明策略绝不会投机，必须同时报告碰撞、进度、完成、跟踪与回退。控制器比较不得只看return。

## 终止、TimeLimit与统计

`terminated` 仅为gap<=0，保留负gap；`truncated` 为无碰撞且达到120步（12s，覆盖stopgo重启）。同一步碰撞优先terminated，truncated=false。结束后必须reset，继续step报错。碰撞的TD bootstrap mask=0，TimeLimit的mask=1；截断时使用返回的最终观测bootstrap，不能误用下一次reset观测。训练合成分支的长度限制也不是物理终止。

step返回 `(raw_next_obs,reward,terminated,truncated,info)`。info含：

- `state`/`next_state`、`raw_history`：真实环境下一状态与更新历史（仅采集/评价，不另给policy泄露信息）；
- `action_index`、`requested_action_mps2`、`applied_action_mps2`/`applied_u`/`command`、`fallback`；
- gap/ego/lead单位指标，`progress_m`为本步自车位移，`cumulative_progress_m`与`reference_progress_m`为累计自车/前车位移；
- `measured_accel_mps2=(v_next-v_now)/dt`，`jerk_mps3`由实际速度差分加速度再差分，停车边界不使用内部负执行器替代实测加速度；
- `reward_costs`、`dense_cost`、`reward_version`、`collision`、`completed`、`steps`、`time_s`、`bootstrap_mask`。

结束时info另含 `episode`，包括return、length、collision、progress/reference_progress、fallback_count、gap_rmse、jerk_rms。完成仅指跑满12s且自车进度至少为前车参考一半（参考下限5m）；碰撞不能完成。必须同时看进度，不能依赖停车制造安全表现。

## 合成训练转移与信息边界

```python
next_history, reward, terminated, info = synthetic_step(model, raw_history, previous_u, index)
```

只把当前历史最后状态、实际执行指令、历史副本交给 `model.step`；不读场景类型、前车脚本、未来加速度、隐藏condition或真实下一状态。模型下一状态递归更新历史，使用同样动作映射、监督和reward。返回原始下一历史，不返回编码obs，以便继续递归；下一previous指令取info['applied_u']。合成函数没有时间步预算，只返回terminated；`info.truncated=false`，rollout调用方自行处理分支长度、模型不确定性和采样来源，不把人为分支截断当碰撞。

真实脚本只在环境 `step` 内读取。synthetic函数不接收family/condition参数。模型异常或负速度等非法输出显式报错，不静默修正成有利训练样本。真实与合成info以 `transition_source=truth/model` 区分，必须在buffer与统计保留采样来源。相同seed/family/condition生成相同外生随机性，不代表策略产生相同真实状态轨迹。

## 已运行验证

```powershell
python -m pytest tests/test_rl_env.py -q
```

9项检查通过：固定编码与OOD不裁剪、动作/监督、精确reward及batch、8类ID/OOD场景可复现、真实与合成单步/递归完全一致的零前车加速度契约（测试模型）、合成禁止读取脚本、120步TimeLimit bootstrap、接触终止及负gap、停车jerk与非法输入。这里没有宣称学习模型已匹配真值；等价测试只核验共享动作/reward/history管线。

## Gymnasium / Stable-Baselines3桥接

`falconworld.rl_gym.LongitudinalGymEnv`（兼容别名`GymLongitudinal`）包装原环境，不修改原动力学/reward/监督。构造参数：`training_seed=0, seed_namespace=6000000, reward_scale=.01, log_path=None`。Box观测是固定尺度编码25维float32，无裁剪；动作空间Discrete(5)。训练返回reward乘`.01`，info的`raw_reward`及`episode_metrics`保留原单位。需要评估时继续使用原`evaluate_policy`与原环境，不能把训练缩放后的回报当raw return。

物理episode seed为`seed_namespace+training_seed*100000+episode_counter`，家族按counter%4轮换。Gym的显式`reset(seed=...)`重启该已配置episode流，自动`reset(seed=None)`递增counter；配置决定物理namespace，Gym seed不覆盖该namespace。不同训练seed使用独立物理流。check_env应使用独立wrapper实例，其检查步不计入训练预算。

属性包括`history`/`state`/`previous_u`、实际`transition_count`、原始`episode_metrics`/`episodes`及`last_transition`。step info额外提供`raw_history_before`、`previous_u_before`、`raw_history_next`、`raw_reward`、`reward_scale`，用于模型根与诊断，不能作为额外policy输入。`log_path`可选，若提供则写新JSONL文件并拒绝覆盖；默认不写大文件。

wrapper不自动reset，不把TimeLimit当碰撞。SB3的`DummyVecEnv`负责保留`terminal_observation`和`TimeLimit.truncated`，使其on-policy收集器在真实最终观测上bootstrap。`tests/test_rl_gym.py`三项已通过：Gym check_env、同case/动作与原环境逐步完全一致、family/namespace、最终观测及TimeLimit语义、原始日志单位。
