# 纵向数据与模型契约

所有量使用 SI。状态 `[gap, v_ego, v_lead, a_actuator]`，时间步长 0.1 s，历史长度 6 帧且包含当前帧。`model.step(states, actions, history)` 支持单个 `(4,)` 或批次 `(B,4)`；返回相同形状。历史 `(6,4)` 或 `(B,6,4)`。递归预测必须将模型自身输出推入历史，不能插入真实未来帧。当前 `a_actuator` 是仿真特权状态，实车应使用因果估计器，不能视为速度差分在停车处的精确同义量。

`estimate_sensor_state(observation,history,previous_action=None,physics=None)` 显式忽略输入第 4 列，以历史相邻速度差估计上一时刻运动学加速度；可选用已知上一指令和同一校准物理模型向当前传播。所有模型应共用同一个估计器，不能给学习模型特权。停车时无法从恒零速度辨识内部负制动，函数返回的 0 或传播值仅为估计。该模式是理想无噪声速度观测实验，尚不代表真实传感器滤波、延迟或噪声性能。

`make_episode(family, seed, condition)` 和 `scenario_lead_accel(family,t,params)` 仅属于环境，禁止传给模型。前车外生加速度与自车动作独立；学习模型的前车预测头仅接收前车当前速度或过去 6 帧速度，结构上禁止自车动作误作前车制动的原因。

## 仿真假设与强物理基线

没有使用真实小车标定数据。复杂机制仅为加减速响应不对称和速度相关时常：

`tau = (0.45 if u >= a else 0.75) * (1 + 0.045*v_ego)`，单位 s；`a_next=a+dt*(clip(u,-6,2)-a)/tau`。

速度范围 ID 初始 6–13 m/s，OOD 14–19 m/s；OOD 执行器时常额外乘 1.45。都是全尺寸低维教学仿真，不能直接移植 R550。速度非负，gap 采用旧速度显式 Euler，负 gap 保留并终止环境。执行器负加速度在停车后可以仍为负，因此 jerk 应由实际速度差分而非内部状态计算。

`physical_cv` 使用训练转移最小二乘辨识的一个时常和前车恒速。`physical_ca` 额外使用历史相邻速度差推断前车恒加速度。`physical_enriched` 在同一训练集拟合加速时常、制动时常、速度系数，并使用 CA。该模型具有与真值相同的合理结构，是主要强基线；不人为限制物理表达能力。无噪声固定 0.8 s 一阶系统的辨识测试达到浮点精度，保留“参数辨识足以解决该问题”的结论。

## 学习模型

NumPy 实现单隐层 32 单元 tanh MLP、解析反向传播和 Adam。自车预测头输入 `[v_ego,a,u,u-a]`，输出下一执行器状态（neural）或强物理模型残差（residual/history）。已知 gap 和速度运动学仍显式积分，因此 `neural` 表示学习动力学分量的结构化确定性世界模型，并非四维无结构黑盒。

`neural` 和 `residual` 的前车头仅用当前速度；`history` 增加前车速度历史；`learned_lead` 仅替换强物理模型的前车分量，用于定位收益来源。强物理模型已能完全描述 ID 自车机制，残差接近零是正常且应报告的负结果。模型不保证任何一项优于强基线。

`ensemble` 平均三个不同初始化训练的 history 模型；`uncertainty()` 返回成员预测标准差，不能称为校准置信度。`interval_radius()` 为验证集单步绝对误差的 90% 分位数加 1.645 倍分歧。该启发式区间必须用冻结测试集评价实际覆盖率与宽度；不具有递归多步或 OOD 覆盖保证。

多步集成预测采用各成员分别递归自身状态与历史，再对完整轨迹求均值和标准差。`EnsembleModel.step()` 是单步聚合接口，不应反复将该均值写回所有成员的共同历史；非线性模型下这两种运算不同。诊断的 `recursive_trajectory()` 和 `recursive_members()` 已统一此语义，与在线 MPC 成员传播一致。

训练和归一化仅使用 train；每 5 epoch 比较 val 归一化误差，保存最佳权重；早停连续 5 次无改善。smoke/standard/full 最大 epoch 为 80/140/220，固定配置在测试前选定。独立训练种子由调用方传入。所有训练日志包含数据 SHA256、种子、参数、验证轨迹和实际耗时。

## 数据与可移植保存

`generate_dataset(path,preset)` 返回 train/val/test/ood 与 metadata；每集合有 state/action/next_state/history/episode/family/step，逐完整 episode 划分。四场景：匀速、平滑变化、无先兆急刹、停车再启动。每个 ID episode 120 帧或碰撞提前终止；采集动作含随机激励和反馈。manifest 保存生成版本、每回合种子/初值/环境参数/长度和数据摘要。测试标签从不进入训练函数的计算路径。

`train_models(data,path,seed,preset)` 保存 8 个模型，`load_models(path)` 可重新加载。模型格式为无 pickle 的 NPZ，内含显式 JSON 架构与 FP64 NumPy 权重/归一化；仅需 NumPy 推理，适用于 ARM64 Linux，无 x86 机器代码依赖。当前在 Windows 主机完成保存加载逐元素一致验证；这不是 Raspberry Pi 4B 的设备性能验证。选择 NPZ 而非增加 ONNX 运行时依赖，是因网络已由标准矩阵乘法、tanh 和显式运动学构成；部署时仍须测完整 MPC 周期。

独立检查：`python -m pytest tests/test_models.py -q`。完整训练复现由项目统一入口调度；本模块不自行挑选测试表现最佳的种子。

## 2026-11-04 独立复核

`python -m pytest tests/test_models.py tests/test_planning.py tests/test_diagnostics.py -q` 实际通过 14 项。复核未修改 data.py、models.py 或 planning.py 的冻结训练/控制逻辑。信息入口核查确认：训练和物理辨识只读取 train/val；前车头没有自车动作输入；环境脚本、场景类别、隐藏执行器条件不进入规划预测器。线性固定时常辨识、episode 不交叉、因果历史、无 pickle 保存加载、动作敏感性、停车边界和数值失败回退均有检查。

修复诊断中的集成点预测递归：传感消融与候选动作分支原先通过单步均值接口递归，现与在线 MPC 一样保留独立成员历史。新增非线性历史放大回归断言，避免均值反馈使点预测偏离成员轨迹均值。该修改影响后续诊断产物，旧诊断不能冒充新语义结果。

已知科学边界：无历史前车头只能学习当前速度下的平均变化；随机急刹的未知发生时刻不存在可可靠预知的输入。由于强物理基线与 ID 自车生成结构相同，自车残差应近零；收益若出现，主要应由前车历史预测与风险处理解释，并通过 `learned_lead` 与 `history` 的配对结果验证。训练数据速度高于 R550 常见实验速度，任何纵向模型都不能未经重采集/核验直接用于小车跟驰。
