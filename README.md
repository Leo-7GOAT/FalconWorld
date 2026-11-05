# FalconWorld

**面向自动驾驶纵向控制的紧凑世界模型与 PPO 强化学习研究。**

FalconWorld 将可辨识物理动力学、学习残差和前车运动历史组合成动作条件世界模型，用真值模拟器观测启动短模型分支，辅助 Stable-Baselines3 PPO 训练。训练后只需策略网络即可输出纵向控制动作。仓库包含训练代码、固定依赖、对照与消融、已训练策略、合成数据、完整正式测试轨迹和独立审计。

当前完成的是**本地仿真研究**。正式 PPO 实验为一维跟驰；二维阿克曼与 CARLA 是前期世界模型＋MPC 工作。R550 实车 PPO 部署、转向控制、摄像头端到端驾驶及 Raspberry Pi 实时控制验收尚未完成。

> 同新增在线步数下，世界模型辅助 PPO 的平均碰撞数减少，但相对纯 PPO 的回报差异 95% 区间跨零，种子差异明显，并有平顺性和计算成本代价。当前证据支持条件性收益，尚未证明稳定总体优势。

[完整数学推导](docs/WORLD_MODEL_DERIVATION.md) · [正式实验报告](docs/SB3_RL_REPORT.md) · [实验协议](docs/SB3_RL_PROTOCOL.md) · [实验与策略文件](results/README.md) · [English overview](docs/PPO_PROJECT_EN.md)

## 系统流程

```mermaid
flowchart LR
    D[独立 train / val 合成数据] --> M[辨识物理＋残差 / 历史世界模型]
    E[真值模拟器] --> R[当前策略在线 rollout]
    R --> B[复制真实观测根池]
    M --> S[冻结行为策略的一步模型分支]
    B --> S
    S --> G[分支独立 GAE / bootstrap]
    R --> X[固定大小混合学习 buffer]
    G --> X
    X --> P[官方 SB3 PPO 优化]
    P --> A[冻结最终 actor]
    A --> C[状态历史 → 加速度增量 → 公共安全过滤]
```

“真实 rollout”来自独立真值模拟器，不来自真实车辆。世界模型参与训练；在线 actor 推理不需要 MPC 搜索或世界模型递归。

## 世界模型的数学形式

状态为 $s_t=(g_t,v_t,v_t^L,a_t)$：净间距、自车速度、前车速度、执行器内部加速度。采样周期 $\Delta t=0.1$ 秒，期望加速度 $u_t\in[-6,2]$ m/s²。由 $\dot g=v^L-v,\dot v=a,\dot a=(u-a)/\tau$ 得到显式 Euler 转移。

物理执行器考虑加速/制动不对称与速度相关响应：

$$
\tau_t=\begin{cases}\tau_+,&u_t\ge a_t,\\\tau_-,&u_t<a_t\end{cases}(1+\kappa v_t),\qquad
a^{\mathrm{phys}}_{t+1}=a_t+\frac{\Delta t(u_t-a_t)}{\max(\Delta t,\tau_t)}.
$$

$(\tau_+,\tau_-,\kappa)$ 仅在 train 上用一步转移最小二乘辨识。ID 合成数据约得 $(0.45,0.75,0.045)$，不是小车实测参数。

模型包含两个小型 tanh MLP：自车残差头 4→32→1，前车历史头 6→32→1。

$$
z_t=[v_t,a_t,u_t,u_t-a_t],\qquad
q_t=[v_t^L,v_{t-4}^L-v_{t-5}^L,\ldots,v_t^L-v_{t-1}^L].
$$

$$
\widehat a_{t+1}=\operatorname{clip}(a^{\mathrm{phys}}_{t+1}+R_\omega(z_t),-6,2),\qquad
\widehat{\Delta v_t^L}=\operatorname{clip}(L_\eta(q_t),-0.6,0.2).
$$

完整世界模型为

$$
\widehat s_{t+1}=\begin{bmatrix}
g_t+\Delta t(v_t^L-v_t)\\
\max(0,v_t+\Delta t a_t)\\
\max(0,v_t^L+\widehat{\Delta v_t^L})\\
\widehat a_{t+1}
\end{bmatrix}.
$$

**速度更新使用旧 $a_t$；前车头输出速度增量，不再乘一次 $\Delta t$。** 六帧历史因果递归更新，不读取真实未来、场景标签或隐藏条件。网络按标准化的一步监督 MSE 拟合；尺度只来自 train，val 早停，test 隔离。当前物理模型已准确解释 ID 自车动力学，残差约为数值零，这一负结果保留。

这是低维确定性状态转移世界模型，不生成 RGB 视频，也不等同于 Dreamer/RSSM 潜变量模型。参数辨识、标准化、损失函数、单位和递归推导见 [数学文档](docs/WORLD_MODEL_DERIVATION.md)。

## 世界模型如何辅助 PPO

观测为六帧四维状态加上一次实际加速度命令，共 25 维；采用固定尺度，不裁剪 OOD 输入。Actor、critic 分别使用 64×64 tanh 隐层。Actor 输出五分类加速度**增量**：

$$
\Delta u_j\in\{-0.8,-0.4,0,0.4,0.8\},\qquad
u_t=\mathcal S\!\left(s_t,\operatorname{clip}(u_{t-1}+\Delta u_j,-6,2)\right).
$$

$\mathcal S$ 为公共安全过滤器，不是形式化安全保证。当前策略不输出航向或转角；$a_t$ 是仿真特权输入，实车需要另外估计。

每轮先采集 2048 条在线模拟器转移，再用**同一个尚未更新的行为策略**从本轮观测池抽根，生成一步模型分支。主方法用 1024 条合成样本替换学习 buffer 的一半；真实轨迹与每条模型分支分别计算 GAE，再调用官方 PPO 优化器。没有额外策略更新，不把不同分支拼成连续轨迹，也不混入旧策略离线 replay。

$$
\delta_i=\bar r_i+\gamma(1-d_i)V_{\rm old}(o_{i+1})-V_{\rm old}(o_i),\qquad
\widehat A_i=\delta_i+\gamma\lambda(1-b_i)\widehat A_{i+1}.
$$

$d_i$ 表示真正终止，$b_i$ 切断跨 episode/分支的优势链；碰撞不 bootstrap，时间截断及非终止分支末端保留最终状态 bootstrap。训练统一 $\bar r=0.01r$，评估使用未缩放原始回报；$\gamma=0.99,\lambda=0.95$。

$$
\rho_i(\phi)=\frac{\pi_\phi(j_i\mid o_i)}{\pi_{\rm old}(j_i\mid o_i)},\qquad
L^{\rm clip}=\mathbb E_{D_{\rm mix}}\left[
\min\{\rho_i\widehat A_i,\operatorname{clip}(\rho_i,0.8,1.2)\widehat A_i\}\right].
$$

实现沿用 SB3 的 minibatch 优势标准化、value MSE、Adam 和梯度裁剪。主方法 buffer、epochs、batch 和实际 Adam 次数与纯 PPO 相同。混合样本分布和短分支 GAE 改变了训练目标，因此它是模型辅助 PPO 变体，不能宣称原始 PPO 无偏性或 MBPO 性能定理直接适用。

**一步分支的具体作用：**显式 Euler 下，同根同动作的下一间距和自车速度一致，即时奖励也一致。世界模型预测的下一前车速度/执行器状态主要通过 critic bootstrap 改变优势目标。三步分支还会改变后续奖励。不能将一步结果解释为“学到了更好的即时奖励”。

## 方案优势与边界

| 设计 | 潜在价值 | 本项目证据与限制 |
|---|---|---|
| 物理先验＋小型学习头 | 减少未知量，可解释，CPU 可训练 | ID 自车残差近零；尚无额外自车动力学收益 |
| 真实观测根上的短分支 | 不新增真值环境交互，生成当前策略局部反事实目标 | 一步均值好于三步消融；离线采集与生成时间仍需计费 |
| 同步冻结行为策略 | 分支 old log probability 可核验 | 不能消除模型状态分布偏差和 GAE 时域差异 |
| 训练用模型、推理用 actor | 在线执行不必递归模型或搜索候选轨迹 | 官方 `PPO.load()` 已验证；Pi 时延、车辆控制未验收 |
| 对照、消融、完整轨迹 | 可核查收益来源、预算和失败种子 | 相对纯 PPO 主回报区间跨零，不支持稳定总体优越 |

在固定非终止区域、确定性转移且 value 为 $L_V$-Lipschitz 等假设下，由三角不等式有一步目标偏差界：

$$
|\widehat{\mathcal T}^{\pi}V-\mathcal T^{\pi}V|
\le\mathbb E_{j\sim\pi}\left[|\widehat r-r|+\gamma L_V\|\widehat F-F\|\right].
$$

同动作序列下，若单步误差至多 $\varepsilon$、真实转移 Lipschitz 常数为 $L_F$，则 $e_k\le\varepsilon\sum_{j=0}^{k-1}L_F^j$。短分支限制开环误差传播，但更依赖 critic；碰撞边界不连续，不能直接套全局平滑界。这是有条件误差分析，**不是性能改善证明**；完整假设见 [推导第 4 节](docs/WORLD_MODEL_DERIVATION.md#4-为什么有潜在帮助又为什么不保证改善)。

## 正式实验

场景为匀速、温和变速、急刹、启停；每回合最多 120 步（12 秒）。ID 初始前车速度 6–13 m/s，OOD 为 14–19 m/s 并将执行器时间常数乘 1.45。这些速度、尺度是合成基准设定，不能直接用于低速 R550。

3 个训练种子 × 7 方法，共 21 个任务、1680 个测试回合。每方法每分布 120 回合，train/val/test 按独立完整 episode 种子隔离；各方法使用配对测试条件。固定最终 checkpoint 评价，不按 test 挑权重。

| 方法 | 含义 | ID 碰撞 / 120 | ID 原始回报 | OOD 碰撞 / 120 | OOD 原始回报 |
|---|---|---:|---:|---:|---:|
| `ppo` | 官方默认优化配置 | 21 | 42.77 | 28 | 26.08 |
| `wm_ppo` | 历史世界模型，一步替换 50% | **2** | 58.77 | **11** | **50.28** |
| `physics_ppo` | 丰富辨识物理模型，同分支规则 | 29 | 32.85 | 37 | 10.62 |
| `real_reuse_control` | 纯 PPO，20 epochs，双倍优化 | 6 | **59.08** | 28 | 20.81 |
| `wm_no_history` | 删除前车速度差历史 | 27 | 33.47 | 25 | 29.68 |
| `wm_long` | 历史模型，三步分支 | 15 | 45.65 | 27 | 26.60 |
| `ppo_total_budget` | 离线采集预算换为额外在线步 | 21 | 43.34 | 28 | 27.86 |

主比较每任务 65,536 新增在线步、10,240 次实际 Adam 更新。公共离线数据为 5,385 train＋1,390 val 转移，只拟合世界模型，不预训练 actor/critic。总交互控制为 72,311 在线步、11,300 次 Adam 更新；额外更新控制为 20,480 次。

训练 seed → 场景族内配对 episode 的分层 bootstrap（2000 次）结果：

- `wm_ppo − ppo` 原始回报：ID **+16.00，95% CI [−7.30, 59.59]**；OOD **+24.20，[−27.45, 90.02]**。均跨零，只有 3 个训练种子，尚未确立稳定总体优势。
- 相对 `physics_ppo`，OOD 回报差 **+39.66，[18.51, 66.99]**，仅为该合成基准的条件性证据。
- ID jerk RMS 差 **+0.672 m/s³，[0.138, 1.329]**，平顺性变差；共享主机平均训练时间约 **145.38 vs 98.63 秒（+47.4%）**，模型池预训练成本另计。这不是 Pi 计时。
- 收益很大部分来自改善纯 PPO seed 1 的失败；seed 0 的 OOD 回报反而从 42.66 降为 12.75。全部种子均保留。

另完成纯 PPO 的 204,800 步 × 3 seed 参考和 240 个测试回合：ID 碰撞 13/120、回报 52.63；OOD 碰撞 29/120、回报 21.80。它不是等预算优势对照，且仍有种子不稳定。多组比较未做多重检验校正；零碰撞样本不等于零风险。

![七方法验证学习曲线，包含在线、原生总交互与共享数据预算横轴](results/sb3_standard/report/sample_efficiency.png)

全部逐种子/逐场景指标、配对区间、成本和失败分析见 [正式报告](docs/SB3_RL_REPORT.md)。独立审计通过表示登记实验完整、日志指标一致，不代表已满足部署门槛。

## 快速复现

已验证环境：Windows、Python 3.11.9、CPU PyTorch 2.15.0+cpu、Stable-Baselines3 2.9.2、Gymnasium 1.4.0。NumPy/SciPy 紧凑模型不要求 GPU。36 项运行依赖锁定于 [requirements-rl-lock.txt](requirements-rl-lock.txt)；已在原主机实际安装并核对，未声称独立干净机器安装验收。

在项目根目录 PowerShell：

```powershell
.\setup_rl.ps1
.\setup_rl.ps1 -VerifyOnly
.\.venv\Scripts\python.exe -m pip install pytest==9.1.1
.\.venv\Scripts\python.exe -m pytest -q

# 小预算流程验证，不支持统计优势结论
.\.venv\Scripts\python.exe reproduce_sb3_rl.py --preset smoke --workers 3

# 正式：测试 → 训练 → 冻结评价 → 独立审计 → 报告
.\.venv\Scripts\python.exe reproduce_sb3_rl.py --preset standard --workers 3
```

既有完整测试为 130 项与 31 个子检查通过，本次发布验证见 [发布记录](docs/PUBLICATION.md)。发布结果位于 `results/`；复现生成新的 `outputs/`，避免直接复用发布结果冒充重跑。只复用指纹一致的完成任务；中断任务归档后重训，不支持任意中间时刻精确恢复。

加载已发布最终策略做仿真评价，或独立核查发布轨迹：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sb3_policy.py --checkpoint results/sb3_standard/seed0/wm_ppo/final_policy.zip --output outputs/published_policy_check --split val
.\.venv\Scripts\python.exe scripts/audit_sb3_rl.py --run results/sb3_standard --output outputs/published_standard_audit
.\.venv\Scripts\python.exe scripts/audit_sb3_reference.py --run results/ppo_reference_204800 --output outputs/published_reference_audit
```

评价目录必须是新目录，重复评价换名字。全部 seed 均发布，seed 0 仅是命令示例。策略 zip 可用官方 `PPO.load()` 加载；它不包含世界模型和模型 RNG，用于冻结推理评价。

## 代码与前期工作

```text
falconworld/                  动力学、世界模型、RL 环境、PPO 分支、MPC 与适配
run_sb3_rl.py                 七方法训练、固定最终策略评价
run_sb3_reference.py          较充分纯 PPO 预算参考
reproduce_sb3_rl.py           一键测试、训练、审计、报告
scripts/                     诊断、审计、评价、导出、设备准备
tests/                       模型、终止语义、概率、预算和审计测试
docs/                        数学推导、协议、报告与负结果
results/sb3_standard/         21 策略、合成数据、1680 测试回合与轨迹
results/ppo_reference_204800/  3 策略、240 测试回合与轨迹
```

前期世界模型＋CEM-MPC 复现见 [README_MPC.md](README_MPC.md)、[纵向模型](docs/MODELS.md)、[二维阿克曼](docs/ACKERMANN.md)、[CARLA](docs/CARLA.md)，其结果不与 PPO 合并。旧自研 PPO/DQN 代码及负结果文档保留，正式对照以本轮 SB3 为准；旧 `outputs/` 全量历史未上传。

R550 接入见 [部署准备](docs/R550.md)。仓库不发布个人原始车辆日志或 SSH 地址。仿真加速度不能直接当作底盘串口字段；还需低速标定、感知与状态估计、控制适配、停车/接管/超时保护以及 ARM64 时延测试。

## 参考研究

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017：PPO clipped surrogate。
- Janner et al., [When to Trust Your Model: Model-Based Policy Optimization](https://jannerm.github.io/mbpo-www/), NeurIPS 2019：短模型分支与模型偏差相关研究；本仓库不复现其完整算法或定理。
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)：库接口；实验版本以本仓库锁文件和登记记录为准。
