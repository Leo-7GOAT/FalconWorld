# 世界模型、短分支与 PPO：公式和实现对应

本文件推导当前正式 `wm_ppo`，对应 `falconworld/models.py`、`falconworld/rl_env.py` 和 `falconworld/sb3_worldmodel_ppo.py`。它是低维状态空间中的确定性预测模型，不是视频生成模型；短分支替换是本项目的 PPO 研究变体，不是原版 MBPO，也没有继承 MBPO 的性能改进定理。

## 1. 问题、观测与控制量

连续时间纵向运动的状态、输入为

$$
s=(g,v,v^L,a)^\top,\qquad u\in[-6,2]\;\mathrm{m/s^2},
$$

其中 $g$ 为前车净间距，$v,v^L$ 为自车、前车速度，$a$ 为自车执行器内部加速度状态，$u$ 为期望加速度命令。忽略离散饱和时，运动学为

$$
\dot g=v^L-v,\quad \dot v=a,\quad
\dot a=\frac{u-a}{\tau(v,u,a)},\quad \dot v^L=a^L.
$$

前车加速度 $a^L$ 是外生量。网络不能访问未来脚本、场景标签或仿真隐藏条件；有限历史不能保证恢复前车的完整隐状态。因此下面的历史状态是一个有限记忆近似，不声称严格满足 Markov 性。

策略输入保留六帧历史和上一次实际应用命令：

$$
H_t=(s_{t-5},\ldots,s_t),\qquad x_t=(H_t,u_{t-1}).
$$

启动时以首帧重复填充。策略观测是固定物理尺度编码的 25 维向量：

$$
o_t=\left[\operatorname{vec}(H_t\oslash[50,20,20,6]),\;u_{t-1}/6\right].
$$

这里逐帧除以同一尺度；不会将 OOD 观测裁剪回训练范围。$a$ 在正式 PPO 实验中是仿真特权状态，真实小车必须另外估计，不能把仿真状态输入当作已经完成的感知链路。

策略 $\pi_\phi(j\mid o_t)$ 输出五分类动作，而不是航向角或舵机 PWM：

$$
\Delta u_j\in\{-0.8,-0.4,0,0.4,0.8\},\quad
u_t^{\rm req}=\operatorname{clip}(u_{t-1}+\Delta u_j,-6,2),
\quad u_t=\mathcal S(s_t,u_t^{\rm req}).
$$

$\mathcal S$ 是所有方法共用的独立安全过滤器，必要时覆盖为紧急制动。策略概率比针对离散动作 $j$ 计算，环境转移和奖励使用过滤后的 $u_t$。这不是形式化安全保证，也不是从摄像头到转向、油门的完整端到端驾驶。

## 2. 从物理方程到可训练世界模型

### 2.1 物理执行器的辨识

采用加速/制动不对称且随车速变化的一阶响应：

$$
\tau_\theta(v,u,a)=
\begin{cases}\tau_+,&u\geq a,\\\tau_-,&u<a\end{cases}
(1+\kappa v),\qquad \theta=(\tau_+,\tau_-,\kappa).
$$

参数仅从 train 的一步转移辨识：

$$
\theta^*=\arg\min_\theta\sum_{i\in D_{\rm train}}
\left[(a'_i-a_i)-\frac{\Delta t(u_i-a_i)}{\tau_\theta(v_i,u_i,a_i)}\right]^2,
$$

约束 $\tau_\pm\in[0.15,2]$ 秒、$\kappa\in[0,0.2]$ 秒/米。代码中的该拟合区间保证 $\tau>\Delta t=0.1$ 秒，因此拟合式与下式的分母保护一致：

$$
a^{\rm phys}_{t+1}=a_t+
\frac{\Delta t(u_t-a_t)}{\max(\Delta t,\tau_{\theta^*}(v_t,u_t,a_t))}.
$$

当前 ID 合成数据辨识约得 $\tau_+=0.45$ 秒、$\tau_-=0.75$ 秒、$\kappa=0.045$ 秒/米。这是合成环境参数，不是 R550 实测标定。

### 2.2 学习自车残差与前车变化

自车头使用 $z_t=[v_t,a_t,u_t,u_t-a_t]$，前车头使用

$$
q_t=[v^L_t,\;v^L_{t-4}-v^L_{t-5},\ldots,v^L_t-v^L_{t-1}].
$$

$q_t$ 恰为 1 个当前速度加 5 个相邻速度差，共 6 维。前车头排除自车状态和动作，避免将自车控制误当作外生前车变化的因果输入。两个头分别为 4→32→1、6→32→1 的 tanh MLP，通用形式为

$$
N_\omega(z)=\mu_y+\sigma_y\odot
\left[W_2\tanh\left(W_1((z-\mu_z)\oslash\sigma_z)+b_1\right)+b_2\right].
$$

输入与目标均值、尺度只用 train 计算；标准差下限分别为 0.05 和 0.001。采用列向量公式，NumPy 实现使用等价的行向量矩阵乘法。

主模型 `history` 是物理加残差自车头 $R_\omega$ 和历史前车头 $L_\eta$ 的组合：

$$
\widehat a_{t+1}=\operatorname{clip}(a^{\rm phys}_{t+1}+R_\omega(z_t),-6,2),\qquad
\widehat{\Delta v^L_t}=\operatorname{clip}(L_\eta(q_t),-0.6,0.2).
$$

前车头预测的是每个 0.1 秒步长的**速度增量**，单位为 m/s；不是加速度，更新时不能再乘一次 $\Delta t$。

### 2.3 完整的离散转移

对连续运动学采用显式 Euler，并处理速度非负约束：

$$
\widehat f_{\theta,\omega,\eta}(s_t,u_t,H_t)=
\begin{bmatrix}
g_t+\Delta t(v^L_t-v_t)\\
\max(0,v_t+\Delta t a_t)\\
\max(0,v^L_t+\widehat{\Delta v^L_t})\\
\widehat a_{t+1}
\end{bmatrix}.
$$

注意第二行使用旧 $a_t$，不能偷偷改成 $\widehat a_{t+1}$。递归时把预测状态移入历史，并将实际应用的 $u_t$ 作为下一步 previous command；不使用未来真实状态来校正分支。

### 2.4 拟合目标与可识别性

两个头的监督目标分别为

$$
y^a_i=a'_i-a^{\rm phys}_i,\qquad y^L_i=v^{L\prime}_i-v^L_i.
$$

各头独立最小化标准化目标的 MSE 加权重正则：

$$
\mathcal L(\omega)=\frac1N\sum_i
\left(\widetilde N_\omega(z_i)-\frac{y_i-\mu_y}{\sigma_y}\right)^2
+\frac{10^{-5}}2(\|W_1\|_F^2+\|W_2\|_F^2).
$$

这里 $\widetilde N$ 是反标准化前输出。Adam 学习率 0.002、batch 256、standard 最多 140 epochs，每 5 epochs 检查 val，连续 5 次无改善早停；val 选择权重，test 不参与。自车残差目标在数值精度内恒零时，输出层置零，避免人为制造学习增益。

MSE 在无限函数容量下对应监督数据条件均值；有限确定性网络只给一个点预测，不建模完整转移分布。随机急刹的不可预知性、分布外变化、历史别名问题仍存在。当前 ID 物理自车机制已被准确辨识，实测残差约为数值零，所以不能声称网络发现了额外自车动力学。

## 3. 世界模型如何进入 PPO

### 3.1 同一任务奖励

令下标 $+$ 表示转移后状态，定义误差

$$
e_g=\operatorname{clip}(g_+-(4+1.2v_+),-50,50),\quad
e_v=\operatorname{clip}(v_+-12,-20,20),\quad
e_b=\operatorname{clip}(\max(0,2+0.5v_+-g_+),0,20).
$$

公共任务代价和原始奖励为

$$
c=0.01e_g^2+0.1e_v^2+0.01u_t^2+0.1(u_t-u_{t-1})^2
+100\mathbf1[g_+\leq0]+10e_b^2,
$$

$$
r=\begin{cases}-100,&g_+\leq0,\\1-\min(2,c/10),&g_+>0.\end{cases}
$$

训练统一用 $\bar r=0.01r$；评估始终用原始 $r$。模型只学转移，奖励按已知同一公式计算，没有单独训练奖励网络。

### 3.2 冻结行为策略和短分支

每轮先用 $\pi_{\phi_{\rm old}}$ 收集 2048 条真实环境在线转移。这里“真实”指独立真值模拟器，**不指真实车辆**。在任何梯度更新前，复制这轮完整观测根池，并执行：

1. 从根池随机抽根 $x_0$，使用同一个冻结的 $\pi_{\phi_{\rm old}}$ 采样分支动作。
2. 用 $\widehat f$、公共过滤器和公共奖励生成 $h=1$ 步分支；消融为 $h=3$。碰撞立即停止。
3. 保存该冻结策略下的动作 log probability、value；分别计算每条分支的 GAE 和 return。
4. 主方法随机替换学习 buffer 中 1024 条样本，buffer 总数仍为 2048；真实轨迹 GAE 已独立算好。
5. 原样调用 SB3 `PPO.train()`，10 epochs、batch 64，每轮实际 320 次 Adam 更新；不对拼接后的 buffer 重算 GAE。

短分支不是旧策略 replay。行为 log probability 匹配只保证分支内动作的概率比一致，不能消除模型转移偏差，也不能使模型状态分布自动变成真实占用分布。短分支还改变了 GAE 时域和根的采样权重，实验收益不能全部归因于预测更准。

### 3.3 GAE、终止与截断

令 $d_i$ 表示碰撞等真正终止，$b_i$ 表示优势递推是否跨出当前轨迹片段，则

$$
\delta_i=\bar r_i+\gamma(1-d_i)V_{\psi_{\rm old}}(o_{i+1})-V_{\psi_{\rm old}}(o_i),
$$

$$
\widehat A_i=\delta_i+\gamma\lambda(1-b_i)\widehat A_{i+1},\qquad
\widehat R_i=\widehat A_i+V_{\psi_{\rm old}}(o_i).
$$

碰撞时 $d_i=b_i=1$；120 步时间上限或分支末端允许最终状态 value bootstrap，但优势链不跨 reset/分支边界。有限分支的末端递推取 $\widehat A_{i+1}=0$。真实和合成轨迹分别处理，不能把两个随机根拼成一条连续轨迹。

主 PPO 使用 $\gamma=0.99,\lambda=0.95$。策略比率与 clipped surrogate 为

$$
\rho_i(\phi)=\exp\{\log\pi_\phi(j_i\mid o_i)-\log\pi_{\phi_{\rm old}}(j_i\mid o_i)\},
$$

$$
L^{\rm clip}=\mathbb E_{i\sim D_{\rm mix}}
\left[\min\big(\rho_i\widehat A_i,
\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\widehat A_i\big)\right],
\quad\epsilon=0.2.
$$

实现沿用 SB3 的 minibatch 优势标准化与损失：

$$
\min_{\phi,\psi}\; -L^{\rm clip}
+0.5\,\mathbb E[(V_\psi-\widehat R)^2]-c_H\mathbb E[\mathcal H(\pi_\phi)],\qquad c_H=0.
$$

这里为清晰起见 $\widehat A$ 在 surrogate 中表示对应 minibatch 标准化后的优势，return 保持原目标；主方法 $D_{\rm mix}$ 是等量真实样本和合成样本的经验混合，而不是不变的真实 PPO 目标。Actor、critic 为独立的 25→64→64 tanh MLP，输出分别为五分类分布与标量 value。

### 3.4 一步分支到底提供了什么

由于显式 Euler 的 $g_+$、$v_+$ 由当前状态决定，同根、同动作、同过滤器下，不同模型产生的即时奖励完全相同。因此 $h=1$ 的模型间 TD 差异为

$$
\widehat\delta^{(A)}-\widehat\delta^{(B)}
=\gamma(1-d)\left[V_{\rm old}(\widehat o_+^{(A)})-V_{\rm old}(\widehat o_+^{(B)})\right].
$$

学习前车/执行器分量通过下一观测的 critic bootstrap 改变优势目标。三步分支还会改变后续间距、速度和奖励。当前诊断验证了一步奖励相等，但没有证明某个 critic 外推一定更准确；见 [模型诊断](SB3_MODEL_DIAGNOSIS.md)。

## 4. 为什么有潜在帮助，又为什么不保证改善

在扩展状态 $x=(H,u_{\rm prev})$ 上，固定一个非终止、确定性转移和动作，并假设 $V$ 在所选归一化度量下为 $L_V$-Lipschitz，则由三角不等式：

$$
\left|\widehat{\mathcal T}^{\pi}V(x)-\mathcal T^{\pi}V(x)\right|
\leq\mathbb E_{j\sim\pi}\left[|\widehat r-r|+
\gamma L_V\|\widehat F(x,j)-F(x,j)\|\right].
$$

此处 $r$ 使用与 value 一致的训练尺度；$F$ 包含过滤器、历史移位和 previous command 更新。该式只在假设成立、且终止处理一致的区域适用；碰撞边界的奖励不连续，不能直接套用全局平滑界。若状态是随机分布，应改用相应概率度量，不能把确定性点预测误差直接当作分布距离。

若在共同有效区域中每步模型误差不超过 $\varepsilon$、真实扩展转移对状态为 $L_F$-Lipschitz，且沿相同离散动作序列比较，则

$$
e_{k+1}\leq L_F e_k+\varepsilon,\quad e_0=0
\;\Longrightarrow\;
e_k\leq\varepsilon\sum_{j=0}^{k-1}L_F^j.
$$

实际闭环动作会随预测观测而变，上式不是整个策略闭环的无条件界。它解释了为什么从真实根启动、使用短分支可以限制开环误差传播，却不证明策略一定改善。分支短会更依赖 critic bootstrap；分支长会累积模型偏差。当前实验中三步分支总体均值也低于一步主方法。

该方案的潜在收益是：不新增真值环境交互便能生成当前策略下的局部反事实训练目标；物理先验减少需学习的未知量；训练后可只运行 actor。实际是否提高样本效率，必须包含离线采集、模型训练、生成时间及额外优化预算进行比较，不能只比较模型样本数。

## 5. 实现索引与参考

| 内容 | 实现 |
|---|---|
| 物理辨识、标准化 MLP、残差/历史模型 | `falconworld/models.py` |
| 观测、动作过滤、奖励和分支步进 | `falconworld/rl_env.py` |
| SB3 环境适配与训练奖励缩放 | `falconworld/rl_gym.py` |
| 冻结行为策略、独立分支 GAE、固定 buffer 替换 | `falconworld/sb3_worldmodel_ppo.py` |
| GAE 终止/截断逻辑 | `falconworld/ppo_agent.py` |
| 预算、数据隔离和最终 checkpoint 评价 | `run_sb3_rl.py` |
| 原始日志和实际 Adam 计数的独立审计 | `scripts/audit_sb3_rl.py` |

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017：PPO clipped objective 的来源。
- Janner et al., [When to Trust Your Model: Model-Based Policy Optimization](https://jannerm.github.io/mbpo-www/), NeurIPS 2019：真实根短模型分支的相关研究；本项目使用 PPO 和固定学习 buffer 替换，不复现该论文的完整算法或定理。
- [Stable-Baselines3 PPO 文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)：库接口参考；可复现版本以本仓库依赖锁和记录为准。
