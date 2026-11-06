# 改进：多步世界模型梯度辅助PPO

2026-11-06：已实现、实际训练，并在独立新测试中观察到改进。旧一步替换方法的失败结果保留，不与本轮合并。

## 主要证据

**收益最明确的是有限交互预算下的学习效率。** 先用seed0/1的独立验证场景，在两个预登记候选中选择配置；随后换新训练seed6/7/8和未查看的测试场景，固定最终策略评价：

|方法|在线步/任务|离线交互/任务|原生总交互|Adam调用/任务|完成/120|
|---|---:|---:|---:|---:|---:|
|纯PPO|16384|0|16384|2560|21|
|更多在线预算纯PPO|20480|0|20480|3200|39|
|模型梯度辅助PPO|16384|3000|19384|2816|120|
|丰富物理模型辅助|16384|3000|19384|2816|120|

模型方案并非免费：使用额外离线数据、256次actor预热、约42.6万步可微模型转移，以及额外计算。总交互控制给纯PPO更多真值交互和更多Adam调用，仍在本轮短预算新测试中落后。

短预算相对纯PPO原始回报差：ID **+153.95，95%区间[145.18,162.86]**；OOD **+113.98，[104.72,122.91]**。相对更多在线预算纯PPO，分别为 **+141.36，[113.76,159.17]**、**+103.61，[73.21,123.46]**。

另一个独立长预算实验使用新seed3/4/5，每任务65536在线步：

|指标|纯PPO|模型梯度辅助PPO|
|---|---:|---:|
|ID完成/60|57|59|
|OOD完成/60|60|60|
|ID横向RMSE m|0.10025|0.07529|
|OOD横向RMSE m|0.13231|0.08683|
|ID原始回报|160.334|162.255|
|OOD原始回报|123.307|124.199|
|ID转角变化率RMS rad/s|0.78834|0.57019|
|OOD转角变化率RMS rad/s|0.77423|0.55599|

长预算下，OOD回报差 **+0.892，[0.167,1.923]**，OOD横向RMSE差 **−0.04548m，[−0.05816,−0.03289]**。ID回报差区间 **[−1.418,6.076]** 跨零，不能声称所有条件稳定胜出。长预算仅平滑动作控制、匹配额外Adam调用的PPO控制以及丰富物理模型控制全部保留在报告中。

每个实验仅三个训练种子；分层配对bootstrap先重采样训练seed，再在seed内按直线/曲线成对重采样场景，2000次。多重比较未校正，有限仿真场景上的区间不提供跨任务或实车保证。丰富物理模型也获得类似收益，因此**目前证明的是模型辅助训练机制有价值，尚未证明学习残差优于可辨识的丰富物理模型**。

## 改了什么

旧方法在更新前用一步TD样本替换半数真实GAE，可能损失转向延迟所需的长期信用分配。新方法保留全部真实PPO rollout、GAE和return，以冻结的世界模型辅助actor：

1. 将拟合物理＋岭回归残差模型转换为数值一致的Torch实现，模型参数固定为buffer。
2. 用已知参考路线和物理范围采样2048个状态根，不读取测试场景或真实未来；对actor做256次模型辅助预热，critic和动作方差不预热。
3. 每4次PPO梯度更新，从当前真实小批量中取16个状态根，递归使用actor均值和模型预测16步。
4. 对预测状态的横向、航向、速度误差及动作变化代价反向传播，只更新策略；附加损失与PPO损失相加后，共用一次backward、梯度裁剪和Adam step。
5. 在线执行只运行actor，仍输出加速度和前轮转角，不需要世界模型或MPC。

损失关系为：

```math
L_{\mathrm{total}}=L_{\mathrm{PPO}}+\beta L_{\mathrm{model}},\qquad\beta=1.
```

```math
L_{\mathrm{model}}=\frac{1}{BH}\sum_{i=1}^{B}\sum_{k=1}^{H}c(\widehat s_{i,k},\mu_\phi(\widehat o_{i,k})),\qquad H=16.
```

预测代价包含跟踪与动作变化约束，未使用环境奖励的代价饱和，且平滑权重更强；actor均值滚动也不是随机策略期望的精确积分。它是显式改变训练目标的模型辅助PPO变体，不继承原版PPO无偏性或性能改善保证。损失通过执行器延迟和后续状态传播梯度，不等于旧方法的一步critic bootstrap。

`ppo_compute`匹配长预算下额外256次Adam调用；`smooth_ppo`采用相同预热和辅助更新安排，但只约束动作变化，不调用模型；短预算`ppo_total_budget`给纯PPO20480在线步，超过模型方案19384原生总交互。前两项不能完全匹配模型反向传播的墙钟，因此计算开销仍须单列。

## 成本与可复现材料

长预算模型预热＋训练平均约125.96秒，纯PPO约48.18秒，约2.61倍；短预算分别约38.33秒和12.59秒，约3.04倍。模型池拟合和候选验证成本另计。共享CPU主机计时，不代表Pi时延。

- [长预算完整报告](../results/ackermann_gradient_final/REPORT.md)：全部seed、配对区间、控制与成本。
- [短预算完整报告](../results/ackermann_gradient_efficiency/REPORT.md)：样本效率与总交互控制。
- [冻结验证选择](../results/ackermann_gradient_selection/selection.json)：候选a/b的全部验证结果。
- [长预算审计](../results/ackermann_gradient_final/audit.json)、[短预算审计](../results/ackermann_gradient_efficiency/audit.json)：各600回合，独立重建真值动力学、扫掠几何、奖励、终止、预算和实际Adam状态，均零错误。

新PPO方法保留SB3原优化流程，关闭辅助项时已验证与官方PPO权重逐位一致；可微模型与NumPy预测一致、梯度有效、模型权重冻结和官方PPO.load均有测试。138项完整测试和31个子检查通过。辅助样本没有完整逐步训练日志，审计不是整个训练过程的形式化证明。

```powershell
.\.venv\Scripts\python.exe run_ackermann_improvement.py --stage select --output outputs/ackermann_rl/new_selection --workers 4
.\.venv\Scripts\python.exe run_ackermann_improvement.py --stage final --selection outputs/ackermann_rl/new_selection/selection.json --output outputs/ackermann_rl/new_final --workers 5
.\.venv\Scripts\python.exe run_ackermann_improvement.py --stage efficiency --selection outputs/ackermann_rl/new_selection/selection.json --output outputs/ackermann_rl/new_efficiency --workers 5
.\.venv\Scripts\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_final
.\.venv\Scripts\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_efficiency
.\.venv\Scripts\python.exe scripts/report_ackermann_improvement.py --run outputs/ackermann_rl/new_final
.\.venv\Scripts\python.exe scripts/report_ackermann_improvement.py --run outputs/ackermann_rl/new_efficiency
```

输出目录必须不存在。原基线、原奖励和旧失败记录均保留，未按新测试调参或选checkpoint。环境仍需要参考路径和仿真执行器状态，无视觉、障碍路径规划、真实车辆部署。
