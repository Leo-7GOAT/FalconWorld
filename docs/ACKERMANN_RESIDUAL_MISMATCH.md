# 残差补偿未建模失配：已验证的条件性结论

2026-11-06 已完成新的合成失配实验，模型先冻结再做预测测试与PPO闭环测试。**当前支持“学习残差能补偿固定物理模型族未覆盖的转向失配”，不支持“学习残差普遍优于知道正确机制的物理模型”。**

## 失配与公平对照

在原阿克曼仿真中加入转向标定增益、偏置与速度相关的单调平滑死区。该现象类型有工程依据，但具体函数和数值是仿真假设，未使用R550实测参数。

```math
e=1.08u_\delta+0.025-\delta,\qquad w=0.032\left[1+0.30(v/5)^2\right].
```

```math
F(e)=e-w\tanh(e/w),\qquad\delta_{t+1}=\mathrm{clip}\left(\delta_t+\frac{0.1F(e)}{(0.22+0.06v_t)c},-0.5,0.5\right).
```

F(0)=0，F与e同号，且导数非负。c=1为ID，OOD响应时间乘1.3；该隐藏条件不进入策略或模型输入。首个非单调平滑近似已归档排除，修正后训练、验证和测试全部换用新种子空间，没有沿用旧test来选配置。

每个训练种子共同使用2400 train＋600 val转移，并允许物理模型重新辨识：

|对照|有效结构|用途|
|---|---|---|
|rich|原丰富物理模型，7参数重拟合|检验原模型族的结构失配|
|affine|物理模型加入转向增益/偏置校准，有效9参数|学习残差的同底座对照|
|residual|同一affine模型＋3→64→64→1 tanh转角残差|不手工写入死区公式，从数据学习修正|
|mechanism|知道正确死区结构，有效12参数，参数仍从同一train拟合|匹配结构的强物理对照|

网络输入为速度、指令与当前转角的差、转向指令；尺度只用train，权重按val早停。测试episode不参与拟合或选择。所有PPO辅助组使用相同16步模型梯度、256次actor预热、16384在线步；环境奖励、动作约束和最终策略规则一致。

## 预测层面：相对同底座affine物理模型

三个训练种子，每种分布30个完整新测试episode，递归窗口先按episode汇总。

|ID指标|affine|residual|变化|
|---|---:|---:|---:|
|0.1秒转角MAE rad|0.00392286|0.000053117|降低98.6%|
|3秒转角MAE rad|0.00789379|0.000126330|降低98.4%|
|3秒位置误差m|0.17207762|0.00389155|降低97.7%|

配对差值residual−affine的95%区间：单步转角误差 **[−0.004180,−0.003581] rad**；3秒位置误差 **[−0.24055,−0.10718] m**。三训练种子方向一致。

这说明残差确实在独立数据上修正了该物理模型族遗漏的非线性，收益不是仅在train上拟合得更好。

## PPO闭环：有一项主任务收益，未全面占优

主实验3 seeds × 5方法，共600次冻结策略闭环评价，方法间使用配对场景。

|ID指标|同底座affine辅助PPO|residual辅助PPO|
|---|---:|---:|
|完成回合|60/60|60/60|
|平均原始回报|155.909|159.248|
|横向RMSE m|0.19435|0.17811|
|速度RMSE m/s|0.22778|0.26630|

预登记主指标之一的ID原始回报差为 **+3.3389，95%区间[0.5816,6.1799]**；三个训练种子分别为+6.3601、+0.5962、+3.0605。完成率没有降低。

但横向RMSE差区间 **[−0.08950,0.05592] m** 跨零，速度误差反而较大，不能写成所有控制指标都提升。相比另一种rich物理底座，残差也没有形成全面闭环优势。完整逐种子指标与其他对照都在[实验报告](../results/ackermann_mismatch_v2/REPORT.md)。

## 边界必须保留

- **准确物理模型仍能更好。** mechanism对照ID单步转角误差约3.78e−7 rad，3秒位置误差约2.00e−5 m，均低于残差模型。残差提供的是不手工给出该机制时的数据驱动补偿能力。
- **OOD未形成统一收益。** 执行器时间常数改变后，3秒位置误差affine为0.3562m、residual为0.3671m，残差较差；PPO OOD回报也没有稳定额外收益。它不具备自动识别任意动态漂移的能力。
- **少量在线步诊断不替代主结论。** 主结果后提出的探索性0/2048在线步诊断使用全新380M场景，两个预算全部保留，结果存在指标和种子差异，不能选择较好预算宣称总体胜出。
- **有额外成本。** 残差网络拟合、验证早停及模型梯度计算均有成本。各辅助组虽然相同在线数据和actor更新预算，但不是相同墙钟开销。
- **仍是合成仿真。** 不是实车失配辨识，没有证明车辆部署、实时性或真实道路优势。三训练种子的bootstrap区间仅适用于该实验；多重比较未校正。

## 审计和复现

141项完整测试及31个子检查通过。独立审计重建了主实验600回合/50931次转移，以及960条预测episode-模型-时域汇总，均零错误。预测审计使用独立NumPy前向实现，不调用被测模型；辅助0/2048步诊断各480回合也审计通过。评估是配对运行，不能把同一场景的多个控制器运行当成新增独立场景。

- [完整主报告与区间](../results/ackermann_mismatch_v2/REPORT.md)
- [预测独立审计](../results/ackermann_mismatch_v2/prediction_audit.json)
- [闭环独立审计](../results/ackermann_mismatch_v2/audit.json)
- 补充诊断：`results/ackermann_mismatch_warm/budget0`与`budget2048`。

```powershell
.\.venv\Scripts\python.exe run_ackermann_mismatch.py --stage prepare --output outputs/ackermann_rl/new_mismatch
.\.venv\Scripts\python.exe run_ackermann_mismatch.py --stage evaluate --output outputs/ackermann_rl/new_mismatch --workers 5
.\.venv\Scripts\python.exe scripts/audit_ackermann_mismatch_prediction.py --run outputs/ackermann_rl/new_mismatch
.\.venv\Scripts\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_mismatch
.\.venv\Scripts\python.exe scripts/report_ackermann_mismatch.py --run outputs/ackermann_rl/new_mismatch
```

输出目录必须不存在。物理现象类型参考官方工程文档：[Dead Zone](https://www.mathworks.com/help/simulink/slref/deadzone.html)、[Coulomb and Viscous Friction](https://www.mathworks.com/help/simulink/slref/coulombandviscousfriction.html)；本文平滑函数与参数是另行明确登记的合成设定。
