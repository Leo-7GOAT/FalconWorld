# 横纵向联合 PPO：首轮实现和复现

后续已完成[多步模型梯度改进](ACKERMANN_MODEL_GRADIENT.md)，在新的冻结测试中改善样本效率和部分控制指标。下文保留首轮一步替换方法的完整负结果。

2026-11-06 已实际实现、训练并评价。策略直接输出期望加速度与前轮转角，执行端不再用传统横向控制器替代 PPO。范围是提供参考路径后的本地仿真跟踪，不是视觉端到端驾驶、路径搜索或实车部署。

## 本轮结果

固定三训练种子、三方法，每任务65536在线步和10240次实际Adam更新；最终策略测试共288回合。每方法分布内/分布外各48回合：

|方法|ID完成|OOD完成|ID横向RMSE m|OOD横向RMSE m|
|---|---:|---:|---:|---:|
|纯PPO|48/48|48/48|0.100|0.139|
|学习世界模型辅助PPO|34/48|28/48|0.556|0.665|
|丰富物理模型辅助PPO|16/48|19/48|0.694|0.781|

**联合控制确实已训练；本轮世界模型辅助弱于纯PPO，没有建立横向收益。** 未按测试结果挑模型、改参数或隐藏失败种子。只有三个训练种子，纯PPO样本中零越界也不等于零风险。转角命令仍有明显振荡，变化率RMS接近0.8 rad/s限制；平顺性、感知输入、真实车参和部署均待后续工作。

完整[冻结报告、逐种子结果与轨迹图](../results/ackermann_rl/REPORT.md)、[独立审计](../results/ackermann_rl/audit.json)。首轮奖励失败留在本地`outputs/ackermann_rl/quick`，发布的历史记录位于`results/ackermann_rl_v1`。正式新轮为`outputs/ackermann_rl/tracking_v2_quick`，发布映射为`results/ackermann_rl`。

## 接口

- 14维输入：位姿、速度、执行器状态、上一实际指令、参考曲线参数/目标速度、横向与航向误差、3m前瞻偏差。执行器状态和已知路线为仿真输入，不代表已具备真实感知。
- 两维连续输出：归一化动作映射为加速度[-3,2] m/s²、前轮角[-0.5,0.5] rad；共同限制命令jerk 5 m/s³、转角变化率0.8 rad/s。
- 世界模型：复用二维物理加16项基函数岭回归残差；拟合加速度状态、转角、yaw的单步修正；2400 train＋600 val转移，测试数据不参与拟合。
- 辅助机制：冻结当前行为策略，用真实观测根批量生成一步模型分支，独立TD目标替换50%固定buffer；Gaussian原始动作和log probability保留，转移使用裁剪/限速动作；调用原SB3优化，不增加梯度更新。
- ID直线/曲线、目标速度2–4m/s；OOD目标速度4–5m/s、曲线振幅变大、响应时间乘1.3。扫掠车身越出7m道路即失败，达到30m目标成功，160步时间上限截断。无其他车辆或障碍物。

## 复现

先在根目录执行`setup_rl.ps1`准备固定CPU依赖，pytest另按主README安装。输出目录必须不存在：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ackermann_rl.py -q
.\.venv\Scripts\python.exe run_ackermann_rl.py --preset quick --workers 3 --output outputs/ackermann_rl/new_run
.\.venv\Scripts\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_run
.\.venv\Scripts\python.exe scripts/report_ackermann_rl.py --run outputs/ackermann_rl/new_run
```

快速流程检查将`quick`改为`smoke`，仅4096在线步×一个种子，不能用它证明控制质量。

加载最终策略（仿真，不访问车辆）：

```python
from stable_baselines3 import PPO
from falconworld.ackermann_rl import AckermannTrackingEnv

policy = PPO.load('results/ackermann_rl/seed0/ppo/final_policy.zip', device='cpu')
env = AckermannTrackingEnv()
observation, info = env.reset(seed=0)
while True:
    action, _ = policy.predict(observation, deterministic=True)
    observation, learning_reward, terminated, truncated, info = env.step(action)
    # info['applied']: [acceleration_mps2, front_wheel_angle_rad]
    if terminated or truncated:
        break
```

正式九个策略都已使用官方PPO.load重新加载并实际进行测试。保存格式用于冻结评价，不宣称精确续训。

## 验证

新增五项测试覆盖Gym接口、双控制量限速、不同路线下模型分支与独立真值一致性、终止/截断区别、无模型时与官方PPO权重逐位一致、固定更新预算及官方加载。完整135项测试和31个子检查通过。

独立审计不导入环境和策略实现，重建真值动力学、扫掠几何、奖励和横向指标，检查连续性、数据划分、SHA、checkpoint步数和Adam状态。288回合/21588转移零错误；没有逐训练样本执行轨迹，不能视为训练全过程的形式化证明。
