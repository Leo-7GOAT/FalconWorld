# FalconWorld

当前研究目标已切换为**世界模型辅助 PPO 强化学习**，车辆部署与Pi实时验收留到上车阶段。新入口见 [README_RL.md](README_RL.md)，目标见 [RL_SPEC.md](RL_SPEC.md)，进度见 [PROGRESS.md](PROGRESS.md)。下文保留已完成的世界模型＋MPC研究与复现说明，其结果不等于PPO收益证据。

用于规划控制研究的紧凑动作条件世界模型：观测与历史 → 候选动作递归预测 → CEM-MPC → 执行第一步 → 新观测。系统采用基于模型的控制，没有策略或价值学习，不称为强化学习。

完整要求见 [PROJECT_SPEC.md](PROJECT_SPEC.md)，当前验收见 [PROGRESS.md](PROGRESS.md)，实验登记见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。Lesson01/02 保留数值兼容，已不需要用户逐课填写函数。

## 一键复现

实际验证环境：Windows、Python 3.11.9、Ryzen 5 9600X。紧凑模型使用 NumPy CPU，不要求 CUDA。依赖范围见 requirements.txt，已安装版本见 requirements-host-lock.txt。

在项目根目录 PowerShell 执行：

```powershell
.\reproduce.ps1 -Preset smoke -Workers 2
.\.venv\Scripts\python.exe reproduce.py --preset standard --workers 4
```

入口创建或使用 .venv，依次执行测试、纵向训练/开环/闭环、二维实验、全模型 ONNX 一致性、区间/观测/动作分支诊断。standard 含三训练种子及大量完整 MPC 周期，需长时间运行；smoke 仅验证流程，不支持统计优势结论。full 进一步增加训练和回合预算。

数据、代码、模型与配置指纹一致时恢复；指纹变化请指定新输出目录，保留旧证据。outputs 中带 trial 的目录与旧二维 standard 是历史试跑，不与本轮正式结果混合。

## 分步命令

```powershell
python run_experiments.py --preset smoke --output outputs/smoke_verified --workers 2
python run_experiments.py --preset standard --output outputs/standard_verified --workers 4
python run_experiments.py --stage summarize --output outputs/standard_verified
python scripts/run_sensor_closed.py
python -m pytest -q
python check_lesson01.py
python check_lesson02.py
```

输出包括完整 episode 划分清单、模型/训练日志、源快照与摘要、逐周期轨迹 CSV、逐回合 JSONL、开环误差、配对效应区间、耗时分位数和 completeness.json。完整性未通过的结果是 provisional，不能当作完整标准验收。

## 方法与边界

纵向状态为 [gap m, ego m/s, leader m/s, actuator m/s²]，dt=0.1s；最后一项是仿真特权状态。传感实验删除该项，用过去速度差作因果估计，停车处存在不可观测性。训练/验证/测试按完整 episode 分割；物理参数和尺度只用 train，早停与计算预算只用 val。

比较 ACC、辨识物理 CV/CA、丰富物理、确定性 MLP、残差、历史、物理自车加学习前车、集成与风险消融。所有 MPC 共用任务代价、时域与约束；另测验证阶段近似时间匹配预算。报告实际完整周期及超时回退；共享主机负载下计时不代表 Pi 性能。

丰富物理模型可精确辨识 ID 自车机制，残差接近零是应保留的负结果。外生前车的无先兆随机急刹不能可靠提前预知。集成分歧不等于校准置信度。预测改进能否产生驾驶收益须查看闭环结果。

二维模拟覆盖直线、曲线、变速与障碍物，尺度和响应均为仿真假设，不能当成 R550 已标定参数。

- [纵向模型和数据](docs/MODELS.md)
- [二维阿克曼](docs/ACKERMANN.md)
- [CARLA 同步闭环和实际命令](docs/CARLA.md)
- [R550 导出、回放、影子与设备验收](docs/R550.md)
- [设计决定](docs/DECISIONS.md)
- [中文实验报告](docs/REPORT.md)
- [English overview](docs/PROJECT_EN.md)

CARLA 使用本机 C:\path\to\CARLA_0.9.15\WindowsNoEditor 的匹配客户端。R550 的 ROS2/ARM64 验收需设备可达和配置核实；本机导出/回放不能称为车端部署。真实车辆运动仍需明确授权与停车、接管、超时能力核验。
