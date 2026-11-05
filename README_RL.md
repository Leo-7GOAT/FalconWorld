# FalconWorld：世界模型辅助 PPO

当前正式研究采用 Stable-Baselines3 2.9.2 的PPO默认优化器与训练配置作为强基线。旧自研PPO首轮完整保留，不能拿其弱策略来证明世界模型有效。部署、设备连接与Pi实时性留到上车阶段。

目标：RL_SPEC.md；预登记：docs/SB3_RL_PROTOCOL.md；进度：PROGRESS.md；主报告：docs/SB3_RL_REPORT.md。标准21个训练任务、1680回合已齐全并通过独立审计。

## 复现入口

验证环境为Windows、Python 3.11.9、CPU PyTorch。先在工程根目录PowerShell准备并核对依赖（已实际执行）：

```powershell
.\setup_rl.ps1
.\setup_rl.ps1 -VerifyOnly
```

使用该.venv执行一键复现（smoke与standard均已实际退出0）：

```powershell
.\.venv\Scripts\python.exe reproduce_sb3_rl.py --preset smoke --workers 3
.\.venv\Scripts\python.exe reproduce_sb3_rl.py --preset standard --workers 3
```

一键执行全测试、训练/冻结评估、独立原始日志审计和报告生成。依赖闭包36项固定版本见requirements-rl-lock.txt；Torch CPU轮子使用官方CPU源，其他依赖来自PyPI。最终全套测试130项及31个子检查通过，完成性审查见docs/RL_COMPLETION_AUDIT.md。

分步：

```powershell
.\.venv\Scripts\python.exe run_sb3_rl.py --preset standard --workers 3
.\.venv\Scripts\python.exe scripts/audit_sb3_rl.py --run outputs/rl_sb3/standard
.\.venv\Scripts\python.exe scripts/report_sb3_rl.py --input outputs/rl_sb3/standard
```

加载已训练模型的独立评价入口（已实际执行，仅文件与仿真）：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sb3_policy.py --checkpoint outputs/rl_sb3/standard/seed0/wm_ppo/final_policy.zip --output outputs/rl_sb3/loaded_policy_acceptance --split val
```

此命令的输出目录必须是新目录；重复运行时换一个名字，以保留先前证据。策略zip可直接用官方PPO.load加载，不需要在线世界模型。

补充较充分预算参考已实际运行并审计（204800步×3seed，不是等预算模型优势对照）：

```powershell
.\.venv\Scripts\python.exe run_sb3_reference.py
.\.venv\Scripts\python.exe scripts/audit_sb3_reference.py --run outputs/rl_sb3/reference_204800
```

研究结论：当前数据中存在有限条件收益，但相对纯PPO的主回报区间跨零，未建立稳定总体优势；ID平顺性、个别seed退化和额外成本均保留。较充分纯PPO参考仍存在种子不稳定，不意味着已达到实车可靠性门槛。

标准：3训练seed、7方法，每主方法65536新增仿真步，每族每分布30独立配对test episode；总1680回合。PPO默认2048 rollout、64 batch、10 epochs。世界模型在更新前用同behavior策略生成短轨迹，替换半数训练buffer，不增加主比较的优化器更新。物理模型对照使用相同规则。

另外包含无历史/三步rollout消融、20epoch额外更新控制、把offline采集预算换成额外online步的纯PPO总交互控制；最后一段partial rollout精确补齐，实际更新/数据/生成成本单列。模型仅使用共同train/val拟合；offline数据不混作PPO on-policy经验。

source/data/model指纹一致时只复用已完成的任务，中断任务归档后重训，不声称partial精确恢复。最终策略按固定预算评估，val不用于挑test-best。原始指标/逐回合轨迹/独立审计/学习曲线/成本表都在outputs/rl_sb3。

## 保留的前期研究

- 旧世界模型+MPC：docs/REPORT.md，复现说明见主README原章节。
- 自研PPO32k首轮：docs/RL_REPORT.md；outputs/rl_ppo/standard。
- train/val优化尺度诊断：docs/PPO_VALIDATION_DIAGNOSIS.md。
- 延长预算的val诊断：outputs/rl_ppo_budget_validation/seed0_131072/DIAGNOSIS.md。
- 官方默认PPO固定204800步val交叉验证：outputs/sb3_ppo_default_validation/seed0_204800；不属于test或世界模型收益证据。

无优势、预测偏差、保守性与算力代价都必须保留。首轮和新轮不可混合成单个总体效果；实车部署及Raspberry Pi实时性能均未在本轮执行。
