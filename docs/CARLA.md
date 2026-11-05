# CARLA 0.9.15：已实际运行的集成

2026-10-28（Asia/Shanghai）核验的安装为 `C:\path\to\CARLA_0.9.15\WindowsNoEditor`。主工程 Python3.11 无 CARLA，但安装自带 `.venv\Scripts\python.exe` 是 Python3.9.13，客户端0.9.15，匹配服务端0.9.15。现有服务进程命令包含 `-carla-rpc-port=2100`；默认2000端口10秒连接超时不代表安装不可用。成功发现 Town10HD_Opt、同步0.05秒、no_rendering、frame0、无车辆。

## 实际命令与产物

在 FalconWorld 根目录 PowerShell 执行过：

```powershell
& 'C:\path\to\CARLA_0.9.15\WindowsNoEditor\.venv\Scripts\python.exe' -u scripts\carla_benchmark.py --steps 240
```

输出 `outputs/carla/config.json`、四个数据episode CSV、`models.npz`、`selection.json`、四个闭环CSV、`metrics.json`。如果以后服务未启动，先检查进程命令和监听端口；安装的启动命令为：

```powershell
Start-Process -FilePath 'C:\path\to\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe' -ArgumentList '-quality-level=Low','-windowed','-ResX=640','-ResY=360','-carla-rpc-port=2100','-nosplash','-unattended','-RenderOffScreen' -WindowStyle Hidden
```

首次集成已完成全部四个闭环；随后补充完整周期计时时，原服务进程已退出，2100连接20秒超时。检查进程已不存在后实际执行上述启动命令，并成功重新完成整套实验。没有将启动失败作为最终阻塞。

适配器保存/恢复原世界设置，仅删除自身创建的车辆和碰撞传感器，不删除已有actors、不重载已有地图。运行时只能有一个同步tick客户端。当前基准不会运行交通管理器。

## 数据、模型和控制边界

训练episode种子101、102；验证201；冻结测试301。每个240步、dt=0.1秒。训练采集有界随机加减速，每15步换命令。归一化控制为 throttle=max(a,0)/3、brake=max(-a,0)/6，分别限幅[0,1]，是需由响应模型校准的前馈映射。它不是声称加速度命令已被真实车辆执行。输入只有速度、从历史速度差得到的加速度和候选动作，没有读取 CARLA physics 隐藏动力参数作为预测器输入。

物理强基线使用训练数据辨识的线性响应（上一加速度、油门、制动、速度及常数）；学习模型加入速度平方、控制平方、速度与控制乘积、低速饱和特征。两者各自在验证集选择ridge∈{0.001,0.1,1,10}，然后保存并重新加载权重。测试集不参与训练或选择。此处是紧凑多项式世界模型，不称为神经网络；主工程另有MLP与残差模型。

每次规划96候选×20步×3轮，12精英、同代价/限幅/变化率/种子，复用主工程 `falconworld.planning.CEMOptimizer`。候选递归使用自己的速度和加速度。只执行第一步，发送控制、同步tick、读取新观测后重规划。速度目标依次5→2→0→5m/s；路点pure-pursuit转向为共享模块，学习模型参与纵向速度候选评估。该实验没有前车/交互，也没有声称学习模型用于CARLA横向候选评估。

Observation 使用右手系：CARLA(x,y,yaw) → (x,-y,-yaw)，位置m、航向rad、时间为服务端 elapsed_seconds，保留frame；Action 左正前轮转角rad → CARLA右正steer。最大转角从车型轮子配置读取，仅用于转换限幅。同步0.1秒、substep0.01秒×10满足积分约束。运动学加速度=(v_next−v)/dt，jerk由该差分加速度计算，停车时没有借用执行器内部值。

`cycle_ms` 包含CEM全部候选展开、路点转向、动作发送、world.tick、新观测和车道距离查询；`inference_ms` 是全部模型递归部分，不是一次前向。超过规划0.1秒或非有限指令则制动回退，独立记录。主机完整周期耗时不代表Pi性能。

## 实测结果与局限

最终 `metrics.json`：3秒测试递归速度RMSE，校准物理1.4821m/s，学习1.1180m/s（下降24.6%）。0.5秒为0.8161→0.7307，1秒1.2853→1.1004。这里只有1个独立测试episode，窗口数不能当成独立episode统计。

闭环两配对种子401/402、每回合24秒：

|模型|速度RMSE均值 m/s|进度均值 m|碰撞/回合|完成率|周期p95范围 ms|超时/回退|
|---|---:|---:|---:|---:|---:|---:|
|校准物理|1.1450|84.5812|0/2|2/2|5.876–6.753|0|
|学习响应|1.1680|86.6559|0/2|2/2|6.048–6.544|0|

学习预测改善没有变成更好速度跟踪；平均RMSE反而增加约2.0%。进度略高与目标误差共同呈现，不能解释为驾驶更优。模型学习到更丰富的输入响应，但短采集、低速接触/启停和候选优化对模型误差的利用仍是边界；本轮没有为翻转负结果而重调测试集。

初版独立CEM实现的原始结果原样归档于 `outputs/carla_initial_custom_cem`。之后为统一工程主体，改用共享CEM（时间相关噪声、采样最优序列），实际重新跑完整数据、训练与全部四个闭环；最终结果只使用共享CEM这轮，不按胜负挑选版本。`config.json` 保存原世界设置、天气、车型质量/阻力/轮子转角/位置、车身半尺寸及spawn。`python scripts/carla_plot_results.py` 已实际执行，从最终CSV生成 `closed_loop_comparison.png` 和 `summary.json`。

这属于CARLA真实服务端集成实验，规模小于纵向standard；无天气/地图/邻车泛化声明。各episode新建相同型号、相同起始spawn的actor并固定CEM种子，但为保留用户世界未重载地图，未宣称CARLA位级确定性。严格物理重现需要专用服务端在每次重复前设置同步并重载地图，见[官方同步与确定性说明](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)。坐标与API语义见[官方Python API](https://carla.readthedocs.io/en/0.9.15/python_api/)。

## 2026-11-04 接续复核（以此轮目录为准）

本轮发现服务进程未运行后，实际启动已有 CARLA 0.9.15 的2100端口；使用同一Python3.9客户端，执行：

```powershell
& 'C:\path\to\CARLA_0.9.15\WindowsNoEditor\.venv\Scripts\python.exe' -u scripts/carla_benchmark.py --steps 240 --output outputs/carla_verified_20261104
.\.venv\Scripts\python.exe scripts/carla_plot_results.py --input outputs/carla_verified_20261104
```

四个采集episode、训练/验证选择/保存重载、四个24秒闭环均成功。此次修复最后合法预测窗口遗漏（3秒现在8个窗口）与末步碰撞的完成判定，禁止覆盖非空输出目录，配置记录相关源文件SHA。旧结果全部保留，不混用试跑与本轮结果。

本轮冻结测试的0.5/1/3秒速度RMSE分别为：校准物理0.7700/1.2209/1.5970m/s，学习0.6856/1.0399/0.9970m/s。3秒预测误差下降37.6%，仍只有1个测试episode。

两配对闭环：物理平均速度RMSE1.17775m/s、进度83.889m；学习1.18128m/s、87.422m。均0/2碰撞、2/2有限时域完成；全部960周期无超时/回退。完整周期p95物理11.447–11.454ms、学习11.823–12.565ms，为同时运行其他本地实验时的主机测量。预测改善仍未得到平均速度跟踪收益（误差增加0.30%），无学习驾驶优势结论。两种子差异方向并不一致，不能从这个样本规模推断统计显著性。服务端未重载地图，因此与10月轮数值不同并非可位级复现的声明。
