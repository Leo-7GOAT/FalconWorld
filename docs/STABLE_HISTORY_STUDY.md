# 一次基于验证失稳的修复研究

原正式模型、控制器、测试与结果完整保留；本研究只加载train/val。预算在运行前写入 registration.json：6个稳定AR2超参数与3个衰减CA对照参数，按val的episode等权0.5/1/3秒平均gap MAE选择，不再扩展结构搜索。

原失效：神经网络前车单步偏差被自身历史反馈复合，匀速场景产生非零漂移。修复假设：零截距令严格匀速成为固定点；AR2根投影到单位圆内限制速度增量的递归自激。自车仍使用train辨识强物理。稳定性仅涉及该线性速度增量递推，不是碰撞、位置误差或闭环安全保证；饱和、停车和未知转折仍有误差。

选中AR2系数 `[1.0589810675612303, -0.09372386369657686]`，选中衰减CA系数 `[1.0, 0.0]`。同一数据上的确定性拟合只有一次，不将复制到3个NN种子称作3个独立新拟合。平均val分数(m)：`{'physical_enriched': 0.39981683530717166, 'history': 0.8078500913573995, 'stable_history': 0.48932566157469454, 'adaptive_damped_ca': 0.39981683530717166}`，候选减衰减CA=+0.0895088m。

验证集未优于合理衰减CA对照，本候选停止，不值得仅为追求正结果追加测试。

本次固定6+3候选的拟合、验证、对照与profile合计1.270秒，来自selection.json；这是描述性均值差，没有统计区间，不称显著改善。64候选×30步×3迭代的主机完整周期中位数分别为强物理7.92ms、原history16.45ms、稳定AR2 10.01ms、衰减CA 10.42ms；同机争用条件下的一组实测，不能解释为跨硬件固定加速比。

保存加载与选择验证已实际执行：`python scripts/verify_stable_history.py`，仅读val，不重训或重搜。已保存候选与登记val argmin一致；重载val分数与原分数最大差5.55e-17m，候选AR根最大模0.961505，满足选中0.999上界。衰减CA最优gamma=1，因此它是边界恒加速度模型，并非严格衰减稳定模型；不能把其根模1也写成严格小于1。详细模型SHA256和检查记录见verification.json。

`python -m pytest tests/test_stable_history.py tests/test_validation_analysis.py tests/test_report.py -q` 实际7项通过，覆盖根投影、100步匀速不漂移、批量保存加载一致、解析递归误差及报告配对完整性。此候选不进入主benchmark，不追加测试，不替换正式模型。

### 每族三秒递归误差（全部对照保留）

|族|方法|gap MAE m|ego MAE m/s|lead MAE m/s|actuator MAE m/s²|
|---|---|---|---|---|---|
|brake|adaptive_damped_ca|1.12613|6.67413e-12|1.5331|4.88367e-12|
|brake|history|2.1727|1.23382e-06|1.89664|9.56688e-07|
|brake|learned_lead|2.1727|6.67413e-12|1.89664|4.88367e-12|
|brake|physical_ca|1.51211|0.35388|1.5331|0.211376|
|brake|physical_cv|4.85584|0.35388|3.01718|0.211376|
|brake|physical_enriched|1.12613|6.67413e-12|1.5331|4.88367e-12|
|brake|stable_history|1.53699|6.67413e-12|1.65165|4.88367e-12|
|constant|adaptive_damped_ca|1.07125e-11|6.87459e-12|0|5.71259e-12|
|constant|history|1.95641|1.2711e-06|1.64569|6.67488e-07|
|constant|learned_lead|1.95642|6.87459e-12|1.64569|5.71259e-12|
|constant|physical_ca|0.514153|0.465027|0|0.204659|
|constant|physical_cv|0.514153|0.465027|0|0.204659|
|constant|physical_enriched|1.07125e-11|6.87459e-12|0|5.71259e-12|
|constant|stable_history|1.07125e-11|6.87459e-12|0|5.71259e-12|
|mild|adaptive_damped_ca|1.97677|9.70154e-12|1.79491|5.35588e-12|
|mild|history|2.35119|1.18658e-06|1.97065|7.65565e-07|
|mild|learned_lead|2.35119|9.70154e-12|1.97065|5.35588e-12|
|mild|physical_ca|1.91246|0.475723|1.79491|0.214035|
|mild|physical_cv|2.48233|0.475723|1.67277|0.214035|
|mild|physical_enriched|1.97677|9.70154e-12|1.79491|5.35588e-12|
|mild|stable_history|1.79606|9.70154e-12|1.4914|5.35588e-12|
|stopgo|adaptive_damped_ca|1.45937|6.10281e-12|1.61171|4.01275e-12|
|stopgo|history|2.60322|1.18386e-06|2.10758|6.97602e-07|
|stopgo|learned_lead|2.60322|6.10281e-12|2.10758|4.01275e-12|
|stopgo|physical_ca|1.79353|0.451279|1.61171|0.236628|
|stopgo|physical_cv|4.92294|0.451279|3.27635|0.236628|
|stopgo|physical_enriched|1.45937|6.10281e-12|1.61171|4.01275e-12|
|stopgo|stable_history|2.17354|6.10281e-12|2.00697|4.01275e-12|

全部0.1/0.5/1/3秒结果见 by_family.csv，完整搜索记录见 selection.json。物理CA可以精确保持匀速；候选若仅靠更保守的前车趋势衰减获益，必须与同样使用val选择的衰减CA比较。这里没有未来前车脚本、隐藏场景标签或自车动作进入前车头。

### 相同完整CEM工作量的计算代价

|model|candidates|horizon|iterations|full_cycle_p50_ms|full_cycle_p95_ms|rollout_inference_p50_ms|scope|
|---|---|---|---|---|---|---|---|
|physical_enriched|64|30|3|7.921250000435975|8.994985000390443|4.221549996145768|shared host load, validation state, no vehicle I/O|
|history|64|30|3|16.449549999379087|23.519764999218744|12.534649997178349|shared host load, validation state, no vehicle I/O|
|stable_history|64|30|3|10.012449998612283|10.590989999218436|6.434000017179642|shared host load, validation state, no vehicle I/O|
|adaptive_damped_ca|64|30|3|10.4207000003953|12.025740000171934|6.914850006069173|shared host load, validation state, no vehicle I/O|

主机当前仍运行其他实验，计时包含64候选×30步×3迭代，实际争用可能改变deadline回退。该测量不是Pi端、独占硬实时或车辆I/O性能验收。原标准测试不会被替换；任何附加测试必须另行登记、冻结且一次执行，保持原失败结果可见。
