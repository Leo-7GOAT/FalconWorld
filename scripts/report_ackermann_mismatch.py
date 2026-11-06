"""Report prediction versus control claims separately, including null results."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def effects(rows,kind):
    rng=np.random.default_rng(9701);out=[]
    if kind=='prediction':
        strata=[(split,h) for split in ('test','ood') for h in (1,5,10,30)]
        metrics=('steering_mae_rad','position_error_m');method_key='model';case_key='episode'
    else:
        strata=[(dist,None) for dist in ('id','ood')];metrics=('return_raw','completed','lateral_rmse_m')
        method_key='method';case_key='episode_seed'
    for control in ('rich','affine','mechanism'):
        for split,h in strata:
            for metric in metrics:
                differences=[]
                for seed in (0,1,2):
                    def selected(r):return r['training_seed']==seed and (r['split']==split and r['horizon_steps']==h if kind=='prediction' else r['distribution']==split)
                    aa={r[case_key]:r for r in rows if selected(r) and r[method_key]=='residual'}
                    bb={r[case_key]:r for r in rows if selected(r) and r[method_key]==control}
                    assert aa.keys()==bb.keys()
                    differences.append([float(aa[k][metric])-float(bb[k][metric]) for k in sorted(aa)])
                d=np.array(differences);si=rng.integers(3,size=(2000,3))
                if kind=='prediction':ci=rng.integers(d.shape[1],size=(2000,3,d.shape[1]))
                else:ci=np.concatenate([rng.integers(10,size=(2000,3,10)),10+rng.integers(10,size=(2000,3,10))],axis=-1)
                sample=d[si[:,:,None],ci].mean(axis=(1,2))
                out.append(dict(control=control,split=split,horizon_steps=h,metric=metric,difference=float(d.mean()),
                    low=float(np.quantile(sample,.025)),high=float(np.quantile(sample,.975)),per_seed=d.mean(axis=1).tolist()))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--warm',type=Path)
    args=p.parse_args();root=args.run
    prediction=json.loads((root/'prediction_episodes.json').read_text());summary=json.loads((root/'prediction_summary.json').read_text())
    control=json.loads((root/'episodes.json').read_text());closed=json.loads((root/'summary.json').read_text())
    pe=effects(prediction,'prediction');ce=effects(control,'control')
    (root/'paired_prediction_effects.json').write_text(json.dumps(pe,indent=2)+'\n')
    (root/'paired_control_effects.json').write_text(json.dumps(ce,indent=2)+'\n')
    lines=['# 残差处理未建模转向失配：合成实验','',
        '本轮区分三个结论：残差是否修正结构失配、是否改善模型预测、预测改善是否转化为PPO闭环收益。不能将三者等同。',
        '', '## 植物模型和基线公平性','',
        '原七参数丰富物理模型缺少转向标定偏置、增益与平滑死区。本轮只在转角执行器引入这些合成失配，车辆几何、纵向/横摆基础机制、奖励和路线任务保持一致；不是R550实测。',
        '', '设转向驱动力 e=1.08*u_delta+0.025-delta，死区宽度w=0.032*(1+0.30*(v/5)^2)，使用F(e)=e-w*tanh(e/w)。F在零处为零、与e同号，导数1-sech²(e/w)>=0，避免非单调或反向响应。',
        '', '标准工程模型中的死区与库仑/黏性摩擦参考：[MathWorks Dead Zone](https://www.mathworks.com/help/simulink/slref/deadzone.html)、[Coulomb and Viscous Friction](https://www.mathworks.com/help/simulink/slref/coulombandviscousfriction.html)。这里只借鉴失配类型；具体平滑函数、速度系数和数值为仿真假设，并非从这些文档或实车获得的参数。',
        '', '所有方法使用同一2400 train＋600 val转移，并允许物理模型重新拟合：',
        '', '- rich：重新辨识原七参数丰富物理族。',
        '- affine：扩大为可校准转向增益/偏置的物理族，有效9参数。',
        '- residual：在同一个affine模型上学习转角残差，输入速度、指令与当前转角之差、转向指令；3→64→64→1 tanh网络，所有尺度只用train，val早停。',
        '- mechanism：知道平滑摩擦结构、但参数仍从同一train数据估计的物理族，有效12参数。它是匹配结构的强对照，不直接读取真值参数。',
        '', '网络残差不是用真实公式手工写入的特征项。训练/验证/预测测试及PPO测试的随机种子命名空间隔离，模型先冻结再测试。原非单调近似v1已归档并排除，不作为物理证据；v2换用全新310M/320M/330M/340M/350M/360M种子空间。',
        '', '## 冻结预测测试','',
        '每训练种子10个独立ID测试episode及10个响应时间改变的OOD episode。各时域用不重叠窗口，先汇总episode，避免把窗口当独立样本。',
        '', '|分布|时域步数|模型|episode数|转角MAE rad|位置误差m|','|---|---:|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['split']}|{r['horizon_steps']}|{r['model']}|{r['episodes']}|{r['steering_mae_rad']:.8g}|{r['position_error_m']:.8g}|")
    lines+=['','## PPO冻结闭环：全部种子','',
        '三训练种子×五方法，每任务16384在线步。辅助方法统一使用原先验证选出的16步模型梯度、256次actor预热、每4次更新附加一次模型损失；未按本轮test改训练参数或选checkpoint。各辅助模型参数固定。',
        '', '|方法|分布|完成/60|原始回报|横向RMSE m|速度RMSE m/s|转角变化率RMS|','|---|---|---:|---:|---:|---:|---:|']
    for r in closed:
        lines.append(f"|{r['method']}|{r['distribution']}|{r['completed']}/60|{r['return_raw']:.4f}|{r['lateral_rmse_m']:.5f}|{r['speed_rmse_mps']:.5f}|{r['steering_rate_rms_radps']:.5f}|")
    lines+=['','## 配对差异：residual − 对照','',
        '训练seed先重采样，再在seed内配对重采样episode；PPO测试另按直线/曲线分层。2000次bootstrap，只有三个训练种子，多重比较未校正。误差越小越好，回报/完成越大越好。',
        '', '|任务|对照|分布|时域|指标|差值|95%区间|逐seed差值|','|---|---|---|---|---|---:|---|---|']
    for kind,entries in [('prediction',pe),('PPO',ce)]:
        for e in entries:
            lines.append(f"|{kind}|{e['control']}|{e['split']}|{e['horizon_steps']}|{e['metric']}|{e['difference']:.7g}|[{e['low']:.7g},{e['high']:.7g}]|{', '.join(f'{v:.5g}' for v in e['per_seed'])}|")
    lines+=['','## 解释边界','',
        'ID单步及递归预测可以检验残差是否修正未建模失配，但不能据此声称PPO控制全面更好。结构知情物理模型预测仍更准确，说明残差不是超越正确物理定律，而是在不手工写入该机制时学习补偿。',
        '', 'OOD同时改变执行器时间常数和横摆响应；残差训练没有见过这种漂移。OOD多步预测不保证改善，例如3秒位置误差可能弱于仿射物理对照。不能用ID结果宣称残差自动识别任意外部变化。',
        '', 'PPO表中原回报、横向精度、速度误差和转向变化可能相互冲突；完成率已经饱和时不能只报告更高回报。所有种子及配对差异均保留。模型网络拟合、actor预热和模型反向传播有额外成本；纯PPO原生离线预算为0，模型方法3000，辅助更新及warm-start不能视为免费。']
    if args.warm:
        lines+=['','## 探索性补充：在线适应前与2048步后','',
            '该诊断是在主实验结果后提出的新问题，不替代主预登记结论。模型与全部训练超参数保持冻结，另用380M新场景，0和2048两个预算都报告，不挑最佳预算。所有控制器均为模型辅助PPO，不包含未训练纯PPO。',
            '', '|在线步|方法|分布|完成/60|原始回报|横向RMSE|','|---:|---|---|---:|---:|---:|']
        for budget in (0,2048):
            folder=args.warm/f'budget{budget}';rows=json.loads((folder/'episodes.json').read_text());ee=effects(rows,'control')
            (folder/'paired_control_effects.json').write_text(json.dumps(ee,indent=2)+'\n')
            for method in ('rich','affine','residual','mechanism'):
                for dist in ('id','ood'):
                    rr=[r for r in rows if r['method']==method and r['distribution']==dist]
                    lines.append(f"|{budget}|{method}|{dist}|{sum(r['completed'] for r in rr)}/60|{np.mean([r['return_raw'] for r in rr]):.4f}|{np.mean([r['lateral_rmse_m'] for r in rr]):.5f}|")
    a=json.loads((root/'audit.json').read_text());pa=json.loads((root/'prediction_audit.json').read_text())
    lines+=['','## 独立审计与复现','',
        f"主闭环{a['episodes']}回合/{a['steps']}次转移、预测{pa['episode_model_horizon_rows']}条episode-模型-时域汇总均独立审计，零错误。预测审计用独立NumPy实现网络推理和物理递推；闭环审计不导入模拟器，重建真值动力学、扫掠车身几何、奖励、终止和Adam预算。没有完整逐训练样本执行轨迹，审计不是形式化训练证明。",
        '', '```powershell',
        '.\\.venv\\Scripts\\python.exe run_ackermann_mismatch.py --stage prepare --output outputs/ackermann_rl/new_mismatch',
        '.\\.venv\\Scripts\\python.exe run_ackermann_mismatch.py --stage evaluate --output outputs/ackermann_rl/new_mismatch --workers 5',
        '.\\.venv\\Scripts\\python.exe scripts/audit_ackermann_mismatch_prediction.py --run outputs/ackermann_rl/new_mismatch',
        '.\\.venv\\Scripts\\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_mismatch',
        '.\\.venv\\Scripts\\python.exe run_ackermann_mismatch_warm.py --source outputs/ackermann_rl/new_mismatch --output outputs/ackermann_rl/new_warm --workers 4',
        '.\\.venv\\Scripts\\python.exe scripts/report_ackermann_mismatch.py --run outputs/ackermann_rl/new_mismatch --warm outputs/ackermann_rl/new_warm','```',
        '', '![预测误差曲线](prediction_errors.png)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    for name in ('rich','affine','residual','mechanism'):
        rr=[r for r in summary if r['split']=='test' and r['model']==name]
        ax[0].semilogy([r['horizon_steps']*.1 for r in rr],[r['steering_mae_rad'] for r in rr],'-o',label=name)
        rr=[r for r in rr if r['horizon_steps']>1]
        ax[1].semilogy([r['horizon_steps']*.1 for r in rr],[r['position_error_m'] for r in rr],'-o',label=name)
    ax[0].set(xlabel='horizon [s]',ylabel='steering MAE [rad]',title='Unseen ID friction episodes')
    ax[1].set(xlabel='horizon [s]',ylabel='position error [m]',title='Unseen ID recursive prediction')
    for axis in ax:axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'prediction_errors.png',dpi=160);plt.close(fig)
    print(root/'REPORT.md')


if __name__=='__main__':main()
