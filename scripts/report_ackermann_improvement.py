"""All-seed paired results of the preselected model-gradient experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    rows=json.loads((root/'episodes.json').read_text());summary=json.loads((root/'summary.json').read_text())
    reg=json.loads((root/'registration.json').read_text());cost=json.loads((root/'cost_ledger.json').read_text())
    seeds=reg['seeds'];methods=reg['methods'];effects=[];efficiency=reg['stage']=='efficiency'
    rng=np.random.default_rng(711)
    for control in (m for m in methods if m!='wm_gradient'):
        for dist in ('id','ood'):
            for metric in ('return_raw','completed','lateral_rmse_m','speed_rmse_mps','steering_rate_rms_radps'):
                differences=[]
                for seed in seeds:
                    a={r['episode_seed']:r for r in rows if r['method']=='wm_gradient' and r['training_seed']==seed and r['distribution']==dist}
                    b={r['episode_seed']:r for r in rows if r['method']==control and r['training_seed']==seed and r['distribution']==dist}
                    assert a.keys()==b.keys()
                    differences.append([float(a[k][metric])-float(b[k][metric]) for k in sorted(a)])
                diff=np.array(differences);seed_ids=rng.integers(len(seeds),size=(2000,len(seeds)))
                # IDs sort straight then curve; preserve 10+10 family strata.
                case_ids=np.concatenate([rng.integers(10,size=(2000,len(seeds),10)),10+rng.integers(10,size=(2000,len(seeds),10))],axis=2)
                samples=diff[seed_ids[:,:,None],case_ids].mean(axis=(1,2))
                effects.append(dict(control=control,distribution=dist,metric=metric,difference=float(diff.mean()),
                    low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),
                    per_seed=diff.mean(axis=1).tolist()))
    (root/'paired_effects.json').write_text(json.dumps(effects,indent=2)+'\n')
    lines=['# 模型梯度辅助PPO：新种子与新测试','',
        '本轮保留全部真实rollout和GAE，以冻结世界模型的多步可微预测辅助actor，而不是用一步TD样本替换真实经验。在线执行仍仅运行PPO，不运行MPC或世界模型。',
        '', '## 配置选择与隔离','',
        '只在训练seed0/1、独立88M验证场景中比较两个预先登记候选。按完成率、再按原始回报选择b：16步模型预测、辅助系数1、每4次PPO梯度加入一次模型辅助项、256次warm-start。',
        '', ('新seed6/7/8使用16384在线步，纯PPO总交互控制使用20480步。模型方法加3000离线交互共19384，仍少于该控制。配置保持冻结，新测试从99.6M起共600回合。' if efficiency else '选择文件冻结后，最终训练换为seed3/4/5，65536在线步。最终测试换为95.3M起的新场景：每seed每分布每场景族10回合，共600回合。测试不用于重新选择超参数或checkpoint。'),
        '', ('环境、原PPO配置与奖励不变。额外在线预算PPO有3200次Adam更新，模型含预热共2816次。另有不调用世界模型的动作平滑控制；各自成本完整记录。' if efficiency else '原PPO超参数、环境、奖励、观测、动作限制、预算与最终策略评价规则保持一致。另有额外256次PPO更新控制，以及不调用世界模型的动作平滑控制；不能把所有改善都归因于额外梯度或平滑项。'),
        '', '## 方法','',
        '冻结模型由独立激励数据拟合：物理先验加16项非线性特征的岭回归残差。Torch实现与NumPy拟合模型数值一致；模型参数是buffer，训练时只更新策略，模型不接收隐藏condition或未来真实状态。丰富物理模型作为同机制对照。',
        '', '用已知参考曲线采样2048个可行状态根，执行256次actor模型辅助预热；根由几何和范围采样，不来自测试或真实未来。此过程不是额外车辆交互，但256次更新和模型计算成本必须单列。',
        '', '在线阶段，每4次PPO小批量梯度更新，从当前真实小批量取16个根，递归使用当前actor均值与模型预测16步，对横向/航向/速度误差和指令变化的平均代价反向传播。真实GAE/return不改写；附加损失与PPO损失相加后，统一一次backward、梯度裁剪和Adam step。',
        '', '辅助项是可微的预测跟踪代价，包含比环境奖励更强的动作变化约束，也没有环境奖励的代价饱和。因此算法改变了训练目标；不是原版PPO的无偏梯度保证，也不应称为原版MBPO或Dreamer。actor均值滚动也不等于随机策略期望的精确积分。',
        '', '简式：`总损失 = PPO损失 + beta × 预测时域内的平均跟踪/动作变化代价`。模型转移与actor在每一步递归连接梯度，因而能跨执行器延迟分配转向信用；这与旧h1 critic-bootstrap替换机制不同。',
        '', '## 全部最终测试','',
        '|方法|分布|完成/回合|道路越界|原始回报|横向RMSE m|速度RMSE m/s|转角变化率RMS rad/s|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['method']}|{r['distribution']}|{r['completed']}/{r['episodes']}|{r['departures']}|{r['return_raw']:.3f}|{r['lateral_rmse_m']:.4f}|{r['speed_rmse_mps']:.4f}|{r['steering_rate_rms_radps']:.4f}|")
    lines+=['','## 分层配对差异（wm_gradient − 对照）','',
        '训练seed先重采样，再在seed内按直线/曲线分别配对重采样，共2000次。只有三个训练种子，区间不支持跨任务普遍保证，多重比较未校正。完成/回报越大越好，误差/转角变化率越小越好。',
        '', '|对照|分布|指标|差值|95%区间|逐seed差值|','|---|---|---|---:|---|---|']
    for e in effects:
        lines.append(f"|{e['control']}|{e['distribution']}|{e['metric']}|{e['difference']:.5f}|[{e['low']:.5f}, {e['high']:.5f}]|{', '.join(f'{x:.4f}' for x in e['per_seed'])}|")
    lines+=['','## 每个训练种子','', '|seed|方法|分布|完成/回合|原始回报|横向RMSE|','|---:|---|---|---:|---:|---:|']
    for seed in seeds:
        for method in methods:
            for dist in ('id','ood'):
                rr=[r for r in rows if r['training_seed']==seed and r['method']==method and r['distribution']==dist]
                lines.append(f"|{seed}|{method}|{dist}|{sum(r['completed'] for r in rr)}/{len(rr)}|{np.mean([r['return_raw'] for r in rr]):.3f}|{np.mean([r['lateral_rmse_m'] for r in rr]):.4f}|")
    lines+=['','## 数据与计算账本','', '|方法|在线步/任务|总Adam调用/任务|额外离线交互/任务|imagined steps/任务|预热+训练平均秒|','|---|---:|---:|---:|---:|---:|']
    for method in methods:
        rr=[r for r in cost if r['method']==method];r=rr[0]
        lines.append(f"|{method}|{r['online_steps']}|{r['total_optimizer_steps']}|{r['offline_interactions']}|{r['imagined_steps']}|{np.mean([x['warm_seconds']+x['training_seconds'] for x in rr]):.2f}|")
    lines+=['','模型池预训练和验证选择预算另计，不能用同在线步数掩盖3000离线交互、候选筛选、额外模型计算或墙钟开销。共享CPU主机计时不代表Pi。actor warm-start只更新actor参数，critic/log_std的Adam计数不同；审计分别核验。',
        '', '## 判断规则与限制','',
        '预登记主指标为新测试原始回报，同时完成率不能降低；横向RMSE和转向平顺性为次指标。只改善次指标或某个种子，不能表述为稳定总体胜出。',
        '', '仍是给定参考路径的合成仿真，执行器状态为特权输入，尚无视觉输入、障碍规划、真实车辆标定或部署。旧一步替换方法的失败、验证候选a和全部最终种子均保留，不替换旧报告，不把本轮与旧测试混为一个总体效果。',
        '', '## 复现','', '```powershell',
        '.\\.venv\\Scripts\\python.exe run_ackermann_improvement.py --stage select --output outputs/ackermann_rl/new_selection --workers 4',
        '.\\.venv\\Scripts\\python.exe run_ackermann_improvement.py --stage final --selection outputs/ackermann_rl/new_selection/selection.json --output outputs/ackermann_rl/new_final --workers 5',
        '.\\.venv\\Scripts\\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_final',
        '.\\.venv\\Scripts\\python.exe scripts/report_ackermann_improvement.py --run outputs/ackermann_rl/new_final','```']
    audit=json.loads((root/'audit.json').read_text());lines+=['',f"独立审计：{audit['status']}，{audit['episodes']}回合、{audit['steps']}转移，{audit['error_count']}错误。模型生成/辅助样本没有逐步完整训练日志，不能声称形式化验证了全部训练执行。"]
    lines+=['','![预先固定的第一条ID曲线场景](comparison.png)',
        '', '图按首个训练seed、ID、curve、pair0的固定身份选择，不按表现选最佳回合。']
    if efficiency:
        lines=[line.replace('--stage final --selection','--stage efficiency --selection') for line in lines]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Predeclared illustration: first new training seed, first ID curve case.
    seed=seeds[0];eid=(99_000_000 if efficiency else 95_000_000)+seed*100_000+1000
    rng=np.random.default_rng(eid);phase=rng.uniform(-.8,.8);vref=rng.uniform(2,4)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8));x=np.linspace(0,31,300);y=1.3*(np.sin(x/9+phase)-np.sin(phase))
    axes[0].plot(x,y,'k--',label='reference');axes[1].axhline(vref,color='k',linestyle='--')
    for method in ('ppo','smooth_ppo','wm_gradient'):
        with np.load(root/f'seed{seed}'/method/'test/trajectories.npz') as f:t=f[str(eid)]
        axes[0].plot(t[:,6],t[:,7],label=method);axes[1].plot(np.arange(len(t))*.1,t[:,9],label=method)
        axes[2].plot(np.arange(len(t))*.1,t[:,17],label=method)
    axes[0].set(xlabel='x [m]',ylabel='y [m]');axes[1].set(xlabel='time [s]',ylabel='speed [m/s]')
    axes[2].set(xlabel='time [s]',ylabel='front-wheel command [rad]')
    for axis in axes:axis.legend(fontsize=8);axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    print(root/'REPORT.md')


if __name__=='__main__':main()
