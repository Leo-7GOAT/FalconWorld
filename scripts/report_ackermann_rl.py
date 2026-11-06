"""Report every frozen seed; fixed first curve trace, never best-case selection."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    reg=json.loads((root/'registration.json').read_text());summary=json.loads((root/'summary.json').read_text())
    ledger=json.loads((root/'cost_ledger.json').read_text());audit=json.loads((root/'audit.json').read_text())
    records=[]
    for seed in reg['spec']['seeds']:
        for method in reg['methods']:
            records += [dict(json.loads(line),method=method) for line in (root/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines()]
    lines=['# 二维阿克曼：横纵向联合 PPO','',
        '状态：已实现并实际训练；本地合成仿真，不是视觉驾驶、实车部署或稳定模型优势结论。',
        '',f"固定预算：{len(reg['spec']['seeds'])} seeds × 3 方法，每任务 {reg['spec']['steps']} 在线步。最终 checkpoint 测试，共 {len(records)} 回合。",'',
        '## 动作、观测与模型分支','',
        'PPO 独立输出两个连续归一化动作，映射为期望加速度 [-3,2] m/s² 和前轮转角 [-0.5,0.5] rad；共同限制加速度命令变化率 5 m/s³、转角命令变化率 0.8 rad/s。没有传统控制器替代策略的转向输出。',
        '', '14 维观测包括全局位姿、速度、仿真执行器加速度/转角、上一实际指令、已知路线参数/目标速度、横向误差、航向误差和 3m 前瞻参考偏差。没有图像、隐藏动力学 condition 或未来真实状态。路线由上游提供，尚不涉及自主路线选择或障碍物绕行。',
        '', '世界模型复用已实现的二维物理＋16项基函数岭回归残差，预测执行器加速度、转角和横摆修正；不是新建视觉网络。每训练种子用 24×100 train、6×100 val 的自由空间激励数据拟合，ridge 仅按 val 选取；训练前冻结。丰富物理模型是相同机制的强对照。',
        '', '辅助 PPO 从当轮真实观测池抽根，用同一个冻结行为策略批量生成一步分支；保留原始 Gaussian 动作及 log probability，动力学只接收裁剪、限幅与限速后的指令；替换一半固定学习 buffer。真正越界/抵达目标终止不 bootstrap，160步超时保留最终状态 bootstrap。合成一步 TD 与真实 GAE 分开，不能对混合 buffer 重算 GAE。',
        '', '主比较采用 SB3 默认 2048 rollout、64 batch、10 epochs；实际 Adam 更新数相同，模型生成不增加策略更新。纯 PPO 原生离线交互为0，模型方法额外3000步，未宣称同总交互/同时间预算。一步 TD 的目标时域与模型分布均改变，不能将差异全部归因于预测精度。',
        '', '## 场景和训练奖励','',
        'ID：直线/正弦曲线，振幅0或1.3m，目标速度2–4m/s。OOD：曲线振幅1.8m、目标速度4–5m/s、执行器响应时间乘1.3。道路宽7m、轴距1m、车长1.5m，均为仿真假设，不是R550标定。终点x=30m，最长16秒，扫掠车身越出道路即失败；本轮没有其他交通参与者，因此不能把道路越界称为车辆碰撞。',
        '', 'v1累计负跟踪代价使seed0验证策略较早越界，完整失败结果保留在quick。依据验证日志与奖励结构统一修正为v2；未读取v1正式test结果来选奖励。v2各方法使用同一有界存活奖励、跟踪与动作代价、目标到达奖励50、越界奖励−20；训练统一缩放0.01，测试报告原始回报。没有根据v2 test调参或选择checkpoint。',
        '', '## 全部冻结测试结果','',
        '|方法|分布|回合|完成|道路越界|平均原始回报|横向RMSE m|速度RMSE m/s|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['method']}|{r['distribution']}|{r['episodes']}|{r['completions']}|{r['departures']}|{r['return_raw']:.2f}|{r['lateral_rmse_m']:.3f}|{r['speed_rmse_mps']:.3f}|")
    lines += ['', '## 逐训练种子（全部保留）','', '|seed|方法|分布|完成/回合|越界|平均回报|','|---:|---|---|---:|---:|---:|']
    for seed in reg['spec']['seeds']:
        for method in reg['methods']:
            for dist in ('id','ood'):
                rr=[r for r in records if r['training_seed']==seed and r['method']==method and r['distribution']==dist]
                lines.append(f"|{seed}|{method}|{dist}|{sum(r['completed'] for r in rr)}/{len(rr)}|{sum(r['road_departure'] for r in rr)}|{np.mean([r['return_raw'] for r in rr]):.2f}|")
    lines += ['', '## 训练成本','', '|方法|平均训练秒|每任务实际Adam次数|每任务合成替换|额外离线步/任务|','|---|---:|---:|---:|---:|']
    for method in reg['methods']:
        rr=[r for r in ledger if r['method']==method]
        lines.append(f"|{method}|{np.mean([r['training_seconds'] for r in rr]):.2f}|{rr[0]['optimizer_steps']}|{rr[0]['synthetic_replaced']}|{rr[0]['offline_interactions']}|")
    lines += ['', '共享CPU主机计时，不代表Pi实时性；离线数据拟合及验证/测试开销另计，见各目录manifest/complete.json。',
        '', '## 审计与解释边界','',f"独立审计：{audit['status']}，{audit['episodes']}回合，{audit['steps']}转移，{audit['error_count']}错误。",
        '', '独立重建真值动力学、扫掠车身道路判定、原始奖励和指标，并检查轨迹连续性、数据划分、模型/策略SHA、zip中的真实Adam步数。没有逐样本训练执行日志，因此不是完整训练过程的形式化证明。',
        '', '仅三个训练种子，结果不足以建立稳定总体优势；学习世界模型辅助方法可能劣于纯PPO，所有失败与超时均保留。当前实现证明联合转向/加速度策略确实可训练和可加载，不表示已达到车辆安全或部署门槛。',
        '', '![固定第一条曲线测试轨迹](tracking_example.png)',
        '', '图选seed0、ID、curve、pair0，按事先固定身份选取，不按表现选最佳回合。',
        '', '## 复现','', '```powershell', '.\\.venv\\Scripts\\python.exe run_ackermann_rl.py --preset quick --workers 3 --output outputs/ackermann_rl/new_run',
        '.\\.venv\\Scripts\\python.exe scripts/audit_ackermann_rl.py --run outputs/ackermann_rl/new_run',
        '.\\.venv\\Scripts\\python.exe scripts/report_ackermann_rl.py --run outputs/ackermann_rl/new_run', '```', '',
        '首次复现先按根README执行setup_rl.ps1；输出目录必须不存在，不会覆盖冻结记录。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    seed=reg['spec']['seeds'][0];eid=90_000_000+seed*100_000+1000
    rng=np.random.default_rng(eid);phase=rng.uniform(-.8,.8);vref=rng.uniform(2,4)
    fig,ax=plt.subplots(1,3,figsize=(13,3.8));x=np.linspace(0,31,400);y=1.3*(np.sin(x/9+phase)-np.sin(phase))
    ax[0].plot(x,y,'k--',label='reference');ax[0].fill_between(x,y-3.5,y+3.5,color='gray',alpha=.15)
    for method in reg['methods']:
        with np.load(root/f'seed{seed}'/method/'test/trajectories.npz') as f:t=f[str(eid)]
        ax[0].plot(t[:,6],t[:,7],label=method)
        ax[1].plot(np.arange(len(t))*.1,t[:,9],label=method)
        ax[2].plot(np.arange(len(t))*.1,t[:,17],label=method)
    ax[1].axhline(vref,color='k',linestyle='--');ax[0].set(xlabel='x [m]',ylabel='y [m]',title='Fixed first curve / ID case')
    ax[1].set(xlabel='time [s]',ylabel='speed [m/s]');ax[2].set(xlabel='time [s]',ylabel='front-wheel command [rad]')
    for axis in ax:axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'tracking_example.png',dpi=160);plt.close(fig)
    print(root/'REPORT.md')


if __name__=='__main__':main()
