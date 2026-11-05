"""Read PPO run evidence and write a Chinese report; never train or touch devices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.rl_evaluation import evaluation_cases,summarize


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else default


def write_csv(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/rl_ppo/standard'))
    p.add_argument('--output',type=Path,default=Path('docs/RL_REPORT.md'))
    args=p.parse_args();root=args.input
    registration=read(root/'registration.json')
    if registration is None:p.error('Run registration.json is required; no evidence to report')
    artifacts=root/'report';artifacts.mkdir(parents=True,exist_ok=True)
    preset=registration['preset'];seeds=registration['spec']['seeds'];methods=registration['methods']
    rows=[];ledgers=[];curves=[];missing=[];expected_keys=set();actual_keys=[]
    for seed in seeds:
        for method in methods:
            out=root/f'seed{seed}'/method
            ledger=read(out/'complete.json')
            if ledger is not None:ledgers.append(ledger)
            else:missing.append(f'seed{seed}/{method}/complete.json')
            common=read(root/'common_data/manifest.json',{})
            offline=common.get('training_transitions',0)+common.get('validation_transitions',0)
            for r in read(out/'learning_curve.json',[]):
                curves.append(dict(r,common_offline_transitions=offline,
                    shared_total_real_steps=offline+r['new_real_steps'],
                    native_total_real_steps=r['new_real_steps']+(offline if method.startswith('wm_') or method=='physics_ppo' else 0)))
            cases=evaluation_cases(seed,'smoke' if preset=='smoke' else 'standard')
            expected_keys.update((method,seed,c['family'],c['condition'],c['seed']) for c in cases)
            path=out/'test/episodes.jsonl'
            if path.exists():
                for line in path.read_text(encoding='utf8').splitlines():
                    if line.strip():
                        r=json.loads(line);rows.append(r)
                        actual_keys.append((r['method'],r['training_seed'],r['family'],r['condition'],r['scenario_seed']))
    actual_set=set(actual_keys)
    audit=dict(expected_jobs=len(methods)*len(seeds),completed_jobs=len(ledgers),
        expected_episodes=len(expected_keys),actual_episodes=len(rows),unique_episodes=len(actual_set),
        missing_episode_count=len(expected_keys-actual_set),unexpected_episode_count=len(actual_set-expected_keys),
        duplicate_episode_count=len(rows)-len(actual_set),missing_jobs=missing,
        verified_complete=not missing and actual_set==expected_keys and len(actual_keys)==len(actual_set))
    (artifacts/'completeness.json').write_text(json.dumps(audit,indent=2))
    diagnosis_paths=[Path('docs/PPO_VALIDATION_DIAGNOSIS.md'),
        Path('outputs/rl_ppo_budget_validation/seed0_131072/DIAGNOSIS.md')]
    (artifacts/'diagnosis_sources.json').write_text(json.dumps({str(p):
        dict(exists=p.exists(),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)
        for p in diagnosis_paths},indent=2))
    write_csv(artifacts/'cost_table.csv',ledgers)
    write_csv(artifacts/'learning_curves.csv',curves)
    summary=[];comparisons={}
    if rows:
        summary,main_effects=summarize(rows,artifacts,reference='ppo' if audit['verified_complete'] else '__partial_no_inference__')
        if audit['verified_complete']:
            for reference in ('ppo','physics_ppo','ppo_total_budget'):
                if reference in methods:
                    if reference=='ppo':
                        effects=main_effects
                    else:
                        paired_rows=[r for r in rows if r['method'] in ('wm_ppo',reference)]
                        _,effects=summarize(paired_rows,artifacts/f'against_{reference}',reference=reference)
                    comparisons[reference]=[r for r in effects if r['method']=='wm_ppo' and r['family']=='all']
    text=['# 世界模型辅助PPO：证据、收益、代价与适用边界','',
        f"运行目录：`{root}`；preset={preset}；算法为categorical clipped PPO，模型分支辅助on-policy更新，不能称为原版MBPO。",'',
        f"训练学习奖励缩放learning_reward_scale={registration.get('learning_reward_scale','历史运行未登记（未缩放版本）')}。评估始终使用未缩放的环境原始reward/return，不能将学习目标的缩放误读为评价指标变更。",'',
        f"完整性：{'已验证齐全' if audit['verified_complete'] else '进行中／未齐全，不下优势结论'}；任务{len(ledgers)}/{audit['expected_jobs']}，episode {len(rows)}/{len(expected_keys)}，缺失{audit['missing_episode_count']}，重复{audit['duplicate_episode_count']}。",'',
        '固定预算、最终checkpoint测试，val仅监控、不挑test-best。只运行本地仿真，没有R550连接、上线或部署。', '',
        '## 当前交付边界与PPO基线质量','',
        '本报告是每方法每seed 32k在线步首轮正式对照（ppo_total_budget另计），并非整个世界模型改善RL目标已经完成。完整性通过只能证明这轮登记实验齐全，不能证明PPO已学到可用驾驶策略。', '',
        '仅train/val的后续固定预算诊断把scaled纯PPO seed0从32768步继续到131072步；最终8个固定ID验证场景完成率0.50、碰撞率0.50、raw return 1.93，仍未达到预登记完成率≥0.9、碰撞率≤0.1、return≥30的联合门槛。65k/98k中间checkpoint全部保留，未择优替代最终131k。单seed、8场景不能证明更大预算普遍无效，但足以说明当前PPO基线质量尚不足，不能利用弱基线宣称世界模型有效，也不能据负结果断言世界模型普遍无用。', '',
        '诊断来源：docs/PPO_VALIDATION_DIAGNOSIS.md与outputs/rl_ppo_budget_validation/seed0_131072/DIAGNOSIS.md；源摘要另存report/diagnosis_sources.json。SB3默认设置固定204800步的仅验证集交叉检查尚在进行中（pending），不读取test，不挑赢家，不能代替本轮正式测试。', '',
        '## 仅训练／验证集的优化校准','',
        '校准轮将真实和模型GAE学习目标统一使用reward×0.01，环境、评价raw reward与测试清单不变。缩放依据仅train/val的实际Adam步长诊断：fresh Adam一步实测actor参数步长由0.006999变为0.021098（约3.01倍），不能按clip系数声称历史Adam每步被压小82倍；不读取测试结果选择scale。仅验证greedy/stochastic诊断分别4/8和32/64碰撞，两者仅8个独立场景，不能归咎于greedy单一原因。未缩放smoke/standard已有产物保留在outputs/rl_ppo/smoke_unscaled_validation_trial与standard_unscaled_validation_trial，全部已产生checkpoint/metrics归档，不按胜负丢弃旧轮，不宣称旧轮全部登记job都完整。校准重训复用相同common_data/worldmodels，旧轮不替代新轮正式结果。', '',
        '## 实验约定','',
        'ppo使用真实在线rollout；wm_ppo使用history模型一步分支、0.5比例；physics_ppo使用丰富辨识物理同样分支；wm_no_history和wm_long检验历史与分支长度；real_reuse_control使用额外2倍更新，作为独立计算控制。ppo_total_budget把共同offline转移数改花为额外online步，使其原生总真实训练交互与wm_ppo相同，允许更多更新且单独记账。离线数据仅拟合预测模型，不进入PPO actor/critic梯度。真实与模型轨迹分别GAE，分支old_log_prob来自更新前同一冻结behavior策略。', '',
        f"主比较每method/seed新增online预算为{registration['spec']['steps']}真实步，rollout512，batch128，完整rollout每轮16个梯度minibatch（real_reuse_control32）。ppo_total_budget另加offline对应online步数，尾段与更新数按实际账本报告，不能归入同online/同更新数主比较。", '',
        '## 最终策略闭环','',
        '|方法|分布|回合|碰撞|完成|return|进度m|gap RMSE m|jerk RMS m/s³|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append(f"|{r['method']}|{r['condition']}|{r['episodes']}|{r['collisions']}|{r['completions']}|{r['return']:.3f}|{r['progress_m']:.3f}|{r['gap_rmse_m']:.3f}|{r['jerk_rms_mps3']:.3f}|")
    text+=['','未齐全表格仅描述已完成episode，不能据此排名；碰撞早停长度与全部轨迹均保留。完成要求无碰撞跑满120步且进度达到参考阈值，不能仅凭停车的低碰撞率宣称优势。','',
           '## 世界模型相对纯PPO、强物理与同总交互控制的配对效应','']
    if not audit['verified_complete']:
        text+=['正式配对效果与区间待全部登记任务结束后生成；本报告不对partial结果作优势判定。']
    else:
        text+=['差值为wm_ppo−对照；return/progress/completed越高越好，碰撞/误差越低越好。训练seed先重采样，再在seed内按family分层配对episode重采样，95%区间。', '',
               '|对照|分布|指标|差值|95%区间|配对数|','|---|---|---|---:|---|---:|']
        for reference,effects in comparisons.items():
            for r in effects:
                text.append(f"|{reference}|{r['condition']}|{r['metric']}|{r['difference']:.4f}|[{r['ci_low']:.4f},{r['ci_high']:.4f}]|{r['pairs']}|")
        text+=['','smoke只有1个训练seed，区间仅反映场景抽样，不构成训练随机性的优势证据。standard只有3seed，区间仍不稳定。须同时检查碰撞、完成、进度和回报；区间跨零意味着当前证据未辨别差异，不应写成等效或通用优越。']
    text+=['','## 成本与样本效率','',
        'worldmodel_pretrain_wall_s是整套8种预测模型预训练墙钟上界，给物理与神经模型同样标记；不能称单个history/physics模型的训练成本。主训练墙钟另计model_generation_wall_s，验证/测试真实步单列，不混入训练预算。', '',
        '|方法|seed|新增online步|共同offline转移|native总训练交互|shared数据预算交互|梯度minibatches|synthetic生成步|训练墙钟s|模型生成s|预训整套上界s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ledgers:
        fields=('method','training_seed','new_real_steps','common_offline_transitions','native_total_training_interactions',
            'shared_data_budget_interactions','policy_updates','synthetic_generated','training_wall_s','model_generation_wall_s','worldmodel_pretrain_wall_s')
        text.append('|'+'|'.join(f'{r[k]:.3f}' if isinstance(r.get(k),float) else str(r.get(k,'未验证')) for k in fields)+'|')
    text+=['','纯PPO原生不需要offline数据，因此native总成本可低于shared预算视角；提供相同采集成本预算不等于PPO使用了这些数据。synthetic步不是real env steps，额外复用梯度不是免费收益。缺失成本字段一律未验证，不用配置反推actual。','']
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(18,4))
        for mi,method in enumerate(methods):
            color=plt.get_cmap('tab10')(mi%10)
            marker=('o','s','^','D','v','P','X')[mi%7]
            method_values=[]
            for seed in seeds:
                values=sorted((r for r in curves if r['method']==method and r['training_seed']==seed),key=lambda r:r['new_real_steps'])
                if not values:continue
                method_values.extend(values)
                for ax,xkey in zip(axes,('new_real_steps','native_total_real_steps','shared_total_real_steps')):
                    ax.plot([r[xkey] for r in values],[r['mean_return'] for r in values],color=color,alpha=.20,linewidth=.8)
            for ax,xkey in zip(axes,('new_real_steps','native_total_real_steps','shared_total_real_steps')):
                xs=sorted({r[xkey] for r in method_values})
                ys=[float(np.mean([r['mean_return'] for r in method_values if r[xkey]==x])) for x in xs]
                if xs:ax.plot(xs,ys,label=method,color=color,marker=marker,markersize=3,linewidth=1.8)
        for ax,label in zip(axes,('New online real simulator steps','Native consumed offline + online real steps','Shared budget accounting (not native consumption)')):
            ax.set(xlabel=label,ylabel='Fixed ID validation mean episode return');ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(artifacts/'sample_efficiency.png',dpi=150);plt.close(fig)
        # Relative to the report destination; portable within the local project.
        import os
        relative=os.path.relpath(artifacts/'sample_efficiency.png',args.output.parent).replace('\\','/')
        text+=['## 验证学习曲线','',
            'val仅ID、每族smoke1/standard2case；四分之一预算监控，不据此挑test-best。左图新增online，中图原生实际消耗的offline+online真实交互；纯PPO、real_reuse_control和ppo_total_budget均原生零offline。右图仅共同预算记账，不能当作真实数据消耗；原生成本推算采用该方法是否使用预测模型与已记录offline转移数。细淡线为独立训练seed，带标记粗线为同一步数跨seed均值，不是置信区间；不能把曲线点当独立训练重复。', '',f'![样本效率]({relative})','']
    text+=['## 证据链与失效边界','',
        '需要连接预测能力、分支误差/覆盖、PPO学习曲线、最终闭环与总成本。当前登记评价可以检验policy闭环、成本与消融；若未另记录分支预测误差，不能声称完整机理已经证明。', '',
        '预测误差降低不保证PPO改善；模型偏差可能被策略放大。OOD、不可预知急刹、有限数据和短分支覆盖仍是边界。零碰撞不构成通用安全保证；主机周期包含仿真step，不代表Pi性能。全部原始日志、成本、按family指标和配对区间在运行目录保留。', '',
        '正式最终报告必须来自校准后的outputs/rl_ppo/standard且完整性通过；smoke只证明工程路径，未缩放归档只作保留的历史试验。','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(text),encoding='utf8')
    print(json.dumps(dict(report=str(args.output),audit=audit),ensure_ascii=False))


if __name__=='__main__':main()
