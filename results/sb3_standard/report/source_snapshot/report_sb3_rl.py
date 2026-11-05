"""Read independent SB3 world-model PPO evidence; never train or touch devices."""
import argparse
import csv
import hashlib
import json
import math
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


def runtime_section(rows, summary):
    text=['## 控制周期与安全回退','',
        'policy p50/p95是episode内分位数的平均；完整cycle为全部实际周期汇集后的pooled分位数，包含编码、policy、安全检查和仿真step，不含设备IO。训练时生成模型分支，评价在线不执行世界模型。','',
        '|方法|分布|回退总次数|policy均值p50/p95 ms|cycle pooled p50/p95 ms|pooled超100ms率|',
        '|---|---|---:|---|---|---:|']
    for item in summary:
        subset=[r for r in rows if r['method']==item['method'] and r['condition']==item['condition']]
        cycle=[x for r in subset for x in r.get('cycle_ms_samples',[])]
        if not subset or not cycle:continue
        pp50=np.mean([r['policy_p50_ms'] for r in subset]);pp95=np.mean([r['policy_p95_ms'] for r in subset])
        text.append(f"|{item['method']}|{item['condition']}|{sum(r['fallback_count'] for r in subset)}|{pp50:.3f}/{pp95:.3f}|{np.quantile(cycle,.5):.3f}/{np.quantile(cycle,.95):.3f}|{np.mean(np.asarray(cycle)>100):.6f}|")
    return text+['']


def findings(summary, comparisons, ledgers):
    """Interpret fixed contrasts jointly; never choose a winning seed or method."""
    lookup={(r['method'],r['condition']):r for r in summary}
    text=['## 固定主比较的科学结论','',
        '工程结论：策略和价值函数由官方PPO实际训练，世界模型仅在训练时提供当轮behavior策略短分支；部署评价执行学到的policy。模型辅助PPO已实现并闭环运行，与世界模型优势是否成立分别判定。','']
    for condition in ('id','ood'):
        wm=lookup['wm_ppo',condition];base=lookup['ppo',condition]
        effect=next(r for r in comparisons['ppo'] if r['condition']==condition and r['metric']=='return')
        text.append(f"{condition.upper()}：wm_ppo碰撞{wm['collisions']}/{wm['episodes']}，纯PPO {base['collisions']}/{base['episodes']}；raw return {wm['return']:.2f} vs {base['return']:.2f}，差{effect['difference']:+.2f}，95%区间[{effect['ci_low']:.2f},{effect['ci_high']:.2f}]。同时报告进度{wm['progress_m']:.2f} vs {base['progress_m']:.2f}m与jerk {wm['jerk_rms_mps3']:.3f} vs {base['jerk_rms_mps3']:.3f}m/s³。")
    text+=['','这些样本均值显示碰撞减少，但纯PPO主比较的回报区间仍跨零，训练随机性使稳定优势证据不足；不把较少碰撞直接等同全部驾驶指标改善。','']
    for reference in ('physics_ppo','ppo_total_budget','real_reuse_control'):
        for condition in ('id','ood'):
            e=next(r for r in comparisons[reference] if r['condition']==condition and r['metric']=='return')
            c=next(r for r in comparisons[reference] if r['condition']==condition and r['metric']=='collision')
            text.append(f"相对{reference}，{condition.upper()} return差{e['difference']:+.2f} [{e['ci_low']:.2f},{e['ci_high']:.2f}]；碰撞率差{c['difference']*100:+.2f}个百分点 [{c['ci_low']*100:.2f},{c['ci_high']*100:.2f}]。")
    text+=['','本轮对丰富物理辅助PPO的OOD回报与碰撞差异得到这些配对区间支持，仍限于当前场景、三个训练seed与预算。对同总真实交互PPO、额外真实经验复用的return差异区间跨零，不能声称世界模型稳定胜过两种成本控制。后两者分别使用额外online交互或额外梯度，均已单列账本。', '',
        'ID jerk相对纯PPO增加，相关差值区间见上表；平顺性是实际代价。无历史和更长分支不能只挑某个有利seed解释，全部方法/seed表和消融差异一起保留；h3不是自动优于h1。','']
    costs={method:{key:float(np.mean([r[key] for r in ledgers if r['method']==method])) for key in
        ('new_real_steps','native_total_training_interactions','policy_updates','training_wall_s','model_generation_wall_s','worldmodel_pretrain_wall_s')}
        for method in ('ppo','wm_ppo','physics_ppo','ppo_total_budget','real_reuse_control')}
    wm=costs['wm_ppo'];base=costs['ppo'];reuse=costs['real_reuse_control']
    text.append(f"每seed原生真实数据消耗：wm_ppo {wm['native_total_training_interactions']:.0f}、纯PPO {base['native_total_training_interactions']:.0f}；两者online步与optimizer更新相同（{wm['new_real_steps']:.0f}、{wm['policy_updates']:.0f}）。共享主机平均训练墙钟{wm['training_wall_s']:.2f}s vs {base['training_wall_s']:.2f}s，WM增加{(wm['training_wall_s']/base['training_wall_s']-1)*100:.1f}%；其中模型生成{wm['model_generation_wall_s']:.2f}s。额外更新控制为{reuse['training_wall_s']:.2f}s。预训练整套8模型上界另为{wm['worldmodel_pretrain_wall_s']:.2f}s，不算进这些policy训练墙钟，不是单模型成本。")
    text+=['','因此，本轮支持有限的条件收益与跨seed稳定性线索，尚不支持“世界模型普遍改善PPO”“在相同总数据/墙钟成本下必然更优”，也不支持“世界模型无用”。同online/同更新数比较排除了多做policy更新这一解释；额外offline数据、生成计算、控制更平缓/激烈的取舍和单seed退化仍需共同计入。', '',
        '样本效率边界：这些固定预算最终策略差异不能单独证明达到同一驾驶质量所需真实数据更少。验证曲线只有预登记的稀疏监控点，早期学习并非WM一致领先；纯PPO原生无需offline数据。须同时看新增online、native总消耗及墙钟，不能仅按最终return给样本效率结论。','']
    return text


def reference_section(root, artifacts, main_summary, report_parent):
    """A separately registered larger-budget reference, never an equal-budget win."""
    reference=root.parent/'reference_204800'
    audit=read(reference/'completeness.json',{})
    text=['## 补充：固定204800步纯PPO参考（不同预算）','',
        '依据65k验证seed1学习不足而事前固定延长，默认PPO超参数保持；三个seed全部跑完204800步，不选中间最好checkpoint，不使用test决定预算或超参数。此补充不替代前述65k七方法预登记主比较，也不是等预算的模型胜负试验。','']
    if not audit.get('verified_complete'):
        return text+['补充reference仍pending，暂不作数据效率或基线可靠性结论。','']
    ledgers=read(reference/'cost_ledger.json',[])
    summary=read(reference/'summary.json',{}).get('summary',[])
    curve=[];mismatches=[]
    for r in ledgers:
        seed=r['training_seed'];path=reference/f'seed{seed}'/'ppo'
        if r['new_real_steps']!=204800 or r['policy_updates']!=32000 or r['synthetic_generated']!=0:
            mismatches.append(seed)
        if hashlib.sha256((path/'test/episodes.jsonl').read_bytes()).hexdigest()!=r['test_sha256']:
            mismatches.append(seed)
        curve.extend(read(path/'learning_curve.json',[]))
    audit=dict(audit,budget_or_digest_mismatches=mismatches)
    (artifacts/'reference_audit.json').write_text(json.dumps(audit,indent=2))
    write_csv(artifacts/'reference_cost_table.csv',ledgers)
    write_csv(artifacts/'reference_learning_curves.csv',curve)
    if mismatches:
        return text+['补充reference预算/摘要审计未通过，不作比较结论。','']
    text+=[f"补充完整性：{audit['completed_jobs']}/{audit['expected_jobs']}任务、{audit['actual_episodes']}/{audit['expected_episodes']}唯一测试回合。",'',
        '|seed|真实步|完成率(val)|碰撞率(val)|raw return(val)|联合门槛|训练墙钟s|更新数|',
        '|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in ledgers:
        last=max((v for v in curve if v['training_seed']==r['training_seed']),key=lambda v:v['new_real_steps'])
        passed=last['completion_rate']>=.9 and last['collision_rate']<=.1 and last['mean_return']>=30
        text.append(f"|{r['training_seed']}|{r['new_real_steps']}|{last['completion_rate']:.3f}|{last['collision_rate']:.3f}|{last['mean_return']:.2f}|{'通过' if passed else '未通过'}|{r['training_wall_s']:.2f}|{r['policy_updates']}|")
    text+=['','固定ID验证门槛为完成≥.9、碰撞≤.1、return≥30；seed0/2通过，seed1仍未通过，不能用两个较好seed代替全部。每seed仅8个ID val场景，通过也不是普遍可靠性保证。所有四个监控点保留在reference_learning_curves.csv和下图，不择优。','',
        '|策略/预算|分布|回合|碰撞|完成|return|进度m|gap RMSE m|jerk m/s³|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,items in (('纯PPO 65536',[r for r in main_summary if r['method']=='ppo']),
                        ('WM-PPO 65536',[r for r in main_summary if r['method']=='wm_ppo']),('纯PPO 204800',summary)):
        for r in items:
            text.append(f"|{label}|{r['condition']}|{r['episodes']}|{r['collisions']}|{r['completions']}|{r['return']:.2f}|{r['progress_m']:.2f}|{r['gap_rmse_m']:.3f}|{r['jerk_rms_mps3']:.3f}|")
    lookup={(r['method'],r['condition']):r for r in main_summary}
    extended={r['condition']:r for r in summary}
    for condition in ('id','ood'):
        base=lookup['ppo',condition];wm=lookup['wm_ppo',condition];large=extended[condition]
        text.append(f"{condition.upper()}描述：纯PPO增加采样后，return {base['return']:.2f}→{large['return']:.2f}，碰撞{base['collisions']}→{large['collisions']}；相对WM65k的return样本均值差由{wm['return']-base['return']:.2f}变为{wm['return']-large['return']:.2f}。该差值只描述不同预算结果，不作为新的等预算显著性检验或模型节省数据证明。")
    wall=np.mean([r['training_wall_s'] for r in ledgers])
    text+=['',f'204800相当于65k在线交互的3.125倍、每seed32000次optimizer步（65k为10240），纯PPO原生offline0；平均训练墙钟{wall:.2f}s。WM65k另需6775个共同offline转移、模型预训练和生成计算。ID差距随更多纯PPO采样缩小，OOD未呈一致改善，seed1依然不足；这限定了稳定数据效率结论，不能据最终均值断言WM只需较少总成本就必然达到同等质量。','']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(15,4))
    for seed in sorted({r['training_seed'] for r in curve}):
        values=sorted((r for r in curve if r['training_seed']==seed),key=lambda r:r['new_real_steps'])
        for ax,key in zip(axes,('mean_return','completion_rate','collision_rate')):
            ax.plot([r['new_real_steps'] for r in values],[r[key] for r in values],marker='o',label=f'seed{seed}')
    for ax,label in zip(axes,('Raw validation return','Validation completion rate','Validation collision rate')):
        ax.set(xlabel='Pure-PPO online real steps',ylabel=label)
        ax.set_xticks([51200,102400,153600,204800],['51.2k','102.4k','153.6k','204.8k'])
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].axhline(30,linestyle='--',color='gray',linewidth=.7)
    axes[1].axhline(.9,linestyle='--',color='gray',linewidth=.7)
    axes[2].axhline(.1,linestyle='--',color='gray',linewidth=.7)
    figure.tight_layout();figure.savefig(artifacts/'reference_validation.png',dpi=150);plt.close(figure)
    import os
    relative=os.path.relpath(artifacts/'reference_validation.png',report_parent).replace('\\','/')
    text+=[f'![固定更长预算的全部seed验证轨迹]({relative})','']
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/rl_sb3/standard'))
    p.add_argument('--output',type=Path,default=Path('docs/SB3_RL_REPORT.md'))
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
    budget_mismatches=[]
    for r in ledgers:
        expected_real=registration['spec']['steps']+(r['common_offline_transitions'] if r['method']=='ppo_total_budget' else 0)
        epochs=20 if r['method']=='real_reuse_control' else 10
        full,partial=divmod(expected_real,2048)
        expected_updates=full*32*epochs+(math.ceil(partial/64)*epochs if partial else 0)
        if r['new_real_steps']!=expected_real or r['policy_updates']!=expected_updates:
            budget_mismatches.append(dict(method=r['method'],training_seed=r['training_seed'],
                expected_real_steps=expected_real,actual_real_steps=r['new_real_steps'],
                expected_optimizer_steps=expected_updates,actual_optimizer_steps=r['policy_updates']))
    audit['budget_mismatches']=budget_mismatches
    audit['verified_complete']=audit['verified_complete'] and not budget_mismatches
    (artifacts/'completeness.json').write_text(json.dumps(audit,indent=2))
    diagnosis_paths=[Path('docs/SB3_RL_PROTOCOL.md'), Path('docs/SB3_MODEL_DIAGNOSIS.md'), Path('docs/PPO_VALIDATION_DIAGNOSIS.md'),
        Path('outputs/rl_ppo_budget_validation/seed0_131072/DIAGNOSIS.md')]
    (artifacts/'diagnosis_sources.json').write_text(json.dumps({str(p):
        dict(exists=p.exists(),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)
        for p in diagnosis_paths},indent=2))
    write_csv(artifacts/'cost_table.csv',ledgers)
    (artifacts/'policy_configuration.json').write_text(json.dumps([
        dict(method=r['method'],training_seed=r['training_seed'],policy_config=r.get('policy_config'),
             model_metadata=r.get('model_metadata')) for r in ledgers],indent=2))
    write_csv(artifacts/'model_usage.csv',[dict(method=r['method'],training_seed=r['training_seed'],
        generated=r.get('synthetic_generated'),replaced=r.get('synthetic_replaced'),
        **r.get('model_metadata',{})) for r in ledgers])
    write_csv(artifacts/'learning_curves.csv',curves)
    summary=[];comparisons={}
    if rows:
        summary,main_effects=summarize(rows,artifacts,reference='ppo' if audit['verified_complete'] else '__partial_no_inference__')
        if audit['verified_complete']:
            for reference in ('ppo','physics_ppo','ppo_total_budget','real_reuse_control','wm_no_history','wm_long'):
                if reference in methods:
                    if reference=='ppo':
                        effects=main_effects
                    else:
                        paired_rows=[r for r in rows if r['method'] in ('wm_ppo',reference)]
                        _,effects=summarize(paired_rows,artifacts/f'against_{reference}',reference=reference)
                    comparisons[reference]=[r for r in effects if r['method']=='wm_ppo' and r['family']=='all']
    text=['# SB3世界模型辅助PPO：独立强基线对照','',
        f"运行目录：`{root}`；preset={preset}；算法为Stable-Baselines3官方默认PPO，模型分支辅助当轮on-policy更新，不能称为原版MBPO。",'',
        f"训练学习奖励缩放learning_reward_scale={registration.get('learning_reward_scale','历史运行未登记（未缩放版本）')}。评估始终使用未缩放的环境原始reward/return，不能将学习目标的缩放误读为评价指标变更。",'',
        f"完整性：{'已验证齐全' if audit['verified_complete'] else '进行中／审计未通过，不下优势结论'}；任务{len(ledgers)}/{audit['expected_jobs']}，episode {len(rows)}/{len(expected_keys)}，缺失{audit['missing_episode_count']}，重复{audit['duplicate_episode_count']}，真实步数/实际optimizer更新数预算异常{len(budget_mismatches)}。",'',
        '固定预算、最终checkpoint测试，val仅监控、不挑test-best。只运行本地仿真，没有R550连接、上线或部署。', '',
        '## 新轮依据与交付边界','',
        '旧自研PPO32k首轮与131k预算诊断基线质量不足，全部保留在outputs/rl_ppo及rl_ppo_budget_validation，旧报告docs/RL_REPORT.md不被覆盖。仅train/val的独立SB3官方默认交叉检查在51200首个验证点8/8完成、0碰撞、raw return58.78，这只是配置与实现证据，不是新三seed正式测试，也不能从单点推出世界模型优势。', '',
        '新轮standard每method/seed65536真实online步、3seed、7方法；smoke8192只检验工程。实际新PPO基线质量必须结合本轮最终policy与验证曲线核验，不能仅凭换用SB3便假定所有种子已达到可用驾驶水平。完整性通过只说明登记实验齐全，不代表整个目标或世界模型优势已经成立。', '',
        '学习reward统一乘0.01，评价保持同一raw reward。主PPO结构、entropy、优化器超参数使用本机SB3默认；只有real_reuse_control的20epochs是登记的额外计算控制。不存在按新test表现选择缩放、horizon、entropy或checkpoint。', '',
        '## 实验约定','',
        'ppo使用真实在线rollout；wm_ppo使用history模型一步分支，在super().train()前替换一半当轮真实buffer学习样本，不增加总梯度次数；physics_ppo使用丰富辨识物理同样分支；wm_no_history和wm_long检验历史与分支长度；real_reuse_control使用额外2倍更新，作为独立计算控制。ppo_total_budget把共同offline转移数改花为额外online步，使其原生总真实训练交互与wm_ppo相同，允许更多更新且单独记账。离线数据仅拟合预测模型，不进入PPO actor/critic梯度。真实与模型轨迹分别GAE，分支old_log_prob来自更新前同一冻结behavior策略。', '',
        f"主比较每method/seed新增online预算为{registration['spec']['steps']}真实步，rollout2048，batch64，10epochs，每完整rollout实际320个optimizer步（real_reuse_control20epochs为640）。ppo_total_budget另加offline对应online步数，末个partial rollout精确补齐请求的真实交互预算，不向上补成2048倍数；尾段与optimizer hook实测更新数按实际账本报告，不能归入同online/同更新数主比较。", '',
        '## 最终策略闭环','',
        '|方法|分布|回合|碰撞|完成|return|进度m|gap RMSE m|jerk RMS m/s³|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append(f"|{r['method']}|{r['condition']}|{r['episodes']}|{r['collisions']}|{r['completions']}|{r['return']:.3f}|{r['progress_m']:.3f}|{r['gap_rmse_m']:.3f}|{r['jerk_rms_mps3']:.3f}|")
    seed_rows=[]
    for method in methods:
        for condition in ('id','ood'):
            for seed in seeds:
                subset=[r for r in rows if r['method']==method and r['condition']==condition and r['training_seed']==seed]
                if subset:seed_rows.append(dict(method=method,condition=condition,training_seed=seed,episodes=len(subset),
                    collisions=sum(r['collision'] for r in subset),completed=sum(r['completed'] for r in subset),
                    return_mean=float(np.mean([r['return'] for r in subset])),progress_m=float(np.mean([r['progress_m'] for r in subset]))))
    write_csv(artifacts/'per_seed.csv',seed_rows)
    text+=['','## 全部方法的训练seed差异','',
        '|方法|分布|seed|回合|碰撞|完成|return|进度m|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in seed_rows:
        text.append(f"|{r['method']}|{r['condition']}|{r['training_seed']}|{r['episodes']}|{r['collisions']}|{r['completed']}|{r['return_mean']:.2f}|{r['progress_m']:.2f}|")
    text+=['',('训练随机性不能被总体均值掩盖：主wm_ppo在seed0的OOD return低于纯PPO、碰撞更多，seed0/2的ID return也较低；总体收益很大部分来自减少纯PPO seed1训练失败。保留所有seed，不用较好seed替代较差seed。' if audit['verified_complete'] and preset=='standard' else '逐seed保留，不用较好seed替代较差seed；未齐全时不做训练随机性结论。'), '',
        '## 场景族条件','',
        '每格为碰撞数/回合数；raw return。全7方法的完整各族指标保留在family_summary.csv。','',
        '|分布|场景族|ppo|wm_ppo|physics_ppo|real_reuse_control|','|---|---|---|---|---|---|']
    for condition in ('id','ood'):
        for family in ('constant','mild','brake','stopgo'):
            cells=[]
            for method in ('ppo','wm_ppo','physics_ppo','real_reuse_control'):
                subset=[r for r in rows if r['method']==method and r['condition']==condition and r['family']==family]
                cells.append(f"{sum(r['collision'] for r in subset)}/{len(subset)}; {np.mean([r['return'] for r in subset]):.2f}" if subset else '未齐全')
            text.append('|'+condition+'|'+family+'|'+'|'.join(cells)+'|')
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
        text+=['','smoke只有1个训练seed，区间仅反映场景抽样，不构成训练随机性的优势证据。standard只有3seed，区间仍不稳定。零事件或恒定配对差异的经验bootstrap可能给[0,0]，它不表示真实风险为零。区间未经多重比较校正；主wm对ppo/physics按预登记解释，控制与消融/分族视为条件和探索性证据，不按显著性挑赢家。须同时检查碰撞、完成、进度和回报；区间跨零不证明等效。']
        if preset=='standard' and all(k in comparisons for k in ('ppo','physics_ppo','ppo_total_budget','real_reuse_control')):
            text+=findings(summary,comparisons,ledgers)
    text+=['']+runtime_section(rows,summary)
    text+=['','## 成本与样本效率','',
        'worldmodel_pretrain_wall_s是整套8种预测模型预训练墙钟上界，给物理与神经模型同样标记；不能称单个history/physics模型的训练成本。主训练墙钟另计model_generation_wall_s，验证/测试真实步单列，不混入训练预算。', '',
        'policy_updates为optimizer post-step hook实测梯度步数，不是SB3按epoch累计的_n_updates。policy_configuration.json保存各方法/seed实际超参数，model_usage.csv保存模型生成数与实际替换数，避免把h3生成量当成实际学习样本数。', '',
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
            'val固定ID、每族2case共8个独立场景；四分之一预算监控，不据此挑test-best。左图新增online，中图原生实际消耗的offline+online真实交互；纯PPO、real_reuse_control和ppo_total_budget均原生零offline。右图仅共同预算记账，不能当作真实数据消耗；原生成本推算采用该方法是否使用预测模型与已记录offline转移数。细淡线为独立训练seed，带标记粗线为同一步数跨seed均值，不是置信区间；不能把曲线点当独立训练重复。', '',f'![样本效率]({relative})','']
    if preset=='standard':text+=reference_section(root,artifacts,summary,args.output.parent)
    text+=['## 当前模型池的能力归因','',
        '独立纯验证诊断见[SB3_MODEL_DIAGNOSIS.md](SB3_MODEL_DIAGNOSIS.md)及当前root/model_diagnosis；使用当前common_data的12个独立val episode和3个冻结world_models，未读取test选择模型或策略。三个模型seed共享val场景，不当成36个独立episode；这不是旧MPC数据集。', '',
        '当前history在匀速窗口的前车速度MAE：0.1/0.5/1/3秒分别0.00338526/0.0682029/0.306833/2.55033m/s；3秒gap MAE2.58214m，而丰富物理在该族近零。四场景族3秒gap误差均history更大。因此不能把PPO闭环差异归因成“递归预测更准”。', '',
        'history与仅学习前车的learned_lead在全部登记窗口的gap/ego/leader预测最大差为0，自车执行器差最多2.78e-17m/s²；自车残差网络raw输出最大4.84e-18m/s²。本数据的自车确定性结构已由丰富物理拟合充分解决，不支持学习了有意义的额外自车响应这一解释。', '',
        '同一val根、同一实际动作的h1 analytic raw reward跨模型最大差严格0：显式Euler下一gap/ego由旧状态决定。学习前车可以改变next_obs和critic bootstrap，h3还会改变后续奖励。短模型GAE目标、分支根重采样及bootstrap扰动是可能解释（推断），并未证明为收益来源；物理辅助对照与nohistory/h3消融限制这种归因，但不能完全隔离机制。', '',
        '## 证据链与失效边界','',
        '现有证据链是：共同数据拟合的前车预测/近零自车残差→h1 next_obs与bootstrap变化→学习曲线和部分seed失败率改变→有限条件闭环差异→额外数据/训练成本；没有证明“更准多步预测必然带来更好PPO”。', '',
        '预测误差降低不保证PPO改善；模型偏差可能被策略放大。OOD、不可预知急刹、有限数据和短分支覆盖仍是边界。零碰撞不构成通用安全保证；主机周期包含仿真step，不代表Pi性能。全部原始日志、成本、按family指标和配对区间在运行目录保留。', '',
        '本轮正式报告必须来自独立outputs/rl_sb3/standard且完整性通过；smoke不替代standard，旧自研PPO报告保留历史，不混用checkpoint或按胜负选择轮次。','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(text),encoding='utf8')
    print(json.dumps(dict(report=str(args.output),audit=audit),ensure_ascii=False))


if __name__=='__main__':main()
