"""Frozen paired-episode RL evaluation; no training, device access or test selection."""
from pathlib import Path
import csv
import hashlib
import json
from time import perf_counter
import numpy as np

FAMILIES = ('constant', 'mild', 'brake', 'stopgo')
METRICS = ('return', 'collision', 'completed', 'progress_m', 'gap_rmse_m',
           'jerk_rms_mps3', 'fallback_count', 'cycle_p95_ms')


def evaluation_cases(training_seed, preset='smoke', split='test'):
    """Disjoint episode seeds across training seeds, families, conditions and splits."""
    if preset not in ('smoke', 'standard') or split not in ('val', 'test'):
        raise ValueError('Expected smoke/standard and val/test')
    if not isinstance(training_seed, (int, np.integer)) or not 0 <= training_seed < 10:
        raise ValueError('Registered training seeds are integers in [0,9]')
    base = 8_000_000 if split == 'test' else 7_000_000
    count = 2 if preset == 'smoke' else 10
    return [dict(seed=base+training_seed*100_000+ci*10_000+fi*1000+i,
                 family=family, condition=condition)
            for ci, condition in enumerate(('id', 'ood'))
            for fi, family in enumerate(FAMILIES) for i in range(count)]


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r if k != 'cycle_ms_samples'))
    with Path(path).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(agent, eval_seeds, output, *, method='policy', training_seed=0,
                    env_factory=None, max_steps=120):
    """Greedy frozen policy on explicit [{seed,family,condition}] paired cases.

    Agent.act receives only fixed-scale 25D deployment observation. Timing covers
    encode+policy+safety+simulator.step; policy timing excludes the simulator.
    Output is immutable: choose a new directory for each checkpoint/split.
    """
    from .rl_env import LongitudinalRLEnv, decode_observation, encode, REWARD_VERSION
    if max_steps != 120:
        raise ValueError('Registered RL evaluation horizon is 120 steps')
    cases = [dict(c) for c in eval_seeds]
    identities = [(c['family'], c['condition'], int(c['seed'])) for c in cases]
    if not cases or len(set(identities)) != len(cases):
        raise ValueError('Evaluation cases must be nonempty and unique')
    if any(f not in FAMILIES or c not in ('id', 'ood') for f,c,s in identities):
        raise ValueError('Unknown evaluation family or condition')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Evaluation output must be empty; preserve previous evidence')
    manifest = dict(method=method, training_seed=int(training_seed), cases=cases,
                    max_steps=max_steps, reward_version=REWARD_VERSION,
                    policy='final checkpoint, greedy explore=False; no test selection',
                    observation='privileged simulator six [gap,ego,lead,actuator] frames plus previous applied u',
                    timing='cycle includes encoding, policy, safety and truth simulator step; no device IO')
    manifest['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('rl_env.py', 'rl_evaluation.py')}
    _json(out/'manifest.json', manifest)
    rows = []
    for case in cases:
        env = (env_factory or LongitudinalRLEnv)()
        raw, reset_info = env.reset(**case)
        history, previous_u = decode_observation(raw)
        accel_previous = (history[-1,1]-history[-2,1])/env.dt
        returns = progress = reference = 0.
        gaps, jerks, cycles, policies, reasons, trace = [], [], [], [], [], []
        terminated = truncated = False
        for step in range(max_steps):
            begin = perf_counter()
            history, previous_u = decode_observation(raw)
            encoded = encode(history, previous_u)
            index = agent.act(encoded, explore=False)
            policy_ms = (perf_counter()-begin)*1000
            nxt, reward, terminated, truncated, info = env.step(index)
            cycle_ms = (perf_counter()-begin)*1000
            future_history, next_u = decode_observation(nxt)
            old, new = history[-1], future_history[-1]
            acceleration = float((new[1]-old[1])/env.dt)
            jerk = float((acceleration-accel_previous)/env.dt)
            gap_error = float(new[0]-(4+1.2*new[1]))
            if not np.isfinite(reward):
                raise ValueError('Nonfinite evaluation reward')
            returns += float(reward)
            progress += float(old[1]*env.dt)
            reference += float(old[2]*env.dt)
            gaps.append(gap_error)
            jerks.append(jerk)
            cycles.append(cycle_ms)
            policies.append(policy_ms)
            reasons.append(str(info.get('fallback') or ''))
            trace.append(dict(step=step, time_s=(step+1)*env.dt,
                gap_m=float(new[0]), speed_mps=float(new[1]), lead_speed_mps=float(new[2]),
                action_index=int(index), applied_command_mps2=float(next_u), reward=float(reward),
                measured_accel_mps2=acceleration, jerk_mps3=jerk, gap_error_m=gap_error,
                fallback=reasons[-1], policy_ms=policy_ms, cycle_ms=cycle_ms,
                terminated=bool(terminated), truncated=bool(truncated)))
            raw, accel_previous = nxt, acceleration
            if terminated or truncated:
                break
        collision = bool(terminated or future_history[-1,0] <= 0)
        completed = bool(not collision and len(trace)==max_steps and progress >= .5*max(5., reference))
        row = dict(method=method, training_seed=int(training_seed), family=case['family'],
            condition=case['condition'], scenario_seed=int(case['seed']), steps=len(trace),
            return_value=float(returns), collision=int(collision), completed=int(completed),
            progress_m=progress, reference_progress_m=reference,
            gap_rmse_m=float(np.sqrt(np.mean(np.square(gaps)))),
            jerk_rms_mps3=float(np.sqrt(np.mean(np.square(jerks)))),
            fallback_count=sum(bool(r) for r in reasons),
            fallback_reasons=json.dumps({r:reasons.count(r) for r in sorted(set(reasons)) if r}),
            policy_p50_ms=float(np.quantile(policies,.5)), policy_p95_ms=float(np.quantile(policies,.95)),
            cycle_p50_ms=float(np.quantile(cycles,.5)), cycle_p95_ms=float(np.quantile(cycles,.95)),
            timeout_rate=float(np.mean(np.asarray(cycles)>env.dt*1000)), cycle_ms_samples=cycles)
        row['return'] = row.pop('return_value')
        _csv(out/f"trace_{case['condition']}_{case['family']}_{case['seed']}.csv", trace)
        with (out/'episodes.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        rows.append(row)
    _csv(out/'episodes.csv', rows)
    _json(out/'completeness.json', dict(expected=len(cases), actual=len(rows), unique=len(identities), verified_complete=True))
    return rows


def summarize(rows, output, *, reference='ppo', bootstrap_samples=2000):
    """Episode metrics and paired hierarchical bootstrap; frames never resampled."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    identities=[(r['method'],r['training_seed'],r['family'],r['condition'],r['scenario_seed']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate RL episode identities')
    groups={}
    for r in rows:
        groups.setdefault((r['method'],r['condition']),[]).append(r)
    summaries=[]
    for (method,condition), items in sorted(groups.items()):
        summary=dict(method=method,condition=condition,episodes=len(items),
            training_seeds=len({r['training_seed'] for r in items}),
            collisions=sum(r['collision'] for r in items), completions=sum(r['completed'] for r in items))
        summary.update({metric:float(np.mean([r[metric] for r in items])) for metric in METRICS})
        cycles=[x for r in items for x in r.get('cycle_ms_samples',[])]
        if cycles:
            summary.update(pooled_cycle_p50_ms=float(np.quantile(cycles,.5)),
                           pooled_cycle_p95_ms=float(np.quantile(cycles,.95)))
        summaries.append(summary)
    family_summaries=[]
    for (method,condition), items in sorted(groups.items()):
        for family in FAMILIES:
            subset=[r for r in items if r['family']==family]
            if not subset:continue
            item=dict(method=method,condition=condition,family=family,episodes=len(subset),
                training_seeds=len({r['training_seed'] for r in subset}),
                collisions=sum(r['collision'] for r in subset),completions=sum(r['completed'] for r in subset))
            item.update({metric:float(np.mean([r[metric] for r in subset])) for metric in METRICS})
            family_summaries.append(item)
    effects=[]
    for (method,condition), items in sorted(groups.items()):
        if method==reference or (reference,condition) not in groups:
            continue
        key=lambda r:(r['training_seed'],r['family'],r['scenario_seed'])
        baseline={key(r):r for r in groups[(reference,condition)]}
        if set(map(key,items)) != set(baseline):
            raise ValueError('Incomplete paired RL comparison')
        for family in ('all',*FAMILIES):
            pairs=[(r,baseline[key(r)]) for r in items if family=='all' or r['family']==family]
            if not pairs:continue
            for metric in METRICS:
                strata={}
                for a,b in pairs:
                    strata.setdefault((a['training_seed'],a['family']),[]).append(a[metric]-b[metric])
                seeds=sorted({s for s,f in strata})
                rng=np.random.default_rng(9401)
                boot=[]
                for _ in range(bootstrap_samples):
                    draws=[]
                    for s in rng.choice(seeds,len(seeds),replace=True):
                        for (ss,f),values in strata.items():
                            if ss==s:draws.append(float(np.mean(rng.choice(values,len(values),replace=True))))
                    boot.append(float(np.mean(draws)))
                effects.append(dict(method=method,reference=reference,condition=condition,family=family,
                    metric=metric,pairs=len(pairs),training_seeds=len(seeds),
                    difference=float(np.mean([a[metric]-b[metric] for a,b in pairs])),
                    ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                    unit='training seed then family-stratified paired episode bootstrap',
                    limitation='only one training seed; no training-randomness uncertainty' if len(seeds)==1 else 'three seeds give unstable training-randomness intervals'))
    _csv(output/'summary.csv',summaries)
    _csv(output/'family_summary.csv',family_summaries)
    _csv(output/'paired_effects.csv',effects)
    _json(output/'summary.json',dict(summary=summaries,family_summary=family_summaries,paired_effects=effects))
    return summaries,effects


def write_training_report(rows, output, *, ledgers=(), validation_rows=(), reference='ppo'):
    """Build an evidence report from final episodes and explicit runner actuals.

    Ledger entries: method, training_seed, actual (arbitrary cost dictionary).
    Curves: method, training_seed, online_steps, offline_transitions, val_return.
    Missing costs remain unverified; no configured budget is substituted.
    """
    output=Path(output)
    summaries,effects=summarize(rows,output,reference=reference)
    ledgers=list(ledgers)
    curves=list(validation_rows)
    _json(output/'cost_ledger.json',ledgers)
    _json(output/'validation_curves.json',curves)
    text=['# 世界模型辅助PPO实验报告', '',
          '仅本地仿真。测试使用固定最终贪心策略；验证曲线不能当测试结果。', '',
          '|方法|分布|回合|训练种子|碰撞|完成|回报|进度m|gap RMSE m|jerk RMS m/s³|',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f"|{r['method']}|{r['condition']}|{r['episodes']}|{r['training_seeds']}|{r['collisions']}|{r['completions']}|{r['return']:.3f}|{r['progress_m']:.3f}|{r['gap_rmse_m']:.3f}|{r['jerk_rms_mps3']:.3f}|")
    text+=['','## 配对效应与边界','',
           '差值为方法−对照；return/progress/completed越高越好，碰撞、误差、jerk越低越好。区间按训练seed→family内配对episode分层bootstrap，不使用相邻帧作为独立样本。', '',
           '|方法|对照|分布|指标|配对数|差值|95%区间|', '|---|---|---|---|---:|---:|---|']
    for r in effects:
        if r['family']=='all' and r['metric'] in ('return','collision','completed','progress_m','gap_rmse_m'):
            text.append(f"|{r['method']}|{reference}|{r['condition']}|{r['metric']}|{r['pairs']}|{r['difference']:.4f}|[{r['ci_low']:.4f},{r['ci_high']:.4f}]|")
    seeds={r['training_seed'] for r in rows}
    text+=['',f'训练种子数{len(seeds)}。单seed不提供训练随机性证据，3seed的分层区间仍不稳定。预测更好不自动代表PPO更好，零碰撞不构成通用安全保证。', '',
           '## 数据、更新与墙钟成本','',
           '下表直接列出runner的actual字段；缺项未验证，不用配置预算或buffer容量猜测。离线数据仅用于模型拟合，不作off-policy PPO actor训练。共同离线采集成本与纯PPO原生zero-offline成本必须同时解释。', '',
           '|方法|训练seed|实际成本字段|值|','|---|---:|---|---|']
    for ledger in ledgers:
        actual=ledger.get('actual')
        if not isinstance(actual,dict) or not actual:
            text.append(f"|{ledger.get('method','unknown')}|{ledger.get('training_seed','?')}|actual成本账本|未验证|")
            continue
        for k,v in sorted(actual.items()):
            rendered=json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v)
            text.append(f"|{ledger.get('method','unknown')}|{ledger.get('training_seed','?')}|{k}|{rendered.replace('|','/')}|")
    if not ledgers:text.append('|全部|?|实际成本账本|未提供，无法验证样本效率和总成本|')
    required={'method','training_seed','online_steps','offline_transitions','val_return'}
    valid=[r for r in curves if required.issubset(r)]
    if valid:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(1,2,figsize=(12,4))
        for (method,seed) in sorted({(r['method'],r['training_seed']) for r in valid}):
            values=sorted((r for r in valid if (r['method'],r['training_seed'])==(method,seed)),key=lambda r:r['online_steps'])
            y=[r['val_return'] for r in values]
            label=f'{method} seed{seed}'
            axes[0].plot([r['online_steps'] for r in values],y,label=label)
            axes[1].plot([r['online_steps']+r['offline_transitions'] for r in values],y,label=label)
        for ax,xlabel in zip(axes,('New online truth steps','Common offline + new online truth steps')):
            ax.set(xlabel=xlabel,ylabel='Validation episode return')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output/'sample_efficiency.png',dpi=150)
        plt.close(figure)
        text+=['','## 验证学习曲线','',
               '左图仅新增真实在线步数；右图计入共同离线真实转移。synthetic模型步数不计入real steps。纯PPO若无需离线数据，其原生总数据成本可低于共同数据预算坐标。', '',
               '![样本效率](sample_efficiency.png)']
    else:
        text+=['','未提供完整验证曲线字段，样本效率曲线未验证；不从最终测试结果虚构学习曲线。']
    text+=['','完整各family指标见family_summary.csv，完整效应见paired_effects.csv，原始控制周期和失败留在episode/trajectory文件。', '']
    (output/'REPORT.md').write_text('\n'.join(text),encoding='utf8')
    return output/'REPORT.md'
