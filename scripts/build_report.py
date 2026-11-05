"""Build evidence reports from frozen experiment logs; safe to rerun mid-benchmark."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('physical_cv', 'physical_ca', 'physical_enriched', 'neural',
          'residual', 'history', 'learned_lead', 'ensemble')


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def read_jsonl(path):
    """Ignore an interrupted final write, never silently ignore interior corruption."""
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                     ['|'+'|'.join(str(c) for c in row)+'|' for row in rows])


def f(value):
    return f'{float(value):.4g}'


def episode_mae(rows, metrics, seed_key='training_seed', split_key='split'):
    """Equal weight for episode means, not for windows from surviving episodes."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row.get(split_key, 'test'), float(row['horizon_s']),
                row.get(seed_key, '0'), row['episode'])].append(row)
    episodes = []
    for (model, split, horizon, seed, episode), windows in sorted(groups.items()):
        episodes.append(dict(model=model, split=split, horizon_s=horizon,
                             training_seed=seed, episode=episode, windows=len(windows),
                             **{metric: float(np.mean([float(r[metric]) for r in windows])) for metric in metrics}))
    means = defaultdict(list)
    for row in episodes:
        means[row['model'], row['split'], row['horizon_s']].append(row)
    summary = [dict(model=k[0], split=k[1], horizon_s=k[2], episodes=len(v),
                    training_seeds=len(set(r['training_seed'] for r in v)),
                    **{m: float(np.mean([r[m] for r in v])) for m in metrics})
               for k, v in sorted(means.items())]
    return episodes, summary


def longitudinal_audit(path, rows):
    config = read_json(path/'run_config.json', {})
    preset = config.get('preset', 'smoke')
    seeds = config.get('seeds', [])
    budgets = ('equal', 'time') if config.get('budget') == 'both' else (config.get('budget', 'equal'),)
    expected = set()
    missing_models = []
    for seed in seeds:
        models = list(MODELS)
        missing_models.extend(str(path/f'seed{seed}'/'models'/f'{name}.npz')
                              for name in models if not (path/f'seed{seed}'/'models'/f'{name}.npz').exists())
        for budget in budgets:
            names = models + [name+'_risk' for name in models if 'ensemble' in name]
            if budget == 'equal':
                names += ['acc']
            for ci, family in enumerate(('constant', 'mild', 'brake', 'stopgo')):
                for pair in range(dict(smoke=2, standard=10, full=60)[preset]):
                    scenario = 900000+ci*1000+pair+(seed*10000 if preset == 'standard' else 0)
                    for condition in ('id', 'ood'):
                        for name in names:
                            expected.add((name, condition, family, seed, scenario, budget))
    actual = [(r['model'], r['condition'], r['family'], int(r['training_seed']),
               int(r['scenario_seed']), r['budget']) for r in rows]
    unique = set(actual)
    complete = bool(expected) and unique == expected and len(actual) == len(unique) and not missing_models
    return dict(status='verified_complete' if complete else 'provisional_incomplete',
                expected=len(expected), actual=len(actual), unique=len(unique),
                missing=len(expected-unique), unexpected=len(unique-expected), duplicates=len(actual)-len(unique),
                missing_model_artifacts=missing_models)


def closed_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['condition'], row['budget']].append(row)
    return [dict(model=k[0], condition=k[1], budget=k[2], episodes=len(v),
                 collisions=sum(int(r['collision']) for r in v),
                 completion=float(np.mean([r['completed'] for r in v])),
                 **{m: float(np.mean([r[m] for r in v])) for m in
                    ('progress_m', 'gap_rmse_m', 'jerk_rms_mps3', 'fallback_count',
                     'inference_p50_ms', 'cycle_p95_ms', 'timeout_rate')}) for k, v in sorted(groups.items())]


def family_tables(run,rows,assets,complete):
    """All-method scenario strata; intervals resample seeds then paired episodes."""
    groups=defaultdict(list)
    for r in rows:groups[r['model'],r['family'],r['condition'],r['budget']].append(r)
    summaries=[];reason_rows=[]
    for key,items in sorted(groups.items()):
        cycles=[];inference=[];missing=0;reasons=Counter();reason_episodes=Counter()
        for item in items:
            path=trajectory_path(run,item)
            if not path.exists():missing+=1;continue
            episode_reason_set=set()
            for point in read_csv(path):
                cycles.append(float(point['cycle_s'])*1000)
                inference.append(float(point['inference_s'])*1000)
                if point['fallback']:
                    reasons[point['fallback']]+=1;episode_reason_set.add(point['fallback'])
            reason_episodes.update(episode_reason_set)
        names={'emergency_gap','deadline_exceeded','solver_failure','model_or_solver_exception','invalid_observation'}|set(reasons)
        for reason in sorted(names):
            reason_rows.append(dict(model=key[0],family=key[1],condition=key[2],budget=key[3],reason=reason,
                trigger_count=reasons[reason],logged_cycles=len(cycles),
                trigger_rate_per_cycle=reasons[reason]/len(cycles) if cycles else None,
                episodes_triggered=reason_episodes[reason],episodes_with_logs=len(items)-missing,
                episode_trigger_rate=reason_episodes[reason]/(len(items)-missing) if len(items)>missing else None,
                status='complete' if complete and not missing else 'provisional'))
        row=dict(model=key[0],family=key[1],condition=key[2],budget=key[3],episodes=len(items),
                 training_seeds=len(set(r['training_seed'] for r in items)),
                 collisions=sum(r['collision'] for r in items),completion_rate=float(np.mean([r['completed'] for r in items])),
                 **{metric:float(np.mean([r[metric] for r in items])) for metric in
                    ('progress_m','gap_rmse_m','jerk_rms_mps3','fallback_count','cycle_p50_ms','cycle_p95_ms','timeout_rate')},
                 actual_logged_cycles=len(cycles),missing_trajectory_logs=missing,
                 pooled_cycle_p50_ms=float(np.quantile(cycles,.5)) if cycles else None,
                 pooled_cycle_p95_ms=float(np.quantile(cycles,.95)) if cycles else None,
                 pooled_inference_p50_ms=float(np.quantile(inference,.5)) if inference else None,
                 pooled_timeout_rate=float(np.mean(np.asarray(cycles)>100)) if cycles else None,
                 status='complete' if complete and not missing else 'provisional')
        summaries.append(row)
    write_csv(assets/f'{run.name}_closed_by_family.csv',summaries)
    write_csv(assets/f'{run.name}_fallback_by_family.csv',reason_rows)
    reason_groups=defaultdict(list)
    for r in reason_rows:reason_groups[r['model'],r['condition'],r['budget'],r['reason']].append(r)
    pooled_reasons=[]
    for k,v in sorted(reason_groups.items()):
        triggers=sum(r['trigger_count'] for r in v);cycles=sum(r['logged_cycles'] for r in v)
        episodes=sum(r['episodes_with_logs'] for r in v);triggered=sum(r['episodes_triggered'] for r in v)
        pooled_reasons.append(dict(model=k[0],condition=k[1],budget=k[2],reason=k[3],trigger_count=triggers,
            logged_cycles=cycles,trigger_rate_per_cycle=triggers/cycles if cycles else None,
            episodes_triggered=triggered,episodes_with_logs=episodes,
            episode_trigger_rate=triggered/episodes if episodes else None,
            status='complete' if complete and all(r['status']=='complete' for r in v) else 'provisional'))
    write_csv(assets/f'{run.name}_fallback_reasons.csv',pooled_reasons)
    effects=[];seed_effects=[]
    for (model,family,condition,budget),items in sorted(groups.items()):
        if model=='physical_enriched':continue
        reference=groups.get(('physical_enriched',family,condition,budget),[])
        lookup={(r['training_seed'],r['scenario_seed']):r for r in reference}
        pairs=[(r,lookup[r['training_seed'],r['scenario_seed']]) for r in items if (r['training_seed'],r['scenario_seed']) in lookup]
        if not pairs:continue
        for metric in ('collision','completed','progress_m','gap_rmse_m','jerk_rms_mps3','fallback_count','cycle_p95_ms'):
            seeds=sorted(set(a['training_seed'] for a,b in pairs))
            deltas={seed:np.asarray([a[metric]-b[metric] for a,b in pairs if a['training_seed']==seed]) for seed in seeds}
            rng=np.random.default_rng(20261104)
            choices=rng.choice(seeds,size=(2000,len(seeds)),replace=True)
            boot=np.zeros((2000,len(seeds)))
            for slot in range(len(seeds)):
                for seed in seeds:
                    mask=choices[:,slot]==seed;n=int(mask.sum());values=deltas[seed]
                    if n:boot[mask,slot]=rng.choice(values,size=(n,len(values)),replace=True).mean(axis=1)
            means=boot.mean(axis=1)
            base=dict(model=model,reference='physical_enriched',family=family,condition=condition,budget=budget,metric=metric)
            effects.append(dict(**base,paired_episodes=len(pairs),training_seeds=len(seeds),
                difference=float(np.mean([a[metric]-b[metric] for a,b in pairs])),
                ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),
                status='complete' if complete else 'provisional',unit='training seed then within-seed paired episode bootstrap'))
            seed_effects.extend(dict(**base,training_seed=seed,paired_episodes=len(values),difference=float(values.mean())) for seed,values in deltas.items())
    write_csv(assets/f'{run.name}_family_paired_effects.csv',effects)
    write_csv(assets/f'{run.name}_family_per_seed_effects.csv',seed_effects)
    lines=[f'## {run.name} 闭环按场景族分层', '',
           '所有方法按family×condition×budget的碰撞、完成、进度、gap、jerk、回退和真实逐周期pooled时延均保留。回合p95均值和pooled p95分别命名。效应为方法减physical_enriched；先重采样训练seed，再在seed内重采样配对episode，不把帧当独立样本。仅3个训练seed的训练不确定性区间仍有限。', '',
           f'[全部方法分族表](report_assets/{run.name}_closed_by_family.csv)；[分族配对效应和区间](report_assets/{run.name}_family_paired_effects.csv)；[逐训练seed效应](report_assets/{run.name}_family_per_seed_effects.csv)。', '']
    lines += [f'[回退原因计数/逐周期触发率/受影响回合比例](report_assets/{run.name}_fallback_reasons.csv)；[按场景族展开的回退归因](report_assets/{run.name}_fallback_by_family.csv)。', '',
              '`emergency_gap`表示独立间距安全监督触发；`deadline_exceeded`表示主机墙钟周期超期而强制制动；`solver_failure`和`model_or_solver_exception`分别表示无有效解与已捕获的模型/求解数值异常；`invalid_observation`表示观测校验失败。零计数也保留。总回退不能掩盖计算超期或模型失效；逐周期触发率与受影响回合比例采用各自分母，不可混用。', '']
    if not complete or any(r['missing_trajectory_logs'] for r in summaries) or run.name!='standard_verified':
        return lines+['当前仅存分族表；本轮未完成标准规模，不输出分族优势结论。', '']
    summaries_lookup={(r['model'],r['family'],r['condition'],r['budget']):r for r in summaries}
    by_effect={(r['model'],r['family'],r['condition'],r['budget'],r['metric']):r for r in effects}
    for family in ('constant','mild','brake','stopgo'):
        for condition in ('id','ood'):
            for budget in ('equal','time'):
                key=('history',family,condition,budget)
                if key not in summaries_lookup:continue
                h=summaries_lookup[key];p=summaries_lookup['physical_enriched',family,condition,budget]
                def effect(metric):
                    r=by_effect.get((*key,metric))
                    return f"{r['difference']:+.4g} [{r['ci_low']:+.4g},{r['ci_high']:+.4g}]" if r else '缺失配对'
                lines += [f"{family}/{condition}/{budget}：登记的history相对丰富物理，gap RMSE差{effect('gap_rmse_m')}m，进度差{effect('progress_m')}m，jerk差{effect('jerk_rms_mps3')}m/s³；碰撞{h['collisions']}/{h['episodes']} 对 {p['collisions']}/{p['episodes']}，完成率{h['completion_rate']:.3f} 对 {p['completion_rate']:.3f}，真实pooled周期p95 {h['pooled_cycle_p95_ms']:.3g} 对 {p['pooled_cycle_p95_ms']:.3g}ms。", '']
    lines += ['上述逐族描述同时给出好坏方向；不能把单个更低的gap误差自动称为总体驾驶改善。残差、仅学习前车、集成风险和经典控制等全部方法均在附表，不能只凭主方法摘录遗漏其他负结果。', '']
    return lines


def sensor_differences(rows):
    """Compare observation modes within the same model and initial scenario."""
    identity = ('model', 'family', 'condition', 'scenario_seed', 'training_seed')
    lookup = {}
    for row in rows:
        key = tuple(str(row[k]) for k in identity)+(row['observation_mode'],)
        if key in lookup:
            raise ValueError('Duplicate sensor observation experiment identity')
        lookup[key] = row
    metrics = ('collision', 'completed', 'progress_m', 'gap_rmse_m', 'jerk_rms_mps3',
               'fallback_count', 'cycle_p95_ms', 'timeout_rate')
    pairs = []
    for key, sensor in sorted(lookup.items()):
        if key[-1] != 'sensor' or key[:-1]+('privileged',) not in lookup:
            continue
        privileged = lookup[key[:-1]+('privileged',)]
        pairs.append({**{k: sensor[k] for k in identity},
                      **{m: float(sensor[m])-float(privileged[m]) for m in metrics}})
    groups = defaultdict(list)
    for row in pairs:
        groups[row['model'], row['condition']].append(row)
    summary = [dict(model=k[0], condition=k[1], paired_episodes=len(v),
                    **{m: float(np.mean([r[m] for r in v])) for m in metrics})
               for k,v in sorted(groups.items())]
    return pairs, summary


def build_sensor_closed(assets):
    path = ROOT/'outputs/sensor_closed_verified'
    rows = read_csv(path/'metrics.csv')
    if not rows:
        return ['## 传感观测闭环敏感性', '', '本轮传感闭环文件尚未生成，未据开环消融推断实车闭环能力。', '']
    pairs, summary = sensor_differences(rows)
    write_csv(assets/'sensor_closed_paired_differences.csv', pairs)
    write_csv(assets/'sensor_closed_paired_summary.csv', summary)
    audit = read_json(path/'completeness.json', {})
    return ['## 传感观测闭环敏感性（smoke）', '',
            f'原始 `{path.relative_to(ROOT)}/metrics.csv`；状态 `{audit}`；读取 {len(rows)} 回合、形成 {len(pairs)} 同模型同初始场景配对。一个训练seed，每族每分布两个场景；只作理想观测敏感性检查，不作为标准规模优势证据。', '',
            '表中全部差值为 **sensor减privileged**，先在同模型、同family/condition/scenario/training_seed内配对，再对episode等权平均。两侧分别真实闭环重规划，不能拿其他方法或不同初值作参照。传感侧删除真实执行器列，只由过去速度差估计；没有仿真测量噪声、网络延迟或真实底盘反馈。', '',
            table(['方法','分布','配对数','碰撞率差','完成率差','进度差m','gap RMSE差m','jerk差m/s³','回退次数差','回合p95差ms','超时率差'],
                  [[r['model'],r['condition'],r['paired_episodes'],
                    *[f(r[m]) for m in ('collision','completed','progress_m','gap_rmse_m','jerk_rms_mps3','fallback_count','cycle_p95_ms','timeout_rate')]] for r in summary]), '',
            '观测估计误差与墙钟deadline引发的回退都可能改变闭环轨迹；差值不能全部解释为状态估计器自身的纯因果效应。尤其停车时速度恒零不能重建隐藏的负制动执行器状态。', '']


def build_diagnostics(path, assets):
    summary = read_json(path/'diagnostics/summary.json', {})
    if not summary:
        return ['### 本轮不确定性诊断', '', f'尚无 `{path.relative_to(ROOT)}/diagnostics/summary.json`，不引用旧试跑诊断替代本轮冻结模型结果。', '']
    manifest = read_json(path/'data/manifest.json', {})
    if summary.get('data_sha256') != manifest.get('sha256'):
        return ['### 本轮不确定性诊断', '', '诊断数据SHA256与本轮冻结manifest不匹配，不将其作为本轮证据。', '']
    model_path = Path(summary.get('model_directory', '')).resolve()
    if not model_path.is_relative_to(path.resolve()):
        return ['### 本轮不确定性诊断', '', f'诊断模型目录 `{model_path}` 不在本轮冻结目录下，不将其作为本轮证据。', '']
    text = ['### 本轮不确定性诊断', '',
            f"诊断模型 `{summary.get('model_directory', '')}`，数据SHA256 `{summary.get('data_sha256', '')}`。校准只用val，经验覆盖/宽度按episode汇总；各成员递归自身历史。分歧不是已校准置信度，较高覆盖也可能来自过宽区间。", '']
    interval = summary.get('interval_summary', [])
    write_csv(assets/f'{path.name}_interval_summary.csv', interval)
    text += [table(['集合','时域s','状态','单位','episodes','覆盖率','覆盖95%区间','区间宽度','宽度95%区间'],
                   [[r['split'],r['horizon_s'],r['component'],r['unit'],r['episodes'],f(r['episode_mean_coverage']),
                     r['coverage_bootstrap95'],f(r['episode_mean_width']),r['width_bootstrap95']] for r in interval]), '']
    # Shared physics may fail collectively while all neural members agree.
    for r in interval:
        if r['split']=='ood' and r['component']=='ego_speed' and float(r['horizon_s'])==3.:
            text += [f"OOD三秒自车速度经验覆盖率为{f(r['episode_mean_coverage'])}，平均宽度{f(r['episode_mean_width'])}m/s。共同结构误差可使集成成员一致出错；该结果不能外推为OOD置信或安全保证。", '']
    text += ['在线 ensemble_risk 仅使用间距成员分歧的启发式惩罚，未使用此处的val经验校准半径。离线覆盖结果不能写成规划器已使用校准风险界。', '']
    sensors = summary.get('sensor_summary', [])
    if sensors:
        write_csv(assets/f'{path.name}_sensor_open_summary.csv', sensors)
        text += ['同模型开环传感减特权误差（三秒；其余时域见附属CSV）：', '',
                 table(['模型','集合','状态','单位','episodes','特权MAE','传感MAE','误差差','配对episode区间'],
                       [[r['model'],r['split'],r['component'],r['unit'],r['episodes'],f(r['privileged_mae']),f(r['sensor_mae']),
                         f(r['sensor_minus_privileged']),r['paired_episode_bootstrap95']] for r in sensors if float(r['horizon_s'])==3.]), '']
    return text


def trajectory_path(path, row):
    fi = ('constant', 'mild', 'brake', 'stopgo').index(row['family'])
    pair = row['scenario_seed']-900000-fi*1000
    if pair >= 10000:
        pair -= row['training_seed']*10000
    return path/f"seed{row['training_seed']}"/'trajectories'/f"{row['budget']}_{row['training_seed']}_{row['condition']}_{row['family']}_{pair}_{row['model']}.csv"


def failure_visuals(path, rows, assets, tag):
    if not rows:
        return []
    # Registered deterministic rule includes every model; no manual cherry picking.
    case = min(rows, key=lambda r: (-r['collision'], r['min_gap_m'], r['budget'], r['model']))
    paired = [r for r in rows if all(r[k] == case[k] for k in
              ('family', 'condition', 'scenario_seed', 'training_seed', 'budget'))]
    logs = {r['model']: read_csv(trajectory_path(path, r)) for r in paired}
    logs = {k: v for k, v in logs.items() if v}
    if not logs:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name, log in sorted(logs.items()):
        time = [float(r['time_s']) for r in log]
        for ax, key in zip(axes, ('gap_m', 'speed_mps', 'command_mps2')):
            ax.plot(time, [float(r[key]) for r in log], label=name)
    for ax, label in zip(axes, ('Gap [m]', 'Ego speed [m/s]', 'Command [m/s²]')):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].axhline(0, color='r', ls='--')
    axes[-1].set_xlabel('Time [s]'); axes[0].legend(ncol=3, fontsize=7)
    fig.suptitle(f"{tag}: minimum-gap case, {case['condition']}/{case['family']}, seed={case['scenario_seed']}, {case['budget']}")
    fig.tight_layout(); fig.savefig(assets/f'{tag}_failure.png', dpi=130); plt.close(fig)
    # Animation is a direct time cursor through recorded clearance, no invented road geometry.
    frames = []
    fig, ax = plt.subplots(figsize=(9, 4))
    end = max(float(v[-1]['time_s']) for v in logs.values())
    maximum = max(float(r['gap_m']) for log in logs.values() for r in log)
    for t in np.linspace(0, end, min(45, int(end/.2)+1)):
        ax.clear()
        for index, (name, log) in enumerate(sorted(logs.items())):
            current = min(log, key=lambda r: abs(float(r['time_s'])-t))
            gap = float(current['gap_m'])
            ax.plot([0, gap], [index, index], '-', alpha=.4)
            ax.scatter([0, gap], [index, index], marker='s', s=45)
            ended = ' (log ended)' if t > float(log[-1]['time_s'])+.05 else ''
            ax.text(gap+.3, index, f"{gap:.1f}m{ended}", fontsize=7)
        ax.set_yticks(range(len(logs)), sorted(logs)); ax.set_xlim(-2, maximum+8)
        ax.set_xlabel('Recorded clearance from ego front [m]')
        ax.set_title(f'{tag} paired clearance replay, t={t:.1f}s; each row has own ego origin')
        fig.tight_layout(); fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    plt.close(fig)
    frames[0].save(assets/f'{tag}_clearance.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
    selection = {k: case[k] for k in ('model', 'family', 'condition', 'scenario_seed', 'training_seed', 'budget', 'collision', 'min_gap_m')}
    (assets/f'{tag}_failure_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    return [f"选例规则：先碰撞、再全方法最小间距，固定排序；选择 `{selection}`。这是失败/边界展示，不代表总体效果。碰撞终止后的帧不补造，轨迹日志记录步进前状态。",
            f'![配对失败案例](report_assets/{tag}_failure.png)', f'![真实日志回放](report_assets/{tag}_clearance.gif)']


def build_longitudinal(path, assets):
    tag = path.name
    rows = [r for p in sorted(path.glob('seed*/closed_*.jsonl')) for r in read_jsonl(p)]
    audit = longitudinal_audit(path, rows)
    (assets/f'{tag}_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    text = [f'## 纵向 {tag}', '', f'原始目录 `{path.relative_to(ROOT)}`；状态 **{audit["status"]}**，期望 {audit["expected"]}、已记录 {audit["actual"]}、缺失 {audit["missing"]}、重复 {audit["duplicates"]}。',
            '未齐全时下表仅为中间记录，模型样本不平衡，不能据此下优势结论。smoke 即使齐全也仅验证工程路径。', '']
    training = [read_json(p) for p in sorted(path.glob('seed*/models/training.json'))]
    text += ['训练成本：'+('; '.join(f"seed {r['seed']}: {r['training_seconds']:.2f}s" for r in training) or '尚无日志')+'；包括该 seed 的全部纵向模型，CPU NumPy 训练，不代表 ARM64 训练或部署性能。', '']
    manifest = read_json(path/'data/manifest.json', {})
    if manifest:
        text += [f"数据版本 `{manifest['version']}`，内容SHA256 `{manifest['sha256']}`；各集合完整episode数："+
                 ', '.join(f"{split}={sum(r['split']==split for r in manifest['episodes'])}" for split in ('train','val','test','ood'))+'。归一化与参数辨识仅train，模型选择与预算仅val。', '']
    raw_open = [r for p in sorted(path.glob('seed*/open_loop_seed*.csv')) for r in read_csv(p)]
    metrics = ('gap_mae_m', 'ego_mae_mps', 'lead_mae_mps', 'actuator_mae_mps2')
    ep, means = episode_mae(raw_open, metrics)
    write_csv(assets/f'{tag}_open_episode.csv', ep); write_csv(assets/f'{tag}_open_summary.csv', means)
    text += ['### 递归开环', '', '先对每个 episode 内所有非重叠窗口取均值，再对 episode 均值等权汇总。标准版在同一冻结数据上训练三个seed，因此下表数量为seed×episode评估次数，不能将跨seed重复的测试episode误当新增独立场景；独立场景数见数据manifest。不是把相邻帧当独立样本，完整 0.1/0.5/1/3s 数值见附属 CSV。', '',
             table(['模型', '集合', '时域 s', 'seed×episode次数', '间距 MAE m', '自车 MAE m/s', '前车 MAE m/s', '执行器 MAE m/s²'],
                   [[r['model'], r['split'], f(r['horizon_s']), r['episodes'], *[f(r[m]) for m in metrics]] for r in means if r['horizon_s'] in (.5, 1., 3.)]), '']
    for split in ('test', 'ood'):
        at3 = {r['model']: r for r in means if r['split']==split and r['horizon_s']==3.}
        baseline = at3.get('physical_enriched')
        if baseline:
            text += [f"{split} 3秒点估计相对强物理（间距MAE {f(baseline['gap_mae_m'])}m、自车MAE {f(baseline['ego_mae_mps'])}m/s）："+
                     '；'.join(f"{name} 间距差{float(row['gap_mae_m'])-baseline['gap_mae_m']:+.4g}m、自车差{float(row['ego_mae_mps'])-baseline['ego_mae_mps']:+.4g}m/s"
                              for name,row in sorted(at3.items()) if name!='physical_enriched')+
                     '。这是全部方法的描述性均值差，不是独立的显著性结论；需要同时检查下方闭环结果。', '']
            if 'history' in at3 and 'learned_lead' in at3:
                delta = at3['history']['gap_mae_m']-at3['learned_lead']['gap_mae_m']
                text += [f"归因检查：history 与仅学习前车 learned_lead 的3秒间距MAE差为{delta:+.4g}m；二者接近表示该条件下学习自车残差没有提供额外可见预测收益，不能将前车历史能力说成自车动力学突破。", '']
    summary = closed_summary(rows); write_csv(assets/f'{tag}_closed_summary.csv', summary)
    text += ['### 闭环全部方法与预算', '',
             table(['模型', '分布', '预算', 'n', '碰撞', '完成率', '进度 m', 'gap RMSE m', 'jerk m/s³', '回退/回合', '推理中位数均值 ms', '回合p95均值 ms', '超时率'],
                   [[r['model'], r['condition'], r['budget'], r['episodes'], r['collisions'],
                     *[f(r[m]) for m in ('completion', 'progress_m', 'gap_rmse_m', 'jerk_rms_mps3', 'fallback_count', 'inference_p50_ms', 'cycle_p95_ms', 'timeout_rate')]] for r in summary]), '']
    effects = read_csv(path/'paired_effects.csv')
    text += ['效应为学习/经典方法减去丰富物理 `physical_enriched`；低误差/碰撞的负差值较好，进度正差值较好。区间依照原始配对文件的统计单位，3训练种子的训练不确定性估计仍不稳定。', '',
             table(['方法', '分布', '预算', '指标', '配对数', '差值', '95%区间'],
                   [[r['model'], r['condition'], r['budget'], r['metric'], r['independent_pairs'], f(r['difference']), f"[{f(r['ci_low'])}, {f(r['ci_high'])}]" ] for r in effects]), ''] if effects else ['配对效应文件尚未生成，不计算或暗示显著性。', '']
    latency = read_csv(path/'latency_summary.csv')
    if latency:
        text += ['完整周期合并分位数（区别于上面的回合p95均值）：', '', table(list(latency[0]), [list(r.values()) for r in latency]), '']
    text += ['周期为主机状态估计、优化和安全检查，不含真实车辆 I/O。共享主机争用、workers 与 deadline 可能影响回退和结果；近似时间预算仅由 val profile 选候选数，需同时检查实际时间，不能宣称严格等时或硬实时。', '',
             '可复现的范围是冻结的软件、数据、种子及实验配置；控制器使用实际墙钟超100ms触发制动回退，所以主机负载和系统调度可能改变触发时刻，进一步改变之后的状态与控制。闭环数值不能保证跨次逐位一致，重复执行也不能据不同负载挑选更好的结果。该基准不证明硬实时或Pi端截止期满足。', '']
    profiles = [(p.parent.name, r) for p in sorted(path.glob('seed*/validation_profile.json')) for r in read_json(p, [])]
    if profiles:
        text += ['验证集完整周期预算选择（每模型均计算候选数×时域×迭代×成员数）：', '',
                 table(['训练seed','模型','等候选ms','近似等时候选数','目标ms','实测ms','相对预算误差','15%内'],
                       [[seed,r['model'],f(r['cycle_median_ms']),r['matched_candidates'],f(r['target_ms']),f(r['matched_validation_ms']),f(r['relative_budget_error']),r['within_15_percent']] for seed,r in profiles]), '']
    text += build_diagnostics(path, assets)
    text += family_tables(path,rows,assets,audit['status']=='verified_complete')
    text += failure_visuals(path, rows, assets, tag)
    return text, audit


def build_ackermann(path, assets):
    complete = read_json(path/'completeness.json', {})
    rows = read_csv(path/'summary.csv')
    text = [f'## 二维 {path.name}', '', f'目录 `{path.relative_to(ROOT)}`；完整性 `{complete or "provisional: no completeness file"}`。', '']
    if rows:
        text += [table(list(rows[0]), [list(r.values()) for r in rows]), '']
    raw = read_csv(path/'open_loop.csv')
    if not raw:
        raw = [dict(r, seed=p.parent.name) for p in path.glob('seed_*/open_loop.csv') for r in read_csv(p)]
    if raw:
        ep, means = episode_mae(raw, ('position_error_m','yaw_error_rad','speed_error_mps','steer_error_rad'), 'seed')
        write_csv(assets/f'ackermann_{path.name}_open_episode.csv', ep)
        write_csv(assets/f'ackermann_{path.name}_open_summary.csv', means)
        text += ['递归开环按 episode 等权（yaw rad、转角 rad；位置 m、速度 m/s）：', '',
                 table(list(means[0]), [list(r.values()) for r in means if r['horizon_s'] in (.5,1,3)]), '']
    effects = read_json(path/'paired_effects.json', [])
    if effects:
        text += ['学习残差减去强丰富物理的效应；完整结果保留全部基线：', '',
                 table(['预算','族','条件','指标','n','差值','训练seed区间'], [[r['budget'],r['family'],r['condition'],r['metric'],r['paired_episodes'],f(r['learned_minus_physics']),r['bootstrap_training_seed_ci95']] for r in effects if r['baseline']=='rich_physics_mpc']), '']
    for name in ('trajectories.png','obstacle_comparison.gif'):
        if (path/name).exists():
            text += [f'![二维真实实验日志](../{(path/name).relative_to(ROOT).as_posix()})', '']
    text += ['条件1.0为标称仿真假设、1.3为响应变化；smoke 只有每族一个回合，不能给出跨条件/训练种子结论。丰富物理能表达生成机制；残差无收益必须如实保留。', '']
    return text


def research_conclusion(run,assets,audit):
    """Interpret the registered history method, never pick a test-set winner."""
    lines=['## 六个研究问题的当前答案', '']
    if audit['status']!='verified_complete':
        return lines+['标准实验尚未齐全，主history相对登记强物理的全局结论暂不成立；不根据部分测试选择新主方法。完成后本节自动给出各分布、各计算预算的配对效应、驾驶代价和归因。', '']
    rows=[r for p in sorted(run.glob('seed*/closed_*.jsonl')) for r in read_jsonl(p)]
    effects=read_csv(run/'paired_effects.csv')
    lookup={(r['model'],r['condition'],r['budget'],r['metric']):r for r in effects}
    latency={(r['model'],r['condition'],r['budget']):r for r in read_csv(run/'latency_summary.csv')}
    groups=defaultdict(list)
    for r in rows:groups[r['model'],r['condition'],r['budget']].append(r)
    primary=[];better=0;worse=0;progress_losses=0
    for condition in ('id','ood'):
        for budget in ('equal','time'):
            h=groups.get(('history',condition,budget),[]);p=groups.get(('physical_enriched',condition,budget),[])
            if not h or not p:continue
            def effect(metric):
                r=lookup.get(('history',condition,budget,metric))
                return f"{float(r['difference']):+.4g} [{float(r['ci_low']):+.4g},{float(r['ci_high']):+.4g}]" if r else '未生成'
            gap=lookup.get(('history',condition,budget,'gap_rmse_m'))
            progress=lookup.get(('history',condition,budget,'progress_m'))
            if gap:
                better+=float(gap['ci_high'])<0;worse+=float(gap['ci_low'])>0
            if progress:progress_losses+=float(progress['ci_high'])<0
            def mean(items,metric):return float(np.mean([r[metric] for r in items]))
            def duration(model,items):
                measured=latency.get((model,condition,budget))
                if measured:return f"{float(measured['p50_ms']):.3g}/{float(measured['p95_ms']):.3g}"
                return f"未生成（回合p95均值{mean(items,'cycle_p95_ms'):.3g}）"
            primary.append([condition,budget,len(h),
                f"{sum(r['collision'] for r in h)}/{len(h)} vs {sum(r['collision'] for r in p)}/{len(p)}",
                f"{mean(h,'completed'):.3f} vs {mean(p,'completed'):.3f}",effect('gap_rmse_m'),effect('progress_m'),effect('jerk_rms_mps3'),
                f"{mean(h,'fallback_count'):.3g} vs {mean(p,'fallback_count'):.3g}",
                f"{duration('history',h)} vs {duration('physical_enriched',p)}",
                f"{mean(h,'timeout_rate'):.3g} vs {mean(p,'timeout_rate'):.3g}",
                f"{min(r['candidates'] for r in h)}–{max(r['candidates'] for r in h)} vs {min(r['candidates'] for r in p)}–{max(r['candidates'] for r in p)}"])
    lines+=['1. **学到了什么、没有学到什么？** 固定前车头学习了速度与历史增量的单步相关性，自车残差在结构匹配ID中接近零。val组件交换把匀速族三秒约1.956m间距误差归因于前车头，单步前车误差0.003423m/s递归到1.646m/s；history与仅学习前车的组合几乎相同。这个证据支持“前车递归失稳”解释，不能包装成新的自车动力学能力，也不能仅凭相关性证明全部闭环差异的因果机制。', '',
            '2. **相对最强合理基线改善了哪些指标？** 下表固定主方法history与预登记physical_enriched，未用测试选其他赢家。差值为history减强物理，方括号为训练seed→配对episode分层95%区间；低误差/jerk负值较好，进度正值较好。主表区间未作多指标校正，不能据大量分层检验挑选正例。', '',
            table(['分布','预算','回合/方法','碰撞h vs p','完成率h vs p','gap RMSE差m [CI]','进度差m [CI]','jerk差m/s³ [CI]','回退/回合h vs p','实际周期p50/p95 ms h vs p','超时率h vs p','实际候选范围h vs p'],primary),'']
    if not better:
        verdict=f'四个分布×预算比较中，主要gap指标没有一个95%区间完整低于零；其中{worse}个区间完整高于零。当前登记结果不支持主history改善平均间距跟踪，更不能宣称世界模型总体驾驶收益成立。'
    else:
        verdict=f'四个分布×预算比较中，主要gap指标有{better}个95%区间完整低于零、{worse}个完整高于零，其余跨零。证据仅支持下表所列条件和指标的局部变化；安全、完成、进度和计算代价同时约束驾驶收益，不能概括为普遍优越。'
    lines += [verdict, '',
              '3. **改善或退化来自哪个组件？** 强物理已辨识ID自车机制，历史模型没有额外可见自车收益。val的ego-only/leader-only交换及learned_lead消融把主要预测变化定位到前车头。一次6个稳定AR2候选+3个衰减CA参数的登记修复，将val平均gap误差0.80785m降到0.489326m，但仍劣于强物理0.399817m，因此保留负结果并停止，不替换正式测试方法。', '',
              f'4. **是否只是更保守或计算更多？** 等候选比较固定64候选×30步×3迭代，主方法实际完整周期增加量见表；近时预算用val选候选数，其真实候选范围与实测p50/p95也同时列出，不能因候选数相同就假设耗时相同。进度配对区间完整小于零的比较有{progress_losses}个，任何较大间距、较低碰撞若伴随进度/完成下降，都不能按同等效率收益解释。回退按原因附表区分超时制动与间距安全监督，不能把主机更频繁制动冒充模型能力。', '']
    family=read_csv(assets/f'{run.name}_family_paired_effects.csv')
    selected=[r for r in family if r['model']=='history' and r['metric']=='gap_rmse_m']
    improved=[f"{r['family']}/{r['condition']}/{r['budget']}" for r in selected if float(r['ci_high'])<0]
    degraded=[f"{r['family']}/{r['condition']}/{r['budget']}" for r in selected if float(r['ci_low'])>0]
    lines += ['5. **哪些场景没有优势或更差？** 主history分族gap误差区间完整偏好的格子：'+('、'.join(improved) or '无')+'；完整偏差的格子：'+('、'.join(degraded) or '无')+'。其余分层区间跨零。这些是完整披露的分层诊断，未作多重比较校正；全部方法、安全/进度/jerk与原始样本数均在分族附表，不能单凭该列表选赢家。ID与OOD分别报告，OOD改变执行器时常和初始速度；有限仿真不能外推任意实车道路。', '']
    logs=[read_json(p) for p in run.glob('seed*/models/training.json')]
    cost=sum(r['training_seconds'] for r in logs)
    worth=('就当前登记主方法而言，没有证据证明额外训练与在线计算成本值得，保留丰富物理+MPC作为本任务更有根据的默认选择。'
           if not better else '目前只有有限条件下的指标支持，是否值得必须限定到这些条件并结合表中安全、进度和实际运行开销；证据不足以替换所有任务的强物理控制器。')
    lines += [f'6. **数据、训练和运行代价是否值得？** 三个训练seed的全部纵向模型训练日志合计{cost:.3f}s；这包括多个模型而非单独history成本。额外数据采集、验证和模型维护也不是零成本，在线成本用上表完整周期而非单次前向衡量。'+worth+' 这只针对当前仿真、观测条件和模型结构，不否定其他机制或有真实标定数据时世界模型可能有价值。', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output/'report_assets'; assets.mkdir(exist_ok=True)
    generated = datetime.now(timezone(timedelta(hours=8), 'Asia/Shanghai')).isoformat(timespec='seconds')
    text = ['# FalconWorld 实验报告', '', f'生成时间：{generated}。本报告由 `python scripts/build_report.py` 从磁盘实验结果生成，可在标准实验结束后重跑。未运行的项目不据代码存在而算完成。', '',
            '## 为什么加入世界模型：证据、收益、代价与适用边界', '',
            '建模问题是执行器响应失配与未知前车运动。学习器使用当前状态、过去6帧和候选动作，模型自身递归未来；CEM比较候选序列并只执行第一步，再读新观测。系统属于基于模型的规划控制，没有训练策略或价值函数，不称强化学习。', '',
            '证据链按数据和物理辨识 → 独立episode递归预测 → 同一MPC候选评估 → 闭环安全/进度/舒适性 → 完整计算周期报告。强物理基线通过train辨识非对称、速度相关时常；固定无噪声一阶结构可由辨识解决，不能把失配tau=0.5作为唯一比较。', '',
            '学习残差、无历史/有历史、仅学习前车、集成和集成风险开关均保留。若 history 与 learned_lead 接近，则不能把前车预测收益归因于学习自车动力学。风险项只用成员间距分歧，离线经验区间是独立诊断；不具备概率安全保证。未知时刻的无先兆急刹不能由模型可靠预知。', '',
            '主实验使用特权执行器状态。传感实验移除该列并因果估计，停车时内部制动不可辨识。ID/OOD分别报告，OOD同时改变速度与执行器响应；这些均是仿真假设，不是R550实测标定。零碰撞不代表通用安全保证；较低预测误差不自动构成闭环收益。', '']
    audits = {}
    for name in ('smoke_verified','standard_verified'):
        path = ROOT/'outputs'/name
        section, audit = build_longitudinal(path, assets)
        text += section; audits[name] = audit
    text += research_conclusion(ROOT/'outputs/standard_verified',assets,audits['standard_verified'])
    text += build_sensor_closed(assets)
    for name in ('smoke_verified','standard_verified'):
        text += build_ackermann(ROOT/'outputs'/'ackermann'/name, assets)
    text += ['## 外部集成与部署证据', '',
             '用户已确认R550未上线并要求暂不部署。本轮只验收本地模型、导出包、日志回放及影子运行；未执行车端安装或真实车辆运动。', '',
             'CARLA实测与模型导出验证独立于上述合成任务，详细条件、版本和限制见 [CARLA](CARLA.md) 与 [R550](R550.md)。不能把本机ONNX验证写成Pi实时性或实车部署完成。', '']
    carla = read_json(ROOT/'outputs/carla_verified_20261104/metrics.json', {})
    if carla:
        cr = carla.get('closed_loop', [])
        if cr:
            keys = ('model','seed','steps','collision','completion','progress_m','speed_rmse_m_s','jerk_rms_m_s3','cycle_p95_ms','timeout_rate')
            text += ['CARLA当前实测有限场景记录（2个场景seed，不能泛化为大规模优势）：', '', table(list(keys), [[r[k] for k in keys] for r in cr]), '']
    exports = read_json(ROOT/'outputs/deployment_20261104/validation.json', [])
    if exports:
        text += [f'主机 ONNX 数值检查记录 {len(exports)} 项；详细逐模型、成员递归及误差见 `outputs/deployment_20261104/validation.json`，不代表ARM64实测。', '']
    text += ['## 当前结论与可复现边界', '',
             '当配对方法都没有观察到碰撞时，经验bootstrap可能返回退化的[0,0]差值区间；这不证明真实碰撞风险为零，也不证明两方法安全等价。', '',
             '逐表保留所有方法和负差值。只有 standard 完整性通过、配对效应区间与安全/进度/耗时同时支持时，才可陈述特定指标收益；任何中间表不能用于选择测试赢家或重复调参。无额外收益同样属于实验结论。训练和模型推理增加成本是否值得，应针对具体场景和设备周期判断，不能由训练loss或主机单次前向速度代替。', '',
             '原始记录、冻结源摘要、数据版本、训练日志和逐周期轨迹位于各实验目录。报告附属CSV保存每episode开环和汇总，不覆盖原始实验数据。中文报告不是外部设备验收凭证，硬件运动仍需明确授权与已核验的停车、接管、超时能力。', '']
    if (ROOT/'docs/VALIDATION_ANALYSIS.md').exists():
        text += ['## 训练与验证原因分析', '',
                 '固定模型的[验证诊断](VALIDATION_ANALYSIS.md)只用train/val：按族比较单步teacher-forcing与0.1/0.5/1/3秒递归误差、冻结自车/前车组件交换、真实val学习曲线及最终train/val误差。该分析解释负结果和适用边界，不训练、不调整本轮模型、不选择测试赢家。', '']
    study=read_json(ROOT/'outputs/stable_history_validation/selection.json',{})
    if study and (ROOT/'docs/STABLE_HISTORY_STUDY.md').exists():
        scores=study['baseline_scores']
        text += ['## 一次验证失败修复的负结果', '',
                 f"[稳定历史候选研究](STABLE_HISTORY_STUDY.md)基于val递归漂移登记6个AR2超参数与3个衰减CA参数，一次搜索耗时{study['elapsed_s']:.3f}s（含验证和主机profile）。只访问train/val，原正式测试结果完整保留。val平均gap分数：原history {scores['history']:.6g}m、稳定AR2 {scores['stable_history']:.6g}m、强物理/合理衰减CA {scores['adaptive_damped_ca']:.6g}m。候选减少了原神经历史头的漂移，但仍未胜过合理强基线，故停止搜索、不追加测试。", '',
                 '这些均为验证集描述性均值，无独立统计区间，不称显著改善，也不构成冻结test优势或闭环驾驶收益证据。零截距与AR根约束仅防止该前车增量模型的特定自激机制，不是碰撞安全保证。', '']
    (args.output/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    english = ['# FalconWorld', '', 'A reproducible research project for compact action-conditioned world models and receding-horizon control, with longitudinal following, synthetic Ackermann driving, CARLA adaptation, and an R550 deployment preparation path.', '',
               'The pipeline generates episode-disjoint data, identifies strong physical dynamics from training data, fits compact neural/residual/history predictors, and evaluates recursive prediction and closed-loop CEM-MPC. Models share action limits, cost and safety supervision. Classic controllers, enriched physics, learned leader-only dynamics and uncertainty ablations remain visible.', '',
               'This is model-based planning and control, not reinforcement learning. The synthetic longitudinal benchmark exposes actuator state; sensor-derived observations are evaluated separately. No future leader script enters a predictor. The enriched physical model can exactly identify the noise-free in-distribution ego mechanism, so a learned residual need not improve it.', '',
               '## Current evidence', '', table(['Run','Status','Recorded / expected episodes'], [[k,v['status'],f"{v['actual']} / {v['expected']}"] for k,v in audits.items()]), '',
               'Incomplete results are provisional; smoke runs test the pipeline and cannot establish statistical superiority. The generated [Chinese report](REPORT.md) includes all model comparisons, episode-weighted recursive errors, paired effects when available, complete-cycle latency, training cost and deterministic failure replays. It retains adverse and null results.', '',
               'Frozen software, data and seeds are reproducible; trajectories are not guaranteed bitwise identical because a measured wall-clock deadline triggers braking fallback, and host load can change that fallback. This does not demonstrate hard real-time behavior.', '',
               'Run `python scripts/build_report.py` after experiments finish to refresh evidence. See the repository README for the verified reproduction commands, [CARLA](CARLA.md) for simulator acceptance, and [R550](R550.md) for replay, shadow mode, ARM64 preparation and hardware acceptance. Host export consistency and synthetic simulations do not establish Raspberry Pi timing or real-vehicle deployment. Real motion requires explicit authorization.', '', f'Generated: {generated}.']
    (args.output/'PROJECT_EN.md').write_text('\n'.join(english)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(args.output/'REPORT.md'), audits=audits), ensure_ascii=False))


if __name__ == '__main__':
    main()
