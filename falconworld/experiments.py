"""Frozen episode evaluation, raw metrics, resumable paired comparisons."""
import csv
import json
import hashlib
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import numpy as np
from .planning import ACC, LongitudinalMPC, CEMConfig


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path,repair=False):
    path=Path(path)
    if not path.exists():return []
    lines=path.read_text(encoding='utf-8').splitlines()
    rows=[]
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
            if not repair:raise ValueError(f'Incomplete JSONL tail: {path}')
            path.with_suffix(path.suffix+'.incomplete_tail.txt').write_text(line,encoding='utf-8')
            temporary=path.with_suffix(path.suffix+'.tmp')
            temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            temporary.replace(path)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def recursive_prediction(model, state, actions, history):
    if getattr(model,'members',None):
        # Match online MPC: each member recurses its own state and history.
        return np.mean([recursive_prediction(member,state,actions,history)
                        for member in model.members],axis=0)
    state, history = np.array(state, copy=True), np.array(history, copy=True)
    trajectory = []
    for action in actions:
        state = model.step(state, action, history=history)
        trajectory.append(state.copy())
        history = np.concatenate((history[1:], state[None]), axis=0)
    return np.asarray(trajectory)


def open_loop(models, data, output, seed):
    rows = []
    for split in ('test', 'ood'):
        arrays = data[split]
        for episode in np.unique(arrays['episode']):
            indices = np.flatnonzero(arrays['episode'] == episode)
            # Nonoverlapping windows preserve recursive semantics; episode remains statistical unit.
            for offset in range(0, len(indices)-29, 30):
                idx = indices[offset:offset+30]
                for name, model in models.items():
                    prediction = recursive_prediction(model, arrays['state'][idx[0]], arrays['action'][idx], arrays['history'][idx[0]])
                    for h in (1,5,10,30):
                        error = np.abs(prediction[h-1]-arrays['next_state'][idx[h-1]])
                        rows.append(dict(model=name, training_seed=seed, split=split, episode=int(episode),
                                         family=str(arrays['family'][idx[0]]), offset=offset, horizon_s=h*.1,
                                         gap_mae_m=float(error[0]), ego_mae_mps=float(error[1]),
                                         lead_mae_mps=float(error[2]), actuator_mae_mps2=float(error[3])))
    write_csv(Path(output)/f'open_loop_seed{seed}.csv', rows)
    return rows


def closed_episode(model, name, family, scenario_seed, condition, training_seed,
                   config, steps, log_path=None, risk_weight=0., observation_mode='privileged'):
    from .data import make_episode, step_truth, scenario_lead_accel
    episode = make_episode(family, scenario_seed, condition=condition)
    state = np.asarray(episode['state'], dtype=float)
    history = np.tile(state, (6,1))
    planner = ACC() if name == 'acc' else LongitudinalMPC(model, config, risk_weight, seed=scenario_seed)
    logs, gaps, speeds, cycles, inference, accel, fallbacks = [], [], [], [], [], [], []
    progress, reference_progress = 0., 0.
    collision = False
    for t in range(steps):
        control_start=perf_counter()
        if observation_mode=='sensor':
            from .diagnostics import sensor_history
            observed=sensor_history(history[:,:3])
            command, info=planner.command(observed[-1],observed)
        elif observation_mode=='privileged':
            command, info = planner.command(state, history)
        else:
            raise ValueError('Unknown observation mode')
        info['cycle_s']=perf_counter()-control_start
        lead_accel = scenario_lead_accel(family, t*.1, episode['params'])
        future = step_truth(state, command, lead_accel, condition=condition)
        progress += state[1]*.1
        reference_progress += state[2]*.1
        accel.append(float((future[1]-state[1])/.1))
        gaps.append(float(future[0]-(4+1.2*future[1])))
        speeds.append(float(future[1]))
        cycles.append(info['cycle_s'])
        inference.append(info['inference_s'])
        fallbacks.append(info['fallback'])
        logs.append(dict(step=t, time_s=t*.1, gap_m=float(state[0]), speed_mps=float(state[1]),
                         lead_mps=float(state[2]), actuator_mps2=float(state[3]), command_mps2=command,
                         measured_accel_mps2=accel[-1], **info))
        state = future
        history = np.concatenate((history[1:], state[None]), axis=0)
        if state[0] <= 0:
            collision = True
            break
    if log_path:
        write_csv(log_path, logs)
    jerk = np.diff(accel)/.1
    return dict(model=name, family=family, condition=condition, scenario_seed=scenario_seed,
                training_seed=training_seed, steps=len(logs), collision=int(collision),
                completed=int(not collision and len(logs)==steps and progress>=.5*max(5.,reference_progress)),
                progress_m=float(progress), reference_progress_m=float(reference_progress),
                gap_rmse_m=float(np.sqrt(np.mean(np.square(gaps)))),
                mean_speed_mps=float(np.mean(speeds)), jerk_rms_mps3=float(np.sqrt(np.mean(jerk**2))) if len(jerk) else 0.,
                fallback_count=sum(bool(s) for s in fallbacks), solver_failures=fallbacks.count('solver_failure'),
                cycle_p50_ms=float(np.quantile(cycles,.5)*1000), cycle_p95_ms=float(np.quantile(cycles,.95)*1000),
                timeout_rate=float(np.mean(np.asarray(cycles)>config.dt)),
                inference_p50_ms=float(np.median(inference)*1000),
                min_gap_m=float(min(state[0],min(row['gap_m'] for row in logs))), candidates=config.candidates,
                horizon=config.horizon, iterations=config.iterations, risk_weight=risk_weight,observation_mode=observation_mode)


def profile_models(models, validation, output, base_config):
    cache=Path(output)/'validation_profile.json'
    cache_fingerprint=dict(config=asdict(base_config),validation_sha256=hashlib.sha256(validation['state'].tobytes()).hexdigest(),
        models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(output)/'models').glob('*.npz')},
        planner_sha256=hashlib.sha256(Path(__file__).with_name('planning.py').read_bytes()).hexdigest())
    cache_meta=Path(output)/'validation_profile_fingerprint.json'
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==cache_fingerprint:
        saved=json.loads(cache.read_text())
        if saved and 'matched_validation_ms' in saved[0]:
            return {r['model']:r['matched_candidates'] for r in saved}
    # Validation-only workload measures full candidate x horizon x iteration cost.
    state = validation['state'][0]
    history = validation['history'][0]
    rows = []
    counts = {}
    for name, model in models.items():
        planner = LongitudinalMPC(model, base_config, seed=99)
        times = []
        for _ in range(4):
            _, info = planner.command(state, history)
            times.append(info['cycle_s'])
        rows.append(dict(model=name, cycle_median_ms=float(np.median(times)*1000), candidates=base_config.candidates))
    minima={}
    for name,model in models.items():
        planner=LongitudinalMPC(model,CEMConfig(candidates=8,elites=8,horizon=base_config.horizon,
            iterations=base_config.iterations,dt=base_config.dt),seed=99)
        trials=[planner.command(state,history)[1]['cycle_s']*1000 for _ in range(3)]
        minima[name]=float(np.median(trials))
    # Use measured minimum workload to choose a feasible common time budget.
    target=max(minima.values())*1.3
    for row in rows:
        low, high = base_config.elites, 16384
        best_count, best_distance, achieved = low, abs(minima[row['model']]-target), minima[row['model']]
        for _ in range(14):
            count=(low+high)//2
            cfg=CEMConfig(horizon=base_config.horizon,candidates=count,iterations=base_config.iterations,
                          elites=base_config.elites,dt=base_config.dt)
            planner=LongitudinalMPC(models[row['model']],cfg,seed=99)
            times=[]
            for j in range(2):
                _,info=planner.command(state,history);times.append(info['cycle_s']*1000)
            elapsed=float(np.median(times))
            if abs(elapsed-target)<best_distance:
                best_count,best_distance,achieved=count,abs(elapsed-target),elapsed
            if elapsed<target:low=count+1
            else:high=count-1
            if high<low:break
        counts[row['model']] = best_count
        row['matched_candidates'] = counts[row['model']]
        row['target_ms'] = target
        row['matched_validation_ms'] = achieved
        row['relative_budget_error'] = (achieved-target)/target
        row['within_15_percent']=abs(achieved-target)/target<=.15
    write_json(Path(output)/'validation_profile.json', rows)
    write_json(cache_meta,cache_fingerprint)
    return counts


def _closed_job(arguments):
    return closed_episode(**arguments)


def benchmark(models, output, seed, preset='smoke', budget='equal', matched=None, workers=1):
    output = Path(output)
    counts = dict(smoke=2, standard=10, full=60)
    lengths = dict(smoke=80, standard=100, full=150)
    count, steps = counts[preset], lengths[preset]
    path = output/f'closed_{budget}_seed{seed}.jsonl'
    existing = read_jsonl(path,repair=True)
    keys = {(r['model'],r['family'],r['condition'],r['scenario_seed']) for r in existing}
    objects = dict(models)
    if budget == 'equal':
        objects = {'acc':None, **objects}
    # Same ensemble and same candidates, with risk toggled, is the uncertainty ablation.
    for name in list(models):
        if 'ensemble' in name:
            objects[name+'_risk'] = models[name]
    signature=dict(preset=preset,budget=budget,steps=steps,count=count,matched=matched if budget=='time' else None,
                   workers=workers,models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/'models').glob('*.npz')},
                   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__),Path(__file__).with_name('planning.py'),Path(__file__).with_name('data.py'),Path(__file__).with_name('models.py')]})
    config_path=output/f'closed_{budget}_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=signature:
        raise ValueError(f'Run config changed: use a new output directory instead of mixing {path}')
    write_json(config_path,signature)
    jobs=[]
    for condition in ('id','ood'):
        for fi,family in enumerate(('constant','mild','brake','stopgo')):
            for i in range(count):
                scenario_seed=900000+fi*1000+i+(seed*10000 if preset=='standard' else 0)
                for name,model in objects.items():
                    if (name,family,condition,scenario_seed) in keys:continue
                    candidates=(matched or {}).get(name.removesuffix('_risk'),64) if budget=='time' else 64
                    jobs.append(dict(model=model,name=name,family=family,scenario_seed=scenario_seed,condition=condition,
                        training_seed=seed,config=CEMConfig(candidates=candidates,elites=min(8,candidates)),steps=steps,
                        log_path=output/'trajectories'/f'{budget}_{seed}_{condition}_{family}_{i}_{name}.csv',
                        risk_weight=1. if name.endswith('_risk') else 0.))
    # Shuffle job submission to distribute resource contention over methods/families.
    np.random.default_rng(1234).shuffle(jobs)
    from concurrent.futures import ProcessPoolExecutor
    executor=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    results=executor.map(_closed_job,jobs,chunksize=1) if executor else map(_closed_job,jobs)
    with path.open('a',encoding='utf-8') as file:
        for done,row in enumerate(results):
            row['budget']=budget
            row['benchmark_workers']=workers
            file.write(json.dumps(row,allow_nan=False)+'\n');file.flush();existing.append(row)
            if done%20==0 or done==len(jobs)-1:
                print(f'closed {budget} seed={seed} {done+1}/{len(jobs)} new episodes',flush=True)
    if executor:executor.shutdown()
    write_csv(output/f'closed_{budget}_seed{seed}.csv',existing)
    return existing


def summarize(output):
    output = Path(output)
    rows = [row for path in output.glob('seed*/closed_*.jsonl') for row in read_jsonl(path)]
    run_config=json.loads((output/'run_config.json').read_text()) if (output/'run_config.json').exists() else {}
    expected_keys=set()
    missing_configs=[]
    for seed in run_config.get('seeds',[]):
        budgets=('equal','time') if run_config.get('budget')=='both' else (run_config.get('budget','equal'),)
        for budget in budgets:
            config_path=output/f'seed{seed}'/f'closed_{budget}_config.json'
            if not config_path.exists():missing_configs.append(str(config_path));continue
            cfg=json.loads(config_path.read_text())
            names=[Path(name).stem for name in cfg['models']]
            names += [name+'_risk' for name in list(names) if 'ensemble' in name]
            if budget=='equal':names.append('acc')
            for fi,family in enumerate(('constant','mild','brake','stopgo')):
                for i in range(cfg['count']):
                    scenario=900000+fi*1000+i+(seed*10000 if cfg['preset']=='standard' else 0)
                    for condition in ('id','ood'):
                        for name in names:expected_keys.add((name,condition,family,seed,scenario,budget))
    identities=[(r['model'],r['condition'],r['family'],r['training_seed'],r['scenario_seed'],r['budget']) for r in rows]
    actual_keys=set(identities)
    missing=expected_keys-actual_keys
    unexpected=actual_keys-expected_keys
    audit=dict(expected_episodes=len(expected_keys),actual_episodes=len(rows),unique_episodes=len(actual_keys),
               duplicate_episodes=len(rows)-len(set(identities)),missing_configs=missing_configs,
               missing_pairs=sorted(missing),unexpected_pairs=sorted(unexpected),
               status='verified_complete' if expected_keys and not missing and not unexpected and len(rows)==len(actual_keys) and not missing_configs else 'provisional_incomplete')
    write_json(output/'completeness.json',audit)
    groups = {}
    for row in rows:
        groups.setdefault((row['model'],row['condition'],row['budget']),[]).append(row)
    summary = []
    for (model,condition,budget),items in sorted(groups.items()):
        row = dict(model=model,condition=condition,budget=budget,episodes=len(items),
                   training_seeds=len(set(r['training_seed'] for r in items)),collisions=sum(r['collision'] for r in items),
                   completion_rate=float(np.mean([r['completed'] for r in items])))
        for metric in ('progress_m','gap_rmse_m','jerk_rms_mps3','fallback_count','cycle_p50_ms','cycle_p95_ms','timeout_rate'):
            row[metric] = float(np.mean([r[metric] for r in items]))
        summary.append(row)
    write_csv(output/'summary.csv',summary)
    # Average training seeds within independent paired scenario before episode bootstrap.
    pairs = []
    reference = 'physical_enriched'
    for model,condition,budget in groups:
        if model == reference or (reference,condition,budget) not in groups:
            continue
        for metric in ('progress_m','gap_rmse_m','jerk_rms_mps3','collision'):
            def aggregate(items):
                mapping = {}
                for r in items:
                    mapping.setdefault((r['family'],r['scenario_seed']),[]).append(r[metric])
                return {k:float(np.mean(v)) for k,v in mapping.items()}
            a,b = aggregate(groups[(model,condition,budget)]),aggregate(groups[(reference,condition,budget)])
            keys = sorted(a.keys() & b.keys())
            delta = np.array([a[k]-b[k] for k in keys])
            if not len(delta):
                continue
            rng = np.random.default_rng(222)
            model_rows=groups[(model,condition,budget)]
            reference_rows=groups[(reference,condition,budget)]
            ref_lookup={(r['training_seed'],r['family'],r['scenario_seed']):r[metric] for r in reference_rows}
            strata={}
            for r in model_rows:
                k=(r['training_seed'],r['family'],r['scenario_seed'])
                if k in ref_lookup:
                    strata.setdefault((k[0],k[1]),[]).append(r[metric]-ref_lookup[k])
            training_seeds=sorted(set(k[0] for k in strata))
            boot=[]
            for _ in range(2000):
                draws=[]
                for s in rng.choice(training_seeds,len(training_seeds),replace=True):
                    for (ss,family),values in strata.items():
                        if ss==s:draws.append(float(np.mean(rng.choice(values,len(values),replace=True))))
                boot.append(float(np.mean(draws)))
            pairs.append(dict(model=model,reference=reference,condition=condition,budget=budget,metric=metric,
                              independent_pairs=len(delta),difference=float(delta.mean()),
                              ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                              unit='training seed then family-stratified paired episode bootstrap',training_seeds=len(training_seeds)))
    write_csv(output/'paired_effects.csv',pairs)
    return summary


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    summary = summarize(output)
    rows = [r for r in summary if r['condition']=='id' and r['budget']=='equal']
    if rows:
        figure, axes = plt.subplots(1,3,figsize=(15,5))
        for ax,metric,title in zip(axes,('progress_m','gap_rmse_m','cycle_p95_ms'),('Progress [m]','Gap RMSE [m]','Mean episode p95 cycle [ms]')):
            ax.bar([r['model'] for r in rows],[r[metric] for r in rows])
            ax.set_title(title)
            ax.tick_params(axis='x',rotation=80)
        figure.tight_layout()
        figure.savefig(output/'comparison.png',dpi=140)
        plt.close(figure)
    pooled={}
    for seed_path in output.glob('seed*'):
        for metrics_path in seed_path.glob('closed_*.jsonl'):
            for line in metrics_path.read_text().splitlines():
                r=json.loads(line)
                fi=('constant','mild','brake','stopgo').index(r['family'])
                offset=r['scenario_seed']-(900000+fi*1000)
                # standard has disjoint scenario instances across training seeds.
                if offset>=10000:offset-=r['training_seed']*10000
                log=seed_path/'trajectories'/f"{r['budget']}_{r['training_seed']}_{r['condition']}_{r['family']}_{offset}_{r['model']}.csv"
                if not log.exists():continue
                key=r['model'],r['condition'],r['budget']
                values=pooled.setdefault(key,[])
                with log.open(newline='') as file:
                    values.extend(float(row['cycle_s'])*1000 for row in csv.DictReader(file))
    write_csv(output/'latency_summary.csv',[dict(model=k[0],condition=k[1],budget=k[2],cycles=len(v),
        p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),timeout_rate=float(np.mean(np.asarray(v)>100))) for k,v in sorted(pooled.items())])
