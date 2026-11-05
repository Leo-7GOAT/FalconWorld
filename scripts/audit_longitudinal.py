"""Independently reconstruct longitudinal metrics from frozen per-cycle CSV logs.

No environment/model/controller implementation is imported or executed. Missing
records remain incomplete; this script never repairs or changes experiment logs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODELS=('physical_cv','physical_ca','physical_enriched','neural','residual','history','learned_lead','ensemble')
FAMILIES=('constant','mild','brake','stopgo')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def json_read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(log,dt,expected_steps):
    if not log:raise ValueError('Empty trajectory')
    columns={key:np.array([float(r[key]) for r in log]) for key in
        ('step','time_s','gap_m','speed_mps','lead_mps','actuator_mps2','command_mps2','measured_accel_mps2','cycle_s','inference_s')}
    if any(not np.isfinite(v).all() for v in columns.values()):raise ValueError('Nonfinite trajectory values')
    n=len(log);gap,ego,lead=columns['gap_m'],columns['speed_mps'],columns['lead_mps']
    if np.any(gap<=0) or np.any(ego<0) or np.any(lead<0):raise ValueError('Invalid pre-step observation')
    if n>expected_steps:raise ValueError('Trajectory exceeds declared duration')
    np.testing.assert_array_equal(columns['step'],np.arange(n))
    np.testing.assert_allclose(columns['time_s'],np.arange(n)*dt,rtol=0,atol=1e-10)
    future_gap=gap+dt*(lead-ego)
    future_speed=np.maximum(0.,ego+dt*columns['actuator_mps2'])
    # These identities check every observed transition, not just the last state.
    np.testing.assert_allclose(future_gap[:-1],gap[1:],rtol=0,atol=1e-9)
    np.testing.assert_allclose(future_speed[:-1],ego[1:],rtol=0,atol=1e-9)
    # Use observed velocity increments except for the absent final post-step row.
    observed_future=np.r_[ego[1:],future_speed[-1]]
    measured_accel=(observed_future-ego)/dt
    np.testing.assert_allclose(measured_accel,columns['measured_accel_mps2'],rtol=0,atol=1e-9)
    if np.any(future_gap[:-1]<=0):raise ValueError('Continued after collision')
    if np.any(columns['command_mps2'] < -6.-1e-10) or np.any(columns['command_mps2']>2.+1e-10):
        raise ValueError('Executed command outside acceleration limits')
    cycles,inference=columns['cycle_s'],columns['inference_s']
    if np.any(cycles<0) or np.any(inference<0) or np.any(inference>cycles+1e-7):
        raise ValueError('Invalid duration accounting')
    progress=float(np.sum(ego)*dt);reference=float(np.sum(lead)*dt)
    jerk=np.diff(measured_accel)/dt
    reasons=Counter(r['fallback'] for r in log if r['fallback'])
    collision=int(future_gap[-1]<=0)
    result=dict(steps=n,collision=collision,
        completed=int(not collision and n==expected_steps and progress>=.5*max(5.,reference)),
        progress_m=progress,reference_progress_m=reference,
        gap_rmse_m=float(np.sqrt(np.mean((future_gap-(4.+1.2*observed_future))**2))),
        mean_speed_mps=float(np.mean(observed_future)),
        jerk_rms_mps3=float(np.sqrt(np.mean(jerk**2))) if n>1 else 0.,
        fallback_count=sum(reasons.values()),fallback_rate=sum(reasons.values())/n,
        solver_failures=reasons['solver_failure'],
        cycle_p50_ms=float(np.quantile(cycles,.5)*1000),cycle_p95_ms=float(np.quantile(cycles,.95)*1000),
        timeout_rate=float(np.mean(cycles>dt)),inference_p50_ms=float(np.median(inference)*1000),
        min_gap_m=float(min(gap.min(),future_gap[-1])),final_gap_m=float(future_gap[-1]),
        final_speed_mps=float(observed_future[-1]))
    return result,cycles,reasons


def expected_identities(config):
    preset=config['preset'];count=dict(smoke=2,standard=10,full=60)[preset]
    budgets=('equal','time') if config['budget']=='both' else (config['budget'],)
    expected=set()
    for seed in config['seeds']:
        for budget in budgets:
            names=(*MODELS,'ensemble_risk',*(['acc'] if budget=='equal' else []))
            for fi,family in enumerate(FAMILIES):
                for pair in range(count):
                    scenario=900000+fi*1000+pair+(seed*10000 if preset=='standard' else 0)
                    for condition in ('id','ood'):
                        expected.update((seed,budget,condition,family,scenario,name) for name in names)
    return expected


def identity(row):return (int(row['training_seed']),row['budget'],row['condition'],row['family'],int(row['scenario_seed']),row['model'])


def logfile(run,key):
    seed,budget,condition,family,scenario,name=key
    preset=json_read(run/'run_config.json')['preset']
    pair=scenario-900000-FAMILIES.index(family)*1000-(seed*10000 if preset=='standard' else 0)
    return run/f'seed{seed}'/'trajectories'/f'{budget}_{seed}_{condition}_{family}_{pair}_{name}.csv'


def audit(run,output=None):
    run=run.resolve();output=(output or run/'independent_audit').resolve();output.mkdir(parents=True,exist_ok=True)
    config=json_read(run/'run_config.json');expected=expected_identities(config)
    errors=[];missing_artifacts=[];interrupted=[];rows=[];current_source_differences=[]
    def fail(kind,**detail):errors.append(dict(kind=kind,**detail))
    if len(set(config['seeds']))!=dict(smoke=1,standard=3,full=5)[config['preset']]:
        fail('preset_training_seed_count_mismatch')
    for file in sorted(run.glob('seed*/closed_*.jsonl')):
        lines=file.read_text(encoding='utf-8').splitlines()
        for index,line in enumerate(lines):
            if not line.strip():continue
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1:interrupted.append(str(file));continue
                raise ValueError(f'Interior JSONL corruption: {file}:{index+1}')
            if file.parent.name!=f"seed{row['training_seed']}" or file.name!=f"closed_{row['budget']}_seed{row['training_seed']}.jsonl":
                fail('row_file_identity_mismatch',path=str(file),identity=identity(row))
            rows.append(row)
    counts=Counter(identity(row) for row in rows);actual=set(counts)
    duplicates=[list(k) for k,v in sorted(counts.items()) if v!=1]
    source_manifest=json_read(run/'source_sha256.json')
    source_by_name={Path(path.replace('\\','/')).name:digest for path,digest in source_manifest.items()}
    if set(source_by_name)!={'run_experiments.py','__init__.py','data.py','models.py','planning.py','experiments.py'}:
        fail('source_manifest_set_mismatch')
    missing_sources=[]
    for relative,digest in source_manifest.items():
        portable=Path(relative.replace('\\','/'))
        archived=run/'source_snapshot'/portable
        if not archived.exists():missing_artifacts.append(str(archived));missing_sources.append(str(archived))
        elif sha256(archived)!=digest:fail('source_snapshot_hash_mismatch',path=str(archived))
        current=ROOT/portable
        if not current.exists() or sha256(current)!=digest:current_source_differences.append(relative)
    budgets=('equal','time') if config['budget']=='both' else (config['budget'],)
    configurations={};artifact_checks=0
    for seed in config['seeds']:
        folder=run/f'seed{seed}'
        for budget in budgets:
            path=folder/f'closed_{budget}_config.json'
            if not path.exists():missing_artifacts.append(str(path));continue
            frozen=json_read(path);configurations[seed,budget]=frozen
            if frozen['preset']!=config['preset'] or frozen['budget']!=budget:
                fail('closed_config_mismatch',path=str(path))
            if frozen['steps']!=dict(smoke=80,standard=100,full=150)[config['preset']] or frozen['count']!=dict(smoke=2,standard=10,full=60)[config['preset']]:
                fail('closed_preset_dimensions_mismatch',path=str(path))
            if set(frozen['models'])!={name+'.npz' for name in MODELS}:
                fail('model_set_mismatch',path=str(path))
            for name,digest in frozen['models'].items():
                model=folder/'models'/name
                if not model.exists():missing_artifacts.append(str(model))
                elif sha256(model)!=digest:fail('model_hash_mismatch',path=str(model))
                else:artifact_checks+=1
            for name,digest in frozen['source_sha256'].items():
                if source_by_name.get(name)!=digest:fail('closed_source_hash_mismatch',path=str(path),source=name)
            if set(frozen['source_sha256'])!={'experiments.py','planning.py','data.py','models.py'}:
                fail('closed_source_set_mismatch',path=str(path))
            if budget=='time':
                profile_path=folder/'validation_profile.json'
                if not profile_path.exists():missing_artifacts.append(str(profile_path))
                else:
                    profile={r['model']:int(r['matched_candidates']) for r in json_read(profile_path)}
                    if frozen['matched']!=profile:fail('matched_count_profile_mismatch',path=str(path))
    residuals=defaultdict(float);pooled=defaultdict(list);reason_totals=defaultdict(Counter);audited=[];trajectory_hashes={}
    pair_observations={}
    for row in rows:
        key=identity(row);seed,budget,condition,family,scenario,name=key
        frozen=configurations.get((seed,budget))
        if frozen is None:continue
        path=logfile(run,key)
        if not path.exists():missing_artifacts.append(str(path));continue
        try:
            log=read_csv(path);recomputed,cycles,reasons=reconstruct(log,float(config['dt']),frozen['steps'])
        except (ValueError,KeyError,AssertionError) as error:
            fail('trajectory_reconstruction_failed',identity=list(key),detail=str(error));continue
        pairing=seed,condition,family,scenario
        initial=np.array([float(log[0][c]) for c in ('gap_m','speed_mps','lead_mps','actuator_mps2')])
        lead_trace=np.array([float(point['lead_mps']) for point in log])
        if pairing in pair_observations:
            initial_reference,lead_reference=pair_observations[pairing]
            common=min(len(lead_trace),len(lead_reference))
            if not np.allclose(initial,initial_reference,rtol=0,atol=1e-9):fail('unpaired_initial_observation',identity=list(key))
            if not np.allclose(lead_trace[:common],lead_reference[:common],rtol=0,atol=1e-9):fail('unpaired_exogenous_leader',identity=list(key))
            if len(lead_trace)>len(lead_reference):pair_observations[pairing]=(initial_reference,lead_trace)
        else:pair_observations[pairing]=initial,lead_trace
        for metric,value in recomputed.items():
            if metric not in row:continue
            error=abs(float(row[metric])-value);residuals[metric]=max(residuals[metric],error)
            if not np.isclose(float(row[metric]),value,rtol=1e-9,atol=1e-8):
                fail('raw_metric_mismatch',identity=list(key),metric=metric,recorded=row[metric],recomputed=value)
        expected_candidates=64 if budget=='equal' else frozen['matched'][name.removesuffix('_risk')]
        if int(row['candidates'])!=expected_candidates:fail('raw_candidate_count_mismatch',identity=list(key))
        if row['horizon']!=30 or row['iterations']!=3:fail('raw_solver_dimensions_mismatch',identity=list(key))
        if row['benchmark_workers']!=frozen['workers']:fail('workers_mismatch',identity=list(key))
        if float(row['risk_weight'])!=(1. if name.endswith('_risk') else 0.):fail('risk_weight_mismatch',identity=list(key))
        if row['observation_mode']!='privileged':fail('unexpected_observation_mode',identity=list(key))
        if name!='acc':
            for step,point in enumerate(log):
                if any(point.get(field,'')=='' for field in ('candidates','horizon','iterations','dt')):
                    fail('missing_logged_solver_dimensions',identity=list(key),step=step);break
                if (int(point['candidates'])!=expected_candidates or int(point['horizon'])!=30
                        or int(point['iterations'])!=3 or abs(float(point['dt'])-float(config['dt']))>1e-12):
                    fail('actual_solver_dimensions_mismatch',identity=list(key),step=step);break
        group=name,condition,budget
        pooled[group].extend(cycles);reason_totals[group].update(reasons)
        audited.append(dict(training_seed=seed,budget=budget,condition=condition,family=family,scenario_seed=scenario,
                            model=name,**recomputed))
        trajectory_hashes[str(path.relative_to(run))]=sha256(path)
    latency=[dict(model=k[0],condition=k[1],budget=k[2],cycles=len(v),p50_ms=float(np.quantile(v,.5)*1000),
                  p95_ms=float(np.quantile(v,.95)*1000),timeout_rate=float(np.mean(np.asarray(v)>float(config['dt']))),
                  fallback_rate=sum(reason_totals[k].values())/len(v),fallback_reasons=dict(reason_totals[k]))
             for k,v in sorted(pooled.items())]
    fallback_rows=[]
    for k,values in sorted(pooled.items()):
        names={'emergency_gap','deadline_exceeded','solver_failure','model_or_solver_exception','invalid_observation'}|set(reason_totals[k])
        for reason in sorted(names):
            fallback_rows.append(dict(model=k[0],condition=k[1],budget=k[2],reason=reason,
                trigger_count=reason_totals[k][reason],logged_cycles=len(values),
                trigger_rate_per_cycle=reason_totals[k][reason]/len(values)))
    # Check final pooled table only when rows are complete; while appending it may
    # legitimately describe an earlier consistent snapshot.
    raw_complete=actual==expected and not duplicates and not interrupted
    table=run/'latency_summary.csv'
    if raw_complete and table.exists():
        reported={(r['model'],r['condition'],r['budget']):r for r in read_csv(table)}
        if set(reported)!=set(pooled):fail('pooled_latency_groups_mismatch')
        for r in latency:
            other=reported.get((r['model'],r['condition'],r['budget']))
            if other:
                for metric in ('cycles','p50_ms','p95_ms','timeout_rate'):
                    if not np.isclose(float(other[metric]),r[metric],rtol=1e-9,atol=1e-8):
                        fail('pooled_latency_mismatch',model=r['model'],condition=r['condition'],budget=r['budget'],metric=metric)
    if duplicates:fail('duplicate_episode_identities',identities=duplicates)
    if actual-expected:fail('unexpected_episode_identities',identities=[list(k) for k in sorted(actual-expected)])
    complete=raw_complete and not missing_artifacts and len(audited)==len(expected) and not errors
    result=dict(status='verified_complete' if complete else ('failed' if errors else 'provisional_incomplete'),
        expected_episodes=len(expected),raw_episodes=len(rows),unique_episodes=len(actual),audited_episodes=len(audited),
        missing_episodes=[list(k) for k in sorted(expected-actual)],duplicates=duplicates,
        missing_artifacts=sorted(set(missing_artifacts)),interrupted_jsonl_tails=interrupted,
        error_count=len(errors),errors=errors,max_abs_metric_discrepancy=dict(residuals),
        model_hash_checks=artifact_checks,source_snapshot_hashes_verified=not missing_sources and not any('source' in e['kind'] for e in errors),
        current_source_differences=current_source_differences,
        independence='No environment/model/planning/metric helper imports; CSV Euler identities and velocity differences only',
        limits='Last post-step gap/speed reconstructed from logged pre-step state and declared Euler semantics; command execution/vehicle I/O not independently observed',
        acc_candidate_scope='ACC has no candidate optimization; raw row candidate count is benchmark bookkeeping, excluded from per-cycle workload check',
        audit_source_sha256=sha256(Path(__file__)),raw_jsonl_sha256={str(p.relative_to(run)):sha256(p) for p in sorted(run.glob('seed*/closed_*.jsonl'))},
        trajectory_sha256=trajectory_hashes,pooled_latency=latency,fallback_reasons=fallback_rows)
    for name,records in (('episode_metrics.csv',audited),('pooled_latency.csv',latency),('fallback_reasons.csv',fallback_rows)):
        if records:
            with (output/name).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/smoke_verified')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run,args.output)
    print(json.dumps({k:result[k] for k in ('status','expected_episodes','raw_episodes','audited_episodes','error_count','max_abs_metric_discrepancy')},ensure_ascii=False))
    if result['status']=='failed':raise SystemExit(1)
    if result['status']!='verified_complete':raise SystemExit(2)


if __name__=='__main__':main()
