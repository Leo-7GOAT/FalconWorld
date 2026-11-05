"""Independent PPO artifact/budget/raw-evaluation audit; never selects a policy.

No FalconWorld training, environment or evaluation implementation is imported.
Torch is used only to inspect trusted local checkpoint storage and Adam counters.
Exit 0: complete and verified; 1: invalid artifact; 2: incomplete experiment.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('ppo','wm_ppo','physics_ppo','real_reuse_control','wm_no_history','wm_long','ppo_total_budget')
MODEL_METHODS={'wm_ppo','physics_ppo','wm_no_history','wm_long'}
WORLD_MODELS={'ensemble','history','learned_lead','neural','physical_ca','physical_cv','physical_enriched','residual'}
FAMILIES=('constant','mild','brake','stopgo')
PRESETS=dict(smoke=dict(seeds=[0],steps=4096,offline_train=2,offline_val=1),
    standard=dict(seeds=[0,1,2],steps=32768,offline_train=12,offline_val=3),
    full=dict(seeds=[0,1,2,3,4],steps=131072,offline_train=24,offline_val=6))


def load_json(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def bool_value(value):
    if value in (True,'True','true',1,'1'):return True
    if value in (False,'False','false',0,'0'):return False
    raise ValueError('Invalid boolean trace field')


def expected_updates(steps,method):
    factor=8 if method=='real_reuse_control' else 4
    return sum(factor*((min(512,steps-start)+127)//128) for start in range(0,steps,512))


def case_identities(seed,preset):
    count=2 if preset=='smoke' else 10
    return {(family,condition,8_000_000+seed*100_000+ci*10_000+fi*1000+i)
            for ci,condition in enumerate(('id','ood')) for fi,family in enumerate(FAMILIES) for i in range(count)}


def initial_state(seed,condition):
    """Independent reconstruction of the documented episode initialization only."""
    rng=np.random.default_rng(seed)
    lead=rng.uniform(6,13) if condition=='id' else rng.uniform(14,19)
    gap=rng.uniform(22,48);ego=max(0.,lead+rng.uniform(-2,2))
    return np.array([gap,ego,lead,0.])


def raw_reward(gap,speed,previous_command,command):
    error=np.clip(gap-(4.+1.2*speed),-50.,50.)
    speed_error=np.clip(speed-12.,-20.,20.)
    reserve=np.clip(np.maximum(0.,2.+.5*speed-gap),0.,20.)
    cost=(.01*error**2+.1*speed_error**2+.01*command**2+
          .1*(command-previous_command)**2+100.*(gap<=0)+10.*reserve**2)
    return np.where(gap<=0,-100.,1.-np.minimum(2.,cost/10.))


def reconstruct_episode(trace,seed,condition):
    if not trace or len(trace)>120:raise ValueError('Invalid trajectory length')
    keys=('step','time_s','gap_m','speed_mps','lead_speed_mps','action_index','applied_command_mps2',
          'reward','measured_accel_mps2','jerk_mps3','gap_error_m','policy_ms','cycle_ms')
    arrays={k:np.asarray([float(r[k]) for r in trace]) for k in keys}
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite trace')
    n=len(trace);dt=.1;s0=initial_state(seed,condition)
    np.testing.assert_array_equal(arrays['step'],np.arange(n))
    np.testing.assert_allclose(arrays['time_s'],np.arange(1,n+1)*dt,rtol=0,atol=1e-10)
    gap,speed,lead,command=(arrays[k] for k in ('gap_m','speed_mps','lead_speed_mps','applied_command_mps2'))
    if np.any(speed<0) or np.any(lead<0) or np.any(command < -6.-1e-10) or np.any(command>2.+1e-10):raise ValueError('State/action bound violation')
    if np.any(arrays['action_index']!=np.floor(arrays['action_index'])) or np.any(arrays['action_index']<0) or np.any(arrays['action_index']>4):
        raise ValueError('Invalid discrete action')
    old_gap=np.r_[s0[0],gap[:-1]];old_speed=np.r_[s0[1],speed[:-1]];old_lead=np.r_[s0[2],lead[:-1]]
    np.testing.assert_allclose(gap,old_gap+dt*(old_lead-old_speed),rtol=0,atol=1e-9)
    accel=(speed-old_speed)/dt;jerk=np.diff(np.r_[0.,accel])/dt
    np.testing.assert_allclose(arrays['measured_accel_mps2'],accel,rtol=0,atol=1e-8)
    np.testing.assert_allclose(arrays['jerk_mps3'],jerk,rtol=0,atol=1e-7)
    if abs(accel[0])>1e-9:raise ValueError('First-step acceleration violates zero actuator initialization')
    errors=gap-(4.+1.2*speed)
    np.testing.assert_allclose(arrays['gap_error_m'],errors,rtol=0,atol=1e-9)
    rewards=raw_reward(gap,speed,np.r_[0.,command[:-1]],command)
    # Directly check every evaluation reward is raw, not multiplied by training .01.
    np.testing.assert_allclose(arrays['reward'],rewards,rtol=0,atol=1e-8,err_msg='Evaluation rewards must use raw units')
    terminated=np.asarray([bool_value(r['terminated']) for r in trace]);truncated=np.asarray([bool_value(r['truncated']) for r in trace])
    if np.any(terminated[:-1]|truncated[:-1]) or np.any(gap[:-1]<=0):raise ValueError('Trace continues after episode end')
    collision=int(gap[-1]<=0)
    if bool(terminated[-1])!=bool(collision) or bool(truncated[-1])!=bool(n==120 and not collision):raise ValueError('Invalid end masks')
    if not (terminated[-1] or truncated[-1]):raise ValueError('Unfinished episode trace')
    progress=float(old_speed.sum()*dt);reference=float(old_lead.sum()*dt)
    cycles,policy=arrays['cycle_ms'],arrays['policy_ms']
    if np.any(policy<0) or np.any(cycles<policy):raise ValueError('Invalid timing intervals')
    reasons=Counter(r['fallback'] for r in trace if r['fallback'])
    return dict(steps=n,collision=collision,completed=int(not collision and n==120 and progress>=.5*max(5.,reference)),
        return_value=float(rewards.sum()),progress_m=progress,reference_progress_m=reference,
        gap_rmse_m=float(np.sqrt(np.mean(errors**2))),jerk_rms_mps3=float(np.sqrt(np.mean(jerk**2))),
        fallback_count=sum(reasons.values()),fallback_reasons=dict(reasons),
        policy_p50_ms=float(np.quantile(policy,.5)),policy_p95_ms=float(np.quantile(policy,.95)),
        cycle_p50_ms=float(np.quantile(cycles,.5)),cycle_p95_ms=float(np.quantile(cycles,.95)),
        timeout_rate=float(np.mean(cycles>100))),cycles,lead


def audit(run,output):
    run=run.resolve();output.mkdir(parents=True,exist_ok=True)
    issues=[];missing=[];tails=[];artifacts={};max_error=defaultdict(float)
    def fail(kind,**detail):issues.append(dict(kind=kind,**detail))
    def require(path):
        if not path.exists():missing.append(str(path));return False
        return True
    registration=load_json(run/'registration.json');preset=registration['preset'];spec=PRESETS[preset]
    if registration['spec']!=spec:fail('preset_spec_mismatch')
    if set(registration['methods'])!=set(METHODS) or len(registration['methods'])!=len(METHODS):fail('registered_method_set_mismatch')
    if registration.get('learning_reward_scale')!=.01:fail('learning_reward_scale_mismatch')
    source_names={'run_rl.py','rl_env.py','ppo_agent.py','rl_evaluation.py','data.py','models.py','planning.py'}
    source_by_name={Path(p.replace('\\','/')).name:h for p,h in registration['sources'].items()}
    if set(source_by_name)!=source_names:fail('source_snapshot_set_mismatch')
    current_differences=[]
    for relative,digest in registration['sources'].items():
        portable=Path(relative.replace('\\','/'));snapshot=run/'source_snapshot'/portable
        if require(snapshot):
            actual=sha(snapshot);artifacts[str(snapshot.relative_to(run))]=actual
            if actual!=digest:fail('snapshot_hash_mismatch',path=str(snapshot))
        if not (ROOT/portable).exists() or sha(ROOT/portable)!=digest:current_differences.append(relative)
    runner=run/'source_snapshot/run_rl.py'
    if runner.exists():
        tree=ast.parse(runner.read_text(encoding='utf-8'));scale=None;scaled_functions=set()
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LEARNING_REWARD_SCALE' for t in node.targets):scale=ast.literal_eval(node.value)
            if isinstance(node,ast.FunctionDef) and node.name in ('real_rollout','model_rollout'):
                for child in ast.walk(node):
                    if isinstance(child,ast.keyword) and child.arg=='rewards' and isinstance(child.value,ast.BinOp) and isinstance(child.value.op,ast.Mult):
                        names={part.id for part in ast.walk(child.value) if isinstance(part,ast.Name)}
                        if names=={'reward','LEARNING_REWARD_SCALE'}:scaled_functions.add(node.name)
        if scale!=.01 or scaled_functions!={'real_rollout','model_rollout'}:fail('train_model_real_scaling_source_mismatch')
    common_path=run/'common_data/manifest.json';common=None;common_count=None
    if require(common_path):
        common=load_json(common_path)
        if hashlib.sha256(json.dumps(common['files'],sort_keys=True).encode()).hexdigest()!=common['sha256']:fail('common_manifest_digest_mismatch')
        split_episodes={}
        for split,label in (('train','training_transitions'),('val','validation_transitions')):
            file=run/'common_data'/f'{split}.npz'
            if require(file):
                digest=sha(file);artifacts[str(file.relative_to(run))]=digest
                if digest!=common['files'][split]:fail('common_npz_digest_mismatch',split=split)
                with np.load(file,allow_pickle=False) as arrays:
                    if len(arrays['state'])!=common[label]:fail('common_transition_count_mismatch',split=split)
                    # The collector's NPZ episode IDs are split-local counters.
                    # Actual scenario identities use the manifest seed+family;
                    # train starts at 3,000,000, val at 3,100,000.
                    entries=[r for r in common['episodes'] if r['split']==split]
                    split_episodes[split]={(r['seed'],r['family']) for r in entries}
                    if len(np.unique(arrays['episode']))!=len(entries) or sum(r['steps'] for r in entries)!=len(arrays['state']):
                        fail('common_episode_manifest_count_mismatch',split=split)
        if split_episodes.get('train',set())&split_episodes.get('val',set()):fail('common_episode_split_leakage')
        common_count=common['training_transitions']+common['validation_transitions']
    model_checks=0
    for seed in spec['seeds']:
        folder=run/f'seed{seed}/world_models';frozen=folder/'frozen_sha256.json'
        if require(frozen):
            hashes=load_json(frozen)
            if set(hashes)!={n+'.npz' for n in WORLD_MODELS}:fail('world_model_set_mismatch',seed=seed)
            for name,digest in hashes.items():
                file=folder/name
                if require(file):
                    actual=sha(file);artifacts[str(file.relative_to(run))]=actual
                    if actual!=digest:fail('world_model_hash_mismatch',path=str(file))
                    model_checks+=1
        log=folder/'training.json'
        if require(log) and common is not None and load_json(log)['data_sha256']!=common['sha256']:fail('world_model_data_provenance_mismatch',seed=seed)
    expected_jobs={(seed,method) for seed in spec['seeds'] for method in METHODS}
    expected={(method,seed,*case) for seed,method in expected_jobs for case in case_identities(seed,preset)}
    actual=[];audited=[];job_ledgers={};completed=[];pooled=defaultdict(list);paired_lead={}
    # Torch only inspects trusted checkpoint dicts; no PPOAgent is imported.
    import torch
    for seed,method in sorted(expected_jobs):
        folder=run/f'seed{seed}'/method;complete_path=folder/'complete.json'
        ledger=load_json(complete_path) if require(complete_path) else None
        if ledger is not None:
            job_ledgers[seed,method]=ledger;completed.append((seed,method))
            if ledger['method']!=method or ledger['training_seed']!=seed or ledger['status']!='verified_complete':fail('job_ledger_identity',job=[seed,method])
        budget=spec['steps']+(common_count if method=='ppo_total_budget' and common_count is not None else 0)
        updates=expected_updates(budget,method);has_model=method in MODEL_METHODS;horizon=3 if method=='wm_long' else int(has_model)
        if ledger is not None and common_count is not None:
            checks=dict(new_real_steps=budget,policy_updates=updates,rollout_horizon=horizon,common_offline_transitions=common_count,
                native_total_training_interactions=budget+(common_count if has_model else 0),shared_data_budget_interactions=budget+common_count,
                evaluation_episodes=len(case_identities(seed,preset)))
            for key,value in checks.items():
                if ledger[key]!=value:fail('training_budget_mismatch',job=[seed,method],field=key,recorded=ledger[key],expected=value)
            generated=ledger['synthetic_generated']
            if ((not has_model and generated!=0) or (horizon==1 and generated!=budget)
                    or (horizon==3 and not budget<generated<=3*budget)):fail('synthetic_generation_count_mismatch',job=[seed,method])
        checkpoint=folder/'final_policy.pt'
        if require(checkpoint):
            digest=sha(checkpoint);artifacts[str(checkpoint.relative_to(run))]=digest
            if ledger and digest!=ledger['checkpoint_sha256']:fail('checkpoint_hash_mismatch',job=[seed,method])
            stored=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if stored.get('format')!='falconworld.categorical_ppo':fail('checkpoint_algorithm_mismatch',job=[seed,method])
            if stored['gradient_updates']!=updates or stored['policy_version']!=(budget+511)//512:fail('checkpoint_update_budget_mismatch',job=[seed,method])
            if stored['metadata']!=dict(method=method,training_seed=seed,new_real_steps=budget):fail('checkpoint_metadata_budget_mismatch',job=[seed,method])
            adam_steps={int(v['step'].item()) for v in stored['optimizer']['state'].values()}
            if adam_steps!={updates}:fail('adam_update_counter_mismatch',job=[seed,method],actual=sorted(adam_steps),expected=updates)
            if stored['config']['obs_dim']!=25 or stored['config']['n_actions']!=5 or stored['config']['seed']!=seed:fail('checkpoint_network_config_mismatch',job=[seed,method])
        curve=folder/'learning_curve.json'
        if require(curve):
            records=load_json(curve)
            if ledger is not None and (not records or records[-1]['new_real_steps']!=budget or records[-1]['policy_updates']!=updates):
                fail('learning_curve_final_budget_mismatch',job=[seed,method])
            if records and records[-1]['new_real_steps']>budget:fail('learning_curve_exceeds_budget',job=[seed,method])
            if ledger is not None and records and records[-1]['synthetic_generated']!=ledger['synthetic_generated']:
                fail('curve_ledger_synthetic_disagreement',job=[seed,method])
            for r in records:
                step=r['new_real_steps'];expected_u=expected_updates(step,method)
                # Full chunks are 512; a non-multiple final chunk uses the remainder.
                minibatches=(8 if method=='real_reuse_control' else 4)*(((step%512 or 512)+127)//128)
                if r['policy_updates']!=expected_u or r['update_gradient_updates']!=expected_u or r['update_minibatches']!=minibatches or r['update_batch_size']!=128:
                    fail('learning_curve_update_accounting_mismatch',job=[seed,method],step=step)
                if r['update_synthetic_samples']!=minibatches*(64 if has_model else 0) or r['update_real_samples']!=minibatches*(64 if has_model else 128):
                    fail('learning_curve_batch_mix_mismatch',job=[seed,method],step=step)
                generated=r['synthetic_generated']
                if ((not has_model and generated!=0) or (horizon==1 and generated!=step)
                        or (horizon==3 and not step<generated<=3*step)):
                    fail('curve_synthetic_count_mismatch',job=[seed,method],step=step)
        manifest_path=folder/'test/manifest.json'
        if require(manifest_path):
            manifest=load_json(manifest_path)
            keys={(r['family'],r['condition'],int(r['seed'])) for r in manifest['cases']}
            if keys!=case_identities(seed,preset) or len(manifest['cases'])!=len(keys):fail('test_manifest_cases_mismatch',job=[seed,method])
            if manifest['method']!=method or manifest['training_seed']!=seed:fail('test_manifest_identity',job=[seed,method])
            for name,digest in manifest['source_sha256'].items():
                if source_by_name.get(name)!=digest:fail('test_manifest_source_mismatch',job=[seed,method],source=name)
        test_path=folder/'test/episodes.jsonl';job_rows=[]
        if not require(test_path):continue
        digest=sha(test_path);artifacts[str(test_path.relative_to(run))]=digest
        if ledger and digest!=ledger['test_sha256']:fail('test_jsonl_hash_mismatch',job=[seed,method])
        lines=test_path.read_text(encoding='utf-8').splitlines()
        for index,line in enumerate(lines):
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1:tails.append(str(test_path));continue
                fail('interior_test_jsonl_corruption',job=[seed,method],line=index+1);continue
            key=(r['method'],r['training_seed'],r['family'],r['condition'],r['scenario_seed']);actual.append(key);job_rows.append(r)
            if r['method']!=method or r['training_seed']!=seed:fail('test_episode_job_mismatch',key=key)
            file=folder/'test'/f"trace_{r['condition']}_{r['family']}_{r['scenario_seed']}.csv"
            if not require(file):continue
            try:metrics,cycles,lead=reconstruct_episode(load_csv(file),r['scenario_seed'],r['condition'])
            except (ValueError,KeyError,AssertionError) as error:
                fail('trace_reconstruction_failure',key=key,detail=str(error));continue
            for field,value in metrics.items():
                if field=='fallback_reasons':
                    if json.loads(r[field])!=value:fail('fallback_reason_mismatch',key=key)
                    continue
                target='return' if field=='return_value' else field
                difference=abs(float(r[target])-value);max_error[target]=max(max_error[target],difference)
                if not np.isclose(r[target],value,rtol=1e-9,atol=1e-7):fail('episode_metric_mismatch',key=key,field=target,recorded=r[target],recomputed=value)
            np.testing.assert_allclose(r['cycle_ms_samples'],cycles,rtol=0,atol=1e-9)
            pair=(seed,r['family'],r['condition'],r['scenario_seed'])
            if pair in paired_lead:
                reference=paired_lead[pair];n=min(len(reference),len(lead))
                if not np.allclose(reference[:n],lead[:n],rtol=0,atol=1e-9):fail('unpaired_exogenous_leader_trace',key=key)
                if len(lead)>len(reference):paired_lead[pair]=lead
            else:paired_lead[pair]=lead
            audited.append(dict(method=method,training_seed=seed,family=r['family'],condition=r['condition'],scenario_seed=r['scenario_seed'],**metrics))
            pooled[method,r['condition']].extend(cycles)
        if ledger is not None and len(job_rows)==len(case_identities(seed,preset)):
            if ledger['test_steps']!=sum(r['steps'] for r in job_rows):fail('test_step_ledger_mismatch',job=[seed,method])
    counts=Counter(actual);duplicates=[list(k) for k,v in counts.items() if v>1];actual_set=set(actual)
    if duplicates:fail('duplicate_test_identities',identities=duplicates)
    if actual_set-expected:fail('unexpected_test_identities',identities=[list(k) for k in sorted(actual_set-expected)])
    global_ledger=run/'cost_ledger.json'
    if require(global_ledger):
        listed=load_json(global_ledger);mapped={(r['training_seed'],r['method']):r for r in listed}
        if len(listed)!=len(mapped) or set(mapped)!=expected_jobs:fail('root_cost_ledger_job_set_mismatch')
        for key,r in mapped.items():
            if key in job_ledgers and r!=job_ledgers[key]:fail('root_job_ledger_disagreement',job=list(key))
    complete_path=run/'completeness.json'
    if require(complete_path):
        reported=load_json(complete_path)
        if reported['expected_episodes']!=len(expected) or reported['expected_jobs']!=len(expected_jobs):fail('root_completeness_expectation_mismatch')
        if reported.get('verified_complete') and (actual_set!=expected or len(completed)!=len(expected_jobs)):fail('false_complete_claim')
    complete=not missing and not tails and not issues and actual_set==expected and len(actual)==len(expected) and len(audited)==len(expected) and set(completed)==expected_jobs
    pooled_rows=[dict(method=k[0],condition=k[1],cycles=len(v),p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),timeout_rate=float(np.mean(np.asarray(v)>100))) for k,v in sorted(pooled.items())]
    result=dict(status='verified_complete' if complete else ('failed' if issues else 'provisional_incomplete'),
        expected_jobs=len(expected_jobs),completed_jobs=len(completed),expected_test_episodes=len(expected),
        raw_test_episodes=len(actual),unique_test_episodes=len(actual_set),audited_test_episodes=len(audited),
        missing_jobs=[list(k) for k in sorted(expected_jobs-set(completed))],
        missing_test_identities=[list(k) for k in sorted(expected-actual_set)],missing_artifacts=sorted(set(missing)),
        interrupted_jsonl_tails=tails,error_count=len(issues),errors=issues,max_abs_metric_difference=dict(max_error),
        world_model_hash_checks=model_checks,current_source_differences=current_differences,
        artifact_sha256=artifacts,audit_source_sha256=sha(Path(__file__)),pooled_latency=pooled_rows,
        raw_evaluation_reward_check='every trace reward independently recomputed in original task units; no .01 evaluation scaling',
        common_episode_identity_scope='NPZ episode counters are split-local; manifest scenario seed+family identities are disjoint',
        independence='No FalconWorld training/environment/evaluation imports; Torch deserializes checkpoint/Adam storage only',
        limits='Training step calls are not independently traced: counts cross-check registration, curve, ledger, checkpoint metadata and actual Adam step counters. No policy-performance selection.')
    (output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for name,rows in (('recomputed_test_episodes.csv',audited),('pooled_latency.csv',pooled_rows)):
        if rows:
            with (output/name).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/rl_ppo/smoke')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=(args.output or args.run/'independent_audit').resolve()
    try:result=audit(args.run,output)
    except (ValueError,KeyError,AssertionError,OSError,RuntimeError,TypeError,EOFError,SyntaxError,pickle.UnpicklingError) as error:
        output.mkdir(parents=True,exist_ok=True)
        result=dict(status='failed',error_count=1,errors=[dict(kind='audit_exception',detail=str(error))])
        (output/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','expected_jobs','completed_jobs','expected_test_episodes','audited_test_episodes','error_count') if k in result}))
    raise SystemExit(0 if result['status']=='verified_complete' else 1 if result['status']=='failed' else 2)


if __name__=='__main__':main()
