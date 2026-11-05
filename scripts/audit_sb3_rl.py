"""Independent SB3 research artifacts, Adam/update budgets and raw-trace audit.

Does not import the runner, environment, PPO subclass or a policy for inference.
Reuses only the independent post-step log reconstruction from audit_ppo.py.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from io import BytesIO
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import pickle
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_ppo import METHODS,MODEL_METHODS,WORLD_MODELS,case_identities,reconstruct_episode,load_csv,load_json,sha

PRESETS=dict(smoke=dict(seeds=[0],steps=8192,offline_train=2,offline_val=1),
    standard=dict(seeds=[0,1,2],steps=65536,offline_train=12,offline_val=3),
    full=dict(seeds=[0,1,2,3,4],steps=262144,offline_train=24,offline_val=6))


def budget_segments(steps):return [min(2048,steps-offset) for offset in range(0,steps,2048)]


def expected_budget(steps,method):
    segments=budget_segments(steps);epochs=20 if method=='real_reuse_control' else 10
    replaced=sum(round(size*.5) for size in segments) if method in MODEL_METHODS else 0
    return dict(segments=segments,epochs=epochs,optimizer_steps=sum(epochs*math.ceil(size/64) for size in segments),
                replaced=replaced,train_calls=len(segments),epoch_updates=epochs*len(segments))


def read_checkpoint(path):
    import torch
    with zipfile.ZipFile(path) as archive:
        data=json.loads(archive.read('data'))
        optimizer=torch.load(BytesIO(archive.read('policy.optimizer.pth')),map_location='cpu',weights_only=False)
        library_version=archive.read('_stable_baselines3_version').decode().strip()
    counters={int(value['step'].item()) for value in optimizer['state'].values() if 'step' in value}
    return data,counters,library_version


def audit(run,output):
    run=run.resolve();output.mkdir(parents=True,exist_ok=True)
    errors=[];missing=[];tails=[];artifacts={};discrepancy=defaultdict(float)
    def fail(kind,**details):errors.append(dict(kind=kind,**details))
    def require(path):
        if not path.exists():missing.append(str(path));return False
        return True
    registration=load_json(run/'registration.json');preset=registration['preset'];spec=PRESETS[preset]
    if registration['spec']!=spec:fail('registered_preset_mismatch')
    if set(registration['methods'])!=set(METHODS) or len(registration['methods'])!=7:fail('registered_methods_mismatch')
    if registration.get('learning_reward_scale')!=.01:fail('registered_reward_scale_mismatch')
    if registration.get('algorithm')!='SB3 default PPO with fresh frozen-policy model branches':fail('wrong_research_run')
    names={'run_sb3_rl.py','run_rl.py','rl_gym.py','sb3_worldmodel_ppo.py','rl_env.py','rl_evaluation.py','ppo_agent.py','models.py','data.py','planning.py'}
    by_name={Path(p.replace('\\','/')).name:d for p,d in registration['sources'].items()}
    if set(by_name)!=names:fail('project_source_set_mismatch')
    current_differences=[]
    for relative,digest in registration['sources'].items():
        portable=Path(relative.replace('\\','/'));snapshot=run/'source_snapshot'/portable
        if require(snapshot):
            actual=sha(snapshot);artifacts[str(snapshot.relative_to(run))]=actual
            if actual!=digest:fail('project_snapshot_sha_mismatch',path=str(snapshot))
        if not (ROOT/portable).exists() or sha(ROOT/portable)!=digest:current_differences.append(relative)
    # Library-version evidence is separate from the original project snapshots.
    distribution=package_metadata.distribution('stable_baselines3')
    installed_sb3_version=distribution.version
    library_sources={}
    for relative in ('stable_baselines3/ppo/ppo.py','stable_baselines3/common/buffers.py'):
        path=Path(distribution.locate_file(relative));library_sources[str(path)]=sha(path)
    versions=registration['versions']
    if installed_sb3_version!=versions['stable_baselines3']:fail('installed_sb3_version_mismatch')
    common=None;offline=None;manifest_path=run/'common_data/manifest.json'
    if require(manifest_path):
        common=load_json(manifest_path)
        if hashlib.sha256(json.dumps(common['files'],sort_keys=True).encode()).hexdigest()!=common['sha256']:fail('common_manifest_sha_mismatch')
        scenario_ids={}
        for split,count_field in (('train','training_transitions'),('val','validation_transitions')):
            path=run/'common_data'/f'{split}.npz'
            if require(path):
                digest=sha(path);artifacts[str(path.relative_to(run))]=digest
                if digest!=common['files'][split]:fail('common_npz_sha_mismatch',split=split)
                with np.load(path,allow_pickle=False) as data:
                    entries=[r for r in common['episodes'] if r['split']==split]
                    scenario_ids[split]={(r['seed'],r['family']) for r in entries}
                    if len(data['state'])!=common[count_field] or sum(r['steps'] for r in entries)!=len(data['state']):fail('common_count_mismatch',split=split)
                    if len(np.unique(data['episode']))!=len(entries):fail('common_episode_manifest_mismatch',split=split)
        if scenario_ids.get('train',set())&scenario_ids.get('val',set()):fail('common_scenario_split_leakage')
        offline=common['training_transitions']+common['validation_transitions']
    model_checks=0
    for seed in spec['seeds']:
        folder=run/f'seed{seed}/world_models';manifest=folder/'frozen_sha256.json'
        if require(manifest):
            digests=load_json(manifest)
            if set(digests)!={name+'.npz' for name in WORLD_MODELS}:fail('world_model_set_mismatch',seed=seed)
            for name,digest in digests.items():
                path=folder/name
                if require(path):
                    actual=sha(path);artifacts[str(path.relative_to(run))]=actual
                    if actual!=digest:fail('world_model_sha_mismatch',path=str(path))
                    model_checks+=1
        log=folder/'training.json'
        if require(log) and common is not None and load_json(log)['data_sha256']!=common['sha256']:fail('world_model_data_provenance',seed=seed)
    jobs={(seed,method) for seed in spec['seeds'] for method in METHODS}
    expected={(method,seed,*case) for seed,method in jobs for case in case_identities(seed,preset)}
    observed=[];audited=[];ledgers={};pooled=defaultdict(list);leader_pairs={};budget_rows=[];checkpoint_versions=set()
    for seed,method in sorted(jobs):
        folder=run/f'seed{seed}'/method;complete=folder/'complete.json'
        ledger=load_json(complete) if require(complete) else None
        if ledger:
            ledgers[seed,method]=ledger
            if (ledger['method'],ledger['training_seed'],ledger['status'])!=(method,seed,'verified_complete'):fail('job_identity_mismatch',job=[seed,method])
        steps=spec['steps']+(offline if method=='ppo_total_budget' and offline is not None else 0)
        budget=expected_budget(steps,method);has_model=method in MODEL_METHODS;last_size=budget['segments'][-1]
        expected_config=dict(n_steps=2048,batch_size=64,n_epochs=budget['epochs'],learning_rate=.0003,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5)
        configuration=folder/'policy_config.json'
        if require(configuration) and load_json(configuration)!=expected_config:fail('official_default_configuration_mismatch',job=[seed,method])
        if ledger is not None and offline is not None:
            required=dict(new_real_steps=steps,policy_updates=budget['optimizer_steps'],synthetic_replaced=budget['replaced'],
                common_offline_transitions=offline,native_total_training_interactions=steps+(offline if has_model else 0),
                shared_data_budget_interactions=steps+offline,evaluation_episodes=len(case_identities(seed,preset)),post_training_probe_resets=16)
            for key,value in required.items():
                if ledger[key]!=value:fail('ledger_budget_mismatch',job=[seed,method],field=key,actual=ledger[key],expected=value)
            if ledger['policy_config']!=expected_config:fail('ledger_configuration_mismatch',job=[seed,method])
            metadata=ledger['model_metadata']
            checks=dict(model_train_calls=budget['train_calls'],model_optimizer_steps=budget['optimizer_steps'],
                model_replaced=budget['replaced'],model_fraction=.5,model_horizon=3 if method=='wm_long' else 1,
                model_reward_scale=.01,model_seed=seed+4321)
            for key,value in checks.items():
                if metadata[key]!=value:fail('model_metadata_budget_mismatch',job=[seed,method],field=key)
            generated=ledger['synthetic_generated'];replaced=budget['replaced']
            valid_generation=(generated==0 if not has_model else generated==replaced if method!='wm_long' else replaced<=generated<=replaced+2*budget['train_calls'])
            if not valid_generation or generated!=metadata['model_generated']:fail('generated_count_mismatch',job=[seed,method])
            if metadata['model_generation_wall_s']!=ledger['model_generation_wall_s'] or ledger['model_generation_wall_s']<0:fail('model_time_accounting_mismatch',job=[seed,method])
            if not has_model and (metadata['model_branch_roots']!=0 or ledger['model_generation_wall_s']!=0):fail('no_model_has_generation',job=[seed,method])
            if has_model:
                last=metadata['last_model_metadata'];last_replaced=round(last_size*.5)
                if last['real_root_pool_size']!=last_size or last['replaced']!=last_replaced:fail('tail_replacement_mismatch',job=[seed,method])
                if last['actual_optimizer_steps']!=budget['epochs']*math.ceil(last_size/64):fail('tail_optimizer_budget_mismatch',job=[seed,method])
                if not last['episode_starts_preserved'] or max(last['max_behavior_logprob_error'],last['max_behavior_value_error'])>1e-5:
                    fail('frozen_behavior_probability_check_failed',job=[seed,method])
            budget_rows.append(dict(method=method,training_seed=seed,online_steps=steps,train_calls=budget['train_calls'],
                optimizer_steps=budget['optimizer_steps'],model_replaced=replaced,model_generated=generated,last_rollout_size=last_size))
        checkpoint=folder/'final_policy.zip'
        if require(checkpoint):
            digest=sha(checkpoint);artifacts[str(checkpoint.relative_to(run))]=digest
            if ledger and digest!=ledger['checkpoint_sha256']:fail('checkpoint_sha_mismatch',job=[seed,method])
            stored,adam_steps,version=read_checkpoint(checkpoint);checkpoint_versions.add(version)
            if version!=versions['stable_baselines3']:fail('checkpoint_sb3_version_mismatch',job=[seed,method])
            if adam_steps!={budget['optimizer_steps']}:fail('zip_adam_counter_mismatch',job=[seed,method],actual=sorted(adam_steps),expected=budget['optimizer_steps'])
            zip_checks=dict(num_timesteps=steps,n_steps=last_size,n_epochs=budget['epochs'],batch_size=64,
                _n_updates=budget['epoch_updates'],model_optimizer_steps=budget['optimizer_steps'],model_train_calls=budget['train_calls'],
                model_replaced=budget['replaced'],seed=seed)
            for key,value in zip_checks.items():
                if stored.get(key)!=value:fail('zip_training_counter_mismatch',job=[seed,method],field=key,actual=stored.get(key),expected=value)
            if any(key in stored for key in ('world_model','_model_rng','_model_torch_rng_state')):fail('external_model_pickled_into_policy',job=[seed,method])
            if stored.get('target_kl') is not None:fail('unregistered_kl_early_stopping',job=[seed,method])
            if ledger and (stored['model_generated']!=ledger['synthetic_generated'] or stored['model_generation_wall_s']!=ledger['model_generation_wall_s']):fail('zip_ledger_generation_mismatch',job=[seed,method])
        curve=folder/'learning_curve.json'
        if require(curve):
            records=load_json(curve)
            if ledger and (not records or records[-1]['new_real_steps']!=steps):fail('curve_final_online_budget_mismatch',job=[seed,method])
            for row in records:
                prefix=expected_budget(row['new_real_steps'],method)
                if row['policy_updates']!=prefix['optimizer_steps']:fail('curve_update_budget_mismatch',job=[seed,method],step=row['new_real_steps'])
            if ledger and records and records[-1]['synthetic_generated']!=ledger['synthetic_generated']:fail('curve_generation_ledger_mismatch',job=[seed,method])
        manifest=folder/'test/manifest.json'
        if require(manifest):
            data=load_json(manifest);keys={(r['family'],r['condition'],r['seed']) for r in data['cases']}
            if keys!=case_identities(seed,preset) or len(keys)!=len(data['cases']):fail('wrong_test_case_set',job=[seed,method])
            if data['method']!=method or data['training_seed']!=seed:fail('test_manifest_job_mismatch',job=[seed,method])
            for name,digest in data['source_sha256'].items():
                if by_name.get(name)!=digest:fail('test_manifest_source_mismatch',job=[seed,method],source=name)
        episodes=folder/'test/episodes.jsonl'
        if not require(episodes):continue
        digest=sha(episodes);artifacts[str(episodes.relative_to(run))]=digest
        if ledger and digest!=ledger['test_sha256']:fail('test_jsonl_sha_mismatch',job=[seed,method])
        lines=episodes.read_text(encoding='utf-8').splitlines();job_rows=[]
        for i,line in enumerate(lines):
            if not line.strip():continue
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i==len(lines)-1:tails.append(str(episodes));continue
                fail('interior_episode_jsonl_corruption',job=[seed,method],line=i+1);continue
            key=(row['method'],row['training_seed'],row['family'],row['condition'],row['scenario_seed']);observed.append(key);job_rows.append(row)
            if row['method']!=method or row['training_seed']!=seed:fail('episode_job_mismatch',key=key)
            trace=folder/'test'/f"trace_{row['condition']}_{row['family']}_{row['scenario_seed']}.csv"
            if not require(trace):continue
            try:metrics,cycles,lead=reconstruct_episode(load_csv(trace),row['scenario_seed'],row['condition'])
            except (ValueError,KeyError,AssertionError) as error:
                fail('independent_trace_reconstruction_failure',key=key,detail=str(error));continue
            for field,value in metrics.items():
                if field=='fallback_reasons':
                    if json.loads(row[field])!=value:fail('fallback_reason_mismatch',key=key)
                    continue
                target='return' if field=='return_value' else field
                difference=abs(row[target]-value);discrepancy[target]=max(discrepancy[target],difference)
                if not np.isclose(row[target],value,rtol=1e-9,atol=1e-7):fail('recomputed_metric_mismatch',key=key,field=target,actual=row[target],recomputed=value)
            if not np.allclose(row['cycle_ms_samples'],cycles,rtol=0,atol=1e-9):fail('cycle_samples_mismatch',key=key)
            pair=(seed,row['family'],row['condition'],row['scenario_seed'])
            if pair in leader_pairs:
                previous=leader_pairs[pair];n=min(len(lead),len(previous))
                if not np.allclose(lead[:n],previous[:n],rtol=0,atol=1e-9):fail('unpaired_exogenous_scenario',key=key)
                if len(lead)>len(previous):leader_pairs[pair]=lead
            else:leader_pairs[pair]=lead
            audited.append(dict(method=method,training_seed=seed,family=row['family'],condition=row['condition'],scenario_seed=row['scenario_seed'],**metrics))
            pooled[method,row['condition']].extend(cycles)
        if ledger and len(job_rows)==len(case_identities(seed,preset)) and ledger['test_steps']!=sum(r['steps'] for r in job_rows):fail('test_steps_ledger_mismatch',job=[seed,method])
    counts=Counter(observed);actual=set(observed);duplicates=[list(k) for k,v in counts.items() if v>1]
    if duplicates:fail('duplicate_test_identities',identities=duplicates)
    if actual-expected:fail('unexpected_test_identities',identities=[list(k) for k in sorted(actual-expected)])
    root_ledger=run/'cost_ledger.json'
    if require(root_ledger):
        listed=load_json(root_ledger);mapping={(r['training_seed'],r['method']):r for r in listed}
        if set(mapping)!=jobs or len(listed)!=len(mapping):fail('root_job_ledger_identity_mismatch')
        for key,value in mapping.items():
            if key in ledgers and value!=ledgers[key]:fail('root_per_job_ledger_disagreement',job=list(key))
    complete_path=run/'completeness.json'
    if require(complete_path):
        report=load_json(complete_path)
        if report['expected_jobs']!=len(jobs) or report['expected_episodes']!=len(expected):fail('wrong_root_expected_counts')
        if report.get('verified_complete') and (len(ledgers)!=len(jobs) or actual!=expected):fail('false_root_complete_claim')
    complete=not errors and not missing and not tails and set(ledgers)==jobs and actual==expected and len(observed)==len(expected) and len(audited)==len(expected)
    latency=[dict(method=k[0],condition=k[1],cycles=len(v),p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),timeout_rate=float(np.mean(np.asarray(v)>100))) for k,v in sorted(pooled.items())]
    result=dict(status='verified_complete' if complete else 'failed' if errors else 'provisional_incomplete',
        expected_jobs=len(jobs),completed_jobs=len(ledgers),expected_test_episodes=len(expected),raw_test_episodes=len(observed),audited_test_episodes=len(audited),
        missing_jobs=[list(k) for k in sorted(jobs-set(ledgers))],missing_test_identities=[list(k) for k in sorted(expected-actual)],
        missing_artifacts=sorted(set(missing)),interrupted_jsonl_tails=tails,error_count=len(errors),errors=errors,
        max_abs_metric_difference=dict(discrepancy),world_model_npz_hash_checks=model_checks,artifact_sha256=artifacts,
        project_source_snapshot_scope='10 project files in registration.sources, archived before training; each SHA verified',
        library_evidence=dict(registered_versions=versions,checkpoint_sb3_versions=sorted(checkpoint_versions),
            installed_sb3_version=installed_sb3_version,current_library_source_sha256=library_sources,
            limitation='Runner did not snapshot original SB3 pip library source at training start; current installed source hashes are audit-time evidence, not original training-source hashes'),
        current_project_source_differences=current_differences,pooled_latency=latency,
        audit_source_sha256=sha(Path(__file__)),independent_metric_helper_sha256=sha(ROOT/'scripts/audit_ppo.py'),
        independence='No runner/environment/PPO algorithm/subclass imported. Only audit_ppo pure-log reconstruction; package metadata locates library files; Torch reads trusted zip optimizer storage. No policy ranking or hyperparameter selection.',
        raw_reward_check='Every evaluation reward independently recomputed in raw task units; .01 training scale is never applied to evaluation.',
        limitations='No per-step training sample trace is saved. Online calls cross-check ledger, zip num_timesteps, curve and hook/Adam counts; individual model branch samples and per-round replacements cannot be independently recovered. Bounds and final metadata are checked.')
    (output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for name,rows in (('budget_checks.csv',budget_rows),('recomputed_test_episodes.csv',audited),('pooled_latency.csv',latency)):
        if rows:
            import csv
            with (output/name).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/rl_sb3/smoke');parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=(args.output or args.run/'independent_audit').resolve()
    try:result=audit(args.run,output)
    except (ValueError,KeyError,AssertionError,OSError,RuntimeError,TypeError,EOFError,SyntaxError,pickle.UnpicklingError,zipfile.BadZipFile) as error:
        output.mkdir(parents=True,exist_ok=True);result=dict(status='failed',error_count=1,errors=[dict(kind='audit_exception',detail=str(error))])
        (output/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','expected_jobs','completed_jobs','expected_test_episodes','audited_test_episodes','error_count') if k in result}))
    raise SystemExit(0 if result['status']=='verified_complete' else 1 if result['status']=='failed' else 2)


if __name__=='__main__':main()
