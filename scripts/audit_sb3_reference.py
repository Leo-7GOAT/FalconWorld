"""Independent fixed-204800 pure-PPO reference audit, separate from 65k research."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from importlib import metadata
import json
from pathlib import Path
import pickle
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_sb3_rl import read_checkpoint,expected_budget
from scripts.audit_ppo import case_identities,reconstruct_episode,load_csv,load_json,sha,WORLD_MODELS


def audit(run,output):
    run=run.resolve();output.mkdir(parents=True,exist_ok=True)
    errors=[];missing=[];tails=[];artifacts={};differences=defaultdict(float)
    def fail(kind,**detail):errors.append(dict(kind=kind,**detail))
    def require(path):
        if not path.exists():missing.append(str(path));return False
        return True
    registration=load_json(run/'registration.json')
    if registration['algorithm']!='fixed extended pure-PPO reference' or registration['methods']!=['ppo']:
        fail('not_the_registered_pure_reference')
    spec=dict(seeds=[0,1,2],steps=204800,offline_train=12,offline_val=3)
    if registration['spec']!=spec:fail('reference_budget_registration_mismatch')
    source_names={'run_sb3_reference.py','run_sb3_rl.py','rl_gym.py','rl_env.py','rl_evaluation.py','sb3_worldmodel_ppo.py','ppo_agent.py','data.py','planning.py'}
    by_name={Path(p.replace('\\','/')).name:d for p,d in registration['sources'].items()}
    if set(by_name)!=source_names:fail('reference_source_snapshot_set_mismatch')
    current_changes=[]
    for relative,digest in registration['sources'].items():
        relative=Path(relative.replace('\\','/'));path=run/'source_snapshot'/relative
        if require(path):
            actual=sha(path);artifacts[str(path.relative_to(run))]=actual
            if actual!=digest:fail('source_snapshot_sha_mismatch',path=str(path))
        if not (ROOT/relative).exists() or sha(ROOT/relative)!=digest:current_changes.append(str(relative))
    common=None;offline=None
    path=run/'common_data/manifest.json'
    if require(path):
        import hashlib
        common=load_json(path)
        if hashlib.sha256(json.dumps(common['files'],sort_keys=True).encode()).hexdigest()!=common['sha256']:fail('common_manifest_digest_mismatch')
        for split,field in (('train','training_transitions'),('val','validation_transitions')):
            file=run/'common_data'/f'{split}.npz'
            if require(file):
                digest=sha(file);artifacts[str(file.relative_to(run))]=digest
                if digest!=common['files'][split]:fail('common_npz_sha_mismatch',split=split)
                with np.load(file,allow_pickle=False) as arrays:
                    if len(arrays['state'])!=common[field]:fail('common_count_mismatch',split=split)
        offline=common['training_transitions']+common['validation_transitions']
    expected={('ppo',seed,*case) for seed in (0,1,2) for case in case_identities(seed,'standard')}
    budgets=expected_budget(204800,'ppo');observed=[];audited=[];ledgers={};versions=set();pooled=defaultdict(list)
    config=dict(n_steps=2048,batch_size=64,n_epochs=10,learning_rate=.0003,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5)
    for seed in (0,1,2):
        folder=run/f'seed{seed}/ppo';complete=folder/'complete.json'
        ledger=load_json(complete) if require(complete) else None
        if ledger is not None:
            ledgers[seed]=ledger
            expected_values=dict(method='ppo',training_seed=seed,status='verified_complete',new_real_steps=204800,policy_updates=32000,
                synthetic_generated=0,synthetic_replaced=0,worldmodel_pretrain_wall_s=0.,model_generation_wall_s=0.,native_total_training_interactions=204800,
                evaluation_episodes=80,post_training_probe_resets=16)
            if offline is not None:expected_values.update(common_offline_transitions=offline,shared_data_budget_interactions=204800+offline)
            for field,value in expected_values.items():
                if ledger[field]!=value:fail('reference_ledger_budget_mismatch',seed=seed,field=field,actual=ledger[field],expected=value)
            if ledger['policy_config']!=config:fail('reference_changed_default_config',seed=seed)
            use=ledger['model_metadata']
            for field,value in dict(model_generated=0,model_replaced=0,model_branch_roots=0,model_train_calls=100,model_optimizer_steps=32000,model_generation_wall_s=0.,last_model_metadata={}).items():
                if use[field]!=value:fail('pure_reference_used_model_or_wrong_updates',seed=seed,field=field)
        file=folder/'policy_config.json'
        if require(file) and load_json(file)!=config:fail('policy_config_file_mismatch',seed=seed)
        # World models are copied as an API dependency, not consumed by pure PPO.
        model_folder=run/f'seed{seed}/world_models';frozen=model_folder/'frozen_sha256.json'
        if require(frozen):
            hashes=load_json(frozen)
            if set(hashes)!={name+'.npz' for name in WORLD_MODELS}:fail('copied_model_set_mismatch',seed=seed)
            for name,digest in hashes.items():
                file=model_folder/name
                if require(file):
                    actual=sha(file);artifacts[str(file.relative_to(run))]=actual
                    if actual!=digest:fail('copied_model_dependency_sha_mismatch',seed=seed,file=name)
        file=folder/'final_policy.zip'
        if require(file):
            digest=sha(file);artifacts[str(file.relative_to(run))]=digest
            if ledger and digest!=ledger['checkpoint_sha256']:fail('checkpoint_sha_mismatch',seed=seed)
            stored,adam,version=read_checkpoint(file);versions.add(version)
            if adam!={32000}:fail('actual_adam_counter_not_32000',seed=seed,counters=sorted(adam))
            for key,value in dict(num_timesteps=204800,n_steps=2048,batch_size=64,n_epochs=10,_n_updates=1000,
                model_optimizer_steps=32000,model_train_calls=100,model_generated=0,model_replaced=0,model_branch_roots=0,model_generation_wall_s=0.,seed=seed).items():
                if stored.get(key)!=value:fail('zip_reference_budget_mismatch',seed=seed,field=key,actual=stored.get(key),expected=value)
            if any(k in stored for k in ('world_model','_model_rng','_model_torch_rng_state')):fail('reference_pickle_contains_external_model',seed=seed)
            if stored.get('target_kl') is not None:fail('reference_kl_early_stop_enabled',seed=seed)
        file=folder/'learning_curve.json'
        if require(file):
            curve=load_json(file)
            if ledger and [r['new_real_steps'] for r in curve]!=[51200,102400,153600,204800]:fail('reference_not_fixed_four_point_validation',seed=seed)
            for row in curve:
                if row['policy_updates']!=expected_budget(row['new_real_steps'],'ppo')['optimizer_steps'] or row['synthetic_generated']!=0:
                    fail('curve_reference_training_budget_mismatch',seed=seed,step=row['new_real_steps'])
        file=folder/'test/manifest.json'
        if require(file):
            manifest=load_json(file);cases={(r['family'],r['condition'],r['seed']) for r in manifest['cases']}
            if manifest['method']!='ppo' or manifest['training_seed']!=seed or cases!=case_identities(seed,'standard') or len(manifest['cases'])!=80:
                fail('test_manifest_identity_mismatch',seed=seed)
            for name,digest in manifest['source_sha256'].items():
                if by_name.get(name)!=digest:fail('test_manifest_source_mismatch',seed=seed,source=name)
        file=folder/'test/episodes.jsonl'
        if not require(file):continue
        digest=sha(file);artifacts[str(file.relative_to(run))]=digest
        if ledger and digest!=ledger['test_sha256']:fail('test_jsonl_sha_mismatch',seed=seed)
        lines=file.read_text(encoding='utf-8').splitlines();job_rows=[]
        for i,line in enumerate(lines):
            if not line.strip():continue
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i==len(lines)-1:tails.append(str(file));continue
                fail('interior_test_jsonl_corruption',seed=seed,line=i+1);continue
            key=(row['method'],row['training_seed'],row['family'],row['condition'],row['scenario_seed']);observed.append(key);job_rows.append(row)
            if row['method']!='ppo' or row['training_seed']!=seed:fail('wrong_reference_test_job',key=key)
            trace=folder/'test'/f"trace_{row['condition']}_{row['family']}_{row['scenario_seed']}.csv"
            if not require(trace):continue
            try:metrics,cycles,_=reconstruct_episode(load_csv(trace),row['scenario_seed'],row['condition'])
            except (ValueError,KeyError,AssertionError) as error:
                fail('raw_trace_reconstruction_failure',key=key,detail=str(error));continue
            for field,value in metrics.items():
                if field=='fallback_reasons':
                    if json.loads(row[field])!=value:fail('fallback_reason_mismatch',key=key)
                    continue
                target='return' if field=='return_value' else field
                error=abs(row[target]-value);differences[target]=max(differences[target],error)
                if not np.isclose(row[target],value,rtol=1e-9,atol=1e-7):fail('raw_episode_metric_mismatch',key=key,field=target)
            if not np.allclose(row['cycle_ms_samples'],cycles,rtol=0,atol=1e-9):fail('trace_cycle_samples_mismatch',key=key)
            audited.append(dict(method='ppo',training_seed=seed,family=row['family'],condition=row['condition'],scenario_seed=row['scenario_seed'],**metrics))
            pooled[row['condition']].extend(cycles)
        if ledger and len(job_rows)==80 and ledger['test_steps']!=sum(r['steps'] for r in job_rows):fail('test_step_ledger_mismatch',seed=seed)
    counts=Counter(observed);actual=set(observed)
    if any(n>1 for n in counts.values()):fail('duplicate_reference_test_identity')
    if actual-expected:fail('unexpected_reference_test_identity',keys=[list(k) for k in sorted(actual-expected)])
    root_ledger=run/'cost_ledger.json'
    if require(root_ledger):
        records=load_json(root_ledger);mapping={r['training_seed']:r for r in records}
        if set(mapping)!={0,1,2} or len(records)!=3:fail('root_reference_job_set_mismatch')
        if any(seed in ledgers and value!=ledgers[seed] for seed,value in mapping.items()):fail('root_job_reference_ledger_disagreement')
    file=run/'completeness.json'
    if require(file):
        completeness=load_json(file)
        if completeness['expected_jobs']!=3 or completeness['expected_episodes']!=240:fail('wrong_reference_expected_counts')
        if completeness.get('verified_complete') and (len(ledgers)!=3 or actual!=expected):fail('false_reference_complete_claim')
    installed=metadata.version('stable_baselines3')
    if (versions and versions!={installed}) or installed!='2.9.2':fail('reference_sb3_version_mismatch',checkpoint_versions=sorted(versions),installed=installed)
    complete=not errors and not missing and not tails and set(ledgers)=={0,1,2} and actual==expected and len(observed)==len(audited)==240
    result=dict(status='verified_complete' if complete else 'failed' if errors else 'provisional_incomplete',
        expected_jobs=3,completed_jobs=len(ledgers),expected_test_episodes=240,raw_test_episodes=len(observed),audited_test_episodes=len(audited),
        missing_test_identities=[list(k) for k in sorted(expected-actual)],missing_artifacts=sorted(set(missing)),interrupted_jsonl_tails=tails,
        error_count=len(errors),errors=errors,max_abs_metric_difference=dict(differences),
        artifact_sha256=artifacts,current_source_differences=current_changes,audit_source_sha256=sha(Path(__file__)),
        reconstruction_helper_sha256=sha(ROOT/'scripts/audit_ppo.py'),zip_reader_helper_sha256=sha(ROOT/'scripts/audit_sb3_rl.py'),
        sb3_version=installed,source_snapshot_scope='Nine registered project source snapshots; original SB3 library source was not archived',
        optimizer_budget=dict(per_seed_online_steps=204800,per_seed_actual_adam_steps=32000,train_calls=100,epochs=10,batch=64),
        raw_reward_check='Every evaluation reward independently recomputed in unscaled task units',
        limitations='No per-step training transition log: ledger, zip num_timesteps and Adam counters cross-check budgets. Zero synthetic counters/source path establish the recorded no-model reference; not a formal execution trace. Extra-budget reference is not an equal-budget model comparison or a vehicle capability claim.')
    (output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    if audited:
        import csv
        with (output/'recomputed_test_episodes.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(audited[0]));writer.writeheader();writer.writerows(audited)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/rl_sb3/reference_204800');parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=(args.output or args.run/'independent_audit').resolve()
    try:result=audit(args.run,output)
    except (ValueError,KeyError,AssertionError,OSError,RuntimeError,TypeError,EOFError,SyntaxError,pickle.UnpicklingError,zipfile.BadZipFile) as error:
        output.mkdir(parents=True,exist_ok=True);result=dict(status='failed',error_count=1,errors=[dict(kind='audit_exception',detail=str(error))])
        (output/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','expected_jobs','completed_jobs','expected_test_episodes','audited_test_episodes','error_count') if k in result}))
    raise SystemExit(0 if result['status']=='verified_complete' else 1 if result['status']=='failed' else 2)


if __name__=='__main__':main()
