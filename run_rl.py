"""Frozen, on-policy PPO comparisons with short world-model branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from falconworld.data import FAMILIES
from falconworld.experiments import write_json
from falconworld.rl_env import LongitudinalRLEnv,encode,synthetic_step,ACTION_DELTAS
from falconworld.ppo_agent import PPOAgent,PPOConfig,compute_gae
from falconworld.models import train_models,load_models
from falconworld.rl_evaluation import evaluate_policy,evaluation_cases,summarize

METHODS=('ppo','wm_ppo','physics_ppo','real_reuse_control','wm_no_history','wm_long','ppo_total_budget')
LEARNING_REWARD_SCALE=.01  # Uniform train-only units; raw evaluation reward is unchanged.
PRESETS=dict(smoke=dict(seeds=[0],steps=4096,offline_train=2,offline_val=1),
             standard=dict(seeds=[0,1,2],steps=32768,offline_train=12,offline_val=3),
             full=dict(seeds=[0,1,2,3,4],steps=131072,offline_train=24,offline_val=6))


def collect_common(output,spec):
    """Training/validation only; behavior is shared and never passed to PPO updates."""
    output=Path(output);output.mkdir(exist_ok=True)
    metadata=output/'manifest.json'
    if metadata.exists():
        meta=json.loads(metadata.read_text());data={}
        for split in ('train','val'):
            file=output/f'{split}.npz'
            if hashlib.sha256(file.read_bytes()).hexdigest()!=meta['files'][split]:raise ValueError('Offline data changed')
            data[split]=dict(np.load(file,allow_pickle=False))
        data['metadata']=meta
        return data
    data={};manifest=[]
    for split,count,base in (('train',spec['offline_train'],3_000_000),('val',spec['offline_val'],3_100_000)):
        rows=[]
        for fi,family in enumerate(FAMILIES):
            for i in range(count):
                seed=base+fi*1000+i;rng=np.random.default_rng(seed+12345)
                env=LongitudinalRLEnv();env.reset(seed=seed,family=family,condition='id')
                for t in range(120):
                    h=env.history;p=env.previous_u
                    if rng.random()<.5:index=int(rng.integers(5))
                    else:
                        g,v,lead,_=h[-1]
                        desired=min(.45*(g-4-1.2*v)+.9*(lead-v),.8*(12-v))
                        index=int(np.argmin(abs(np.clip(p+ACTION_DELTAS,-6,2)-desired)))
                    _,reward,term,trunc,info=env.step(index)
                    rows.append(dict(state=h[-1],action=info['applied_u'],next_state=env.state,
                        history=h,episode=fi*1000+i,family=family,step=t))
                    if term or trunc:break
                manifest.append(dict(split=split,seed=seed,family=family,steps=t+1))
        data[split]={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
        np.savez_compressed(output/f'{split}.npz',**data[split])
    files={split:hashlib.sha256((output/f'{split}.npz').read_bytes()).hexdigest() for split in data}
    meta=dict(version='rl-ppo-common-v1',files=files,episodes=manifest,
        sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        training_transitions=len(data['train']['state']),validation_transitions=len(data['val']['state']),
        policy_access='offline data NEVER used for PPO policy/critic gradient updates',
        worldmodel_access='train fit; val selection; no test data exists in this dataset')
    write_json(metadata,meta);data['metadata']=meta
    return data


def real_rollout(agent,env,steps,episode_counter,seed):
    records=[]
    for _ in range(steps):
        h=env.history;p=env.previous_u;obs=encode(h,p)
        action,logprob,value=agent.sample(obs)
        _,reward,term,trunc,info=env.step(int(action))
        records.append(dict(obs=obs,actions=action,old_log_prob=logprob,old_values=value,
            rewards=reward*LEARNING_REWARD_SCALE,next_values=agent.value(encode(env.history,env.previous_u)),
            terminated=term,episode_end=term or trunc,history=h,previous_u=p))
        if term or trunc:
            episode_counter+=1
            env.reset(seed=4_000_000+seed*100_000+episode_counter,
                      family=FAMILIES[episode_counter%4],condition='id')
    result={k:np.asarray([r[k] for r in records]) for k in records[0]}
    result['advantages'],result['returns']=compute_gae(result['rewards'],result['old_values'],
        result['next_values'],result['terminated'],result['episode_end'],agent.config.gamma,agent.config.gae_lambda)
    return result,episode_counter


def model_rollout(agent,world_model,real,horizon,rng):
    """Fresh current-policy branches; old log probabilities precede any update."""
    branches=[]
    for index in rng.permutation(len(real['obs'])):
        h=real['history'][index].copy();p=float(real['previous_u'][index]);records=[]
        for _ in range(horizon):
            obs=encode(h,p);action,logprob,value=agent.sample(obs)
            h_next,reward,term,info=synthetic_step(world_model,h,p,int(action))
            p_next=info['applied_u']
            records.append(dict(obs=obs,actions=action,old_log_prob=logprob,old_values=value,
                rewards=reward*LEARNING_REWARD_SCALE,next_values=agent.value(encode(h_next,p_next)),terminated=term,episode_end=term))
            h,p=h_next,p_next
            if term:break
        part={k:np.asarray([r[k] for r in records]) for k in records[0]}
        part['advantages'],part['returns']=compute_gae(part['rewards'],part['old_values'],part['next_values'],
            part['terminated'],part['episode_end'],agent.config.gamma,agent.config.gae_lambda)
        branches.append(part)
    return {k:np.concatenate([part[k] for part in branches],axis=0) for k in branches[0]}


def train_job(job):
    root=Path(job['output']);seed=job['seed'];method=job['method'];spec=job['spec']
    out=root/f'seed{seed}'/method
    if (out/'complete.json').exists():
        saved=json.loads((out/'complete.json').read_text())
        for name,key in (('final_policy.pt','checkpoint_sha256'),('test/episodes.jsonl','test_sha256')):
            if hashlib.sha256((out/name).read_bytes()).hexdigest()!=saved[key]:raise ValueError(f'Completed job artifact changed: {out/name}')
        return saved
    if out.exists() and any(out.iterdir()):
        # Completed jobs resume; interrupted jobs are preserved, then restarted.
        archive=out.with_name(out.name+f'_interrupted_{int(__import__("time").time())}')
        out.rename(archive)
    out.mkdir(parents=True,exist_ok=True)
    agent=PPOAgent(PPOConfig(seed=seed))
    models=load_models(root/f'seed{seed}'/'world_models')
    model={'wm_ppo':models['history'],'physics_ppo':models['physical_enriched'],
           'wm_no_history':models['residual'],'wm_long':models['history']}.get(method)
    horizon=3 if method=='wm_long' else 1
    env=LongitudinalRLEnv();env.reset(seed=4_000_000+seed*100_000,family=FAMILIES[0],condition='id')
    rng=np.random.default_rng(seed+1000);counter=0;synthetic_count=0;updates=0;curves=[]
    valcases=[dict(seed=7_000_000+seed*100_000+fi*1000+i,family=family,condition='id')
              for fi,family in enumerate(FAMILIES) for i in range(1 if job['preset']=='smoke' else 2)]
    common=json.loads((root/'common_data/manifest.json').read_text())
    common_count=common['training_transitions']+common['validation_transitions']
    step_budget=spec['steps']+(common_count if method=='ppo_total_budget' else 0)
    start=perf_counter();training_wall=0.;model_wall=0.;validation_steps=0;next_validation=step_budget/4
    for step in range(0,step_budget,512):
        before=perf_counter();real,counter=real_rollout(agent,env,min(512,step_budget-step),counter,seed)
        t=perf_counter();imagined=model_rollout(agent,model,real,horizon,rng) if model is not None else None
        model_wall+=perf_counter()-t
        synthetic_count+=len(imagined['obs']) if imagined is not None else 0
        minibatches=(8 if method=='real_reuse_control' else 4)*int(np.ceil(len(real['obs'])/128))
        metrics=agent.update(real,total_minibatches=minibatches,batch_size=128,
            synthetic_fraction=.5 if model is not None else 0.,model_dataset=imagined)
        updates+=minibatches;training_wall+=perf_counter()-before
        current=step+len(real['obs'])
        if current>=next_validation or current==step_budget:
            next_validation+=step_budget/4
            rows=evaluate_policy(agent,valcases,out/'val'/f'step{current}',method=method,training_seed=seed)
            validation_steps+=sum(r['steps'] for r in rows)
            curves.append(dict(method=method,training_seed=seed,new_real_steps=current,policy_updates=updates,
                synthetic_generated=synthetic_count,mean_return=float(np.mean([r['return'] for r in rows])),
                collision_rate=float(np.mean([r['collision'] for r in rows])),
                progress_m=float(np.mean([r['progress_m'] for r in rows])),
                training_wall_s=training_wall,model_generation_wall_s=model_wall,**{f'update_{k}':v for k,v in metrics.items() if np.isscalar(v)}))
            write_json(out/'learning_curve.json',curves)
            print(f'PPO {method} seed{seed} {current}/{step_budget} val={curves[-1]["mean_return"]:.2f}',flush=True)
    checkpoint=out/'final_policy.pt';agent.save_checkpoint(checkpoint,metadata=dict(method=method,training_seed=seed,new_real_steps=step_budget))
    restored=PPOAgent.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(agent.act(real['obs'],explore=False),restored.act(real['obs'],explore=False))
    # Frozen final checkpoint only: no selecting a test winner or best iteration.
    test_preset='smoke' if job['preset']=='smoke' else 'standard'
    rows=evaluate_policy(restored,evaluation_cases(seed,test_preset,'test'),out/'test',method=method,training_seed=seed)
    wm_log=json.loads((root/f'seed{seed}/world_models/training.json').read_text())
    ledger=dict(method=method,training_seed=seed,new_real_steps=step_budget,policy_updates=updates,
        synthetic_generated=synthetic_count,rollout_horizon=horizon if model else 0,
        training_wall_s=training_wall,model_generation_wall_s=model_wall,validation_steps=validation_steps,
        test_steps=sum(r['steps'] for r in rows),evaluation_episodes=len(rows),
        common_offline_transitions=common['training_transitions']+common['validation_transitions'],
        worldmodel_pretrain_wall_s=wm_log['training_seconds'] if model else 0.,
        native_total_training_interactions=step_budget+(common_count if model else 0),
        shared_data_budget_interactions=step_budget+common_count,
        elapsed_including_evaluation_s=perf_counter()-start,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((out/'test/episodes.jsonl').read_bytes()).hexdigest(),
        resume='completed jobs only; interrupted attempts preserved and restarted',status='verified_complete')
    write_json(out/'complete.json',ledger)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=PRESETS,default='smoke');p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    args=p.parse_args();spec=PRESETS[args.preset];root=args.output or Path('outputs/rl_ppo')/args.preset
    root.mkdir(parents=True,exist_ok=True)
    sources=[Path('run_rl.py'),*[Path('falconworld')/name for name in
        ('rl_env.py','ppo_agent.py','rl_evaluation.py','data.py','models.py','planning.py')]]
    registration=dict(preset=args.preset,spec=spec,methods=args.methods,workers=args.workers,
        learning_reward_scale=LEARNING_REWARD_SCALE,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        algorithm='categorical clipped PPO with fresh current-policy model branches; not original MBPO',
        policy_selection='final checkpoint, fixed budget; validation monitoring only',
        hardware='simulation only; vehicle deployment and Pi real-time acceptance deferred')
    frozen=root/'registration.json'
    if frozen.exists() and json.loads(frozen.read_text())!=registration:raise ValueError('Frozen registration changed; choose new output directory')
    write_json(frozen,registration)
    for source in sources:
        dest=root/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(source,dest)
    data=collect_common(root/'common_data',spec)
    for seed in spec['seeds']:
        target=root/f'seed{seed}'/'world_models'
        if not (target/'training.json').exists():train_models(data,target,seed=seed,preset=args.preset)
        model_manifest=target/'frozen_sha256.json'
        actual={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in target.glob('*.npz')}
        if len(actual)!=8:raise ValueError('Incomplete world-model set')
        if model_manifest.exists() and json.loads(model_manifest.read_text())!=actual:raise ValueError('Frozen world-model artifacts changed')
        write_json(model_manifest,actual)
    jobs=[dict(output=str(root),seed=seed,method=method,spec=spec,preset=args.preset)
          for seed in spec['seeds'] for method in args.methods]
    np.random.default_rng(12345).shuffle(jobs)
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:ledgers=list(pool.map(train_job,jobs))
    else:ledgers=[train_job(job) for job in jobs]
    write_json(root/'cost_ledger.json',ledgers)
    rows=[json.loads(line) for seed in spec['seeds'] for method in args.methods
          for line in (root/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines() if line.strip()]
    summarize(rows,root,reference='ppo')
    expected=len(jobs)*(16 if args.preset=='smoke' else 80)
    write_json(root/'completeness.json',dict(expected_episodes=expected,actual_episodes=len(rows),
        completed_jobs=len(ledgers),expected_jobs=len(jobs),verified_complete=len(rows)==expected and len(ledgers)==len(jobs)))
    print(f'PPO artifacts: {root.resolve()}',flush=True)


if __name__=='__main__':main()
