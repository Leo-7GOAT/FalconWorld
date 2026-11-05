"""Strong-default PPO and fresh world-model branch comparisons; simulation only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from falconworld.experiments import write_json
from falconworld.rl_gym import LongitudinalGymEnv
from falconworld.sb3_worldmodel_ppo import WorldModelPPO
from falconworld.rl_evaluation import evaluate_policy,evaluation_cases,summarize
from falconworld.models import load_models,train_models
from run_rl import collect_common,METHODS

PRESETS=dict(smoke=dict(seeds=[0],steps=8192,offline_train=2,offline_val=1),
    standard=dict(seeds=[0,1,2],steps=65536,offline_train=12,offline_val=3),
    full=dict(seeds=[0,1,2,3,4],steps=262144,offline_train=24,offline_val=6))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvaluationPolicy:
    def __init__(self,model):self.model=model
    def act(self,obs,explore=False):
        action,_=self.model.predict(obs,deterministic=not explore)
        return int(action) if np.ndim(action)==0 else action


def job(arguments):
    root=Path(arguments['root']);seed=arguments['seed'];method=arguments['method'];spec=arguments['spec']
    out=root/f'seed{seed}'/method
    if (out/'complete.json').exists():
        saved=json.loads((out/'complete.json').read_text())
        if digest(out/'final_policy.zip')!=saved['checkpoint_sha256'] or digest(out/'test/episodes.jsonl')!=saved['test_sha256']:
            raise ValueError('Completed strong-PPO artifacts changed')
        return saved
    if out.exists() and any(out.iterdir()):out.rename(out.with_name(method+f'_interrupted_{__import__("time").time_ns()}'))
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    world=load_models(root/f'seed{seed}'/'world_models')
    predictor={'wm_ppo':world['history'],'physics_ppo':world['physical_enriched'],
               'wm_no_history':world['residual'],'wm_long':world['history']}.get(method)
    common=json.loads((root/'common_data/manifest.json').read_text())
    offline=common['training_transitions']+common['validation_transitions']
    budget=spec['steps']+(offline if method=='ppo_total_budget' else 0)
    env=LongitudinalGymEnv(training_seed=seed,seed_namespace=6_000_000,reward_scale=.01)
    extra={'n_epochs':20} if method=='real_reuse_control' else {}
    policy=WorldModelPPO('MlpPolicy',env,device='cpu',seed=seed,world_model=predictor,
        model_fraction=.5,model_horizon=3 if method=='wm_long' else 1,model_seed=seed+4321,**extra)
    optimizer_updates=[0]
    hook=policy.policy.optimizer.register_step_post_hook(lambda *args:optimizer_updates.__setitem__(0,optimizer_updates[0]+1))
    actual_config=dict(n_steps=policy.n_steps,batch_size=policy.batch_size,n_epochs=policy.n_epochs,
        learning_rate=policy.learning_rate,gamma=policy.gamma,gae_lambda=policy.gae_lambda,
        clip_range=policy.clip_range(1.),ent_coef=policy.ent_coef,vf_coef=policy.vf_coef,max_grad_norm=policy.max_grad_norm)
    write_json(out/'policy_config.json',actual_config)
    valcases=[c for c in evaluation_cases(seed,'smoke','val') if c['condition']=='id']
    curves=[];training_wall=0.;validation_steps=0;next_val=budget/4;started=perf_counter()
    try:
        while env.transition_count<budget:
            remaining=budget-env.transition_count;n=min(2048,remaining)
            if n!=policy.n_steps:
                policy.n_steps=n
                policy.rollout_buffer=RolloutBuffer(n,policy.observation_space,policy.action_space,
                    device=policy.device,gae_lambda=policy.gae_lambda,gamma=policy.gamma,n_envs=policy.n_envs)
            before=perf_counter();policy.learn(total_timesteps=n,reset_num_timesteps=False,progress_bar=False)
            training_wall+=perf_counter()-before
            current=env.transition_count
            if current>=next_val or current==budget:
                next_val+=budget/4
                rows=evaluate_policy(EvaluationPolicy(policy),valcases,out/'val'/f'step{current}',method=method,training_seed=seed)
                validation_steps+=sum(r['steps'] for r in rows)
                metadata=policy.model_metadata()
                curves.append(dict(method=method,training_seed=seed,new_real_steps=current,
                    policy_updates=optimizer_updates[0],synthetic_generated=metadata['model_generated'],
                    mean_return=float(np.mean([r['return'] for r in rows])),collision_rate=float(np.mean([r['collision'] for r in rows])),
                    completion_rate=float(np.mean([r['completed'] for r in rows])),progress_m=float(np.mean([r['progress_m'] for r in rows])),
                    training_wall_s=training_wall,model_generation_wall_s=metadata.get('model_generation_wall_s',0.)))
                write_json(out/'learning_curve.json',curves)
                print(f'SB3 {method} seed{seed} {current}/{budget} val={curves[-1]["mean_return"]:.2f} completed={curves[-1]["completion_rate"]:.2f}',flush=True)
        if policy.num_timesteps!=budget or env.transition_count!=budget:raise ValueError('Incorrect actual interaction count')
        path=out/'final_policy.zip';policy.save(path)
        # Loading can reseed global RNG; all training is complete before loading.
        frozen=PPO.load(path,device='cpu')
        probe=np.stack([env.reset()[0] for _ in range(16)])
        np.testing.assert_array_equal(EvaluationPolicy(policy).act(probe),EvaluationPolicy(frozen).act(probe))
        rows=evaluate_policy(EvaluationPolicy(frozen),evaluation_cases(seed,'smoke' if arguments['preset']=='smoke' else 'standard','test'),
            out/'test',method=method,training_seed=seed)
        metadata=policy.model_metadata();wm_log=json.loads((root/f'seed{seed}/world_models/training.json').read_text())
        ledger=dict(method=method,training_seed=seed,new_real_steps=budget,policy_updates=optimizer_updates[0],
            synthetic_generated=metadata['model_generated'],synthetic_replaced=metadata['model_replaced'],model_metadata=metadata,
            training_wall_s=training_wall,model_generation_wall_s=metadata.get('model_generation_wall_s',0.),
            validation_steps=validation_steps,test_steps=sum(r['steps'] for r in rows),evaluation_episodes=len(rows),
            common_offline_transitions=offline,worldmodel_pretrain_wall_s=wm_log['training_seconds'] if predictor else 0.,
            native_total_training_interactions=budget+(offline if predictor else 0),shared_data_budget_interactions=budget+offline,
            checkpoint_sha256=digest(path),test_sha256=digest(out/'test/episodes.jsonl'),policy_config=actual_config,
            elapsed_including_evaluation_s=perf_counter()-started,status='verified_complete',
            post_training_probe_resets=16,training_resume='completed jobs only; no partial optimizer/environment resume')
        write_json(out/'complete.json',ledger)
        return ledger
    finally:hook.remove();env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=PRESETS,default='smoke');p.add_argument('--workers',type=int,default=3)
    p.add_argument('--output',type=Path);args=p.parse_args();spec=PRESETS[args.preset]
    root=args.output or Path('outputs/rl_sb3')/args.preset;root.mkdir(parents=True,exist_ok=True)
    sources=[Path('run_sb3_rl.py'),Path('run_rl.py'),*[Path('falconworld')/name for name in
        ('rl_gym.py','sb3_worldmodel_ppo.py','rl_env.py','rl_evaluation.py','ppo_agent.py','models.py','data.py','planning.py')]]
    registration=dict(preset=args.preset,spec=spec,methods=list(METHODS),workers=args.workers,
        learning_reward_scale=.01,algorithm='SB3 default PPO with fresh frozen-policy model branches',
        sources={str(f):digest(f) for f in sources},versions=dict(torch=torch.__version__,stable_baselines3=stable_baselines3.__version__,numpy=np.__version__),
        policy_selection='fixed final budget, validation monitoring only',hardware='simulation only; deployment/realtime deferred')
    registered=root/'registration.json'
    if registered.exists() and json.loads(registered.read_text())!=registration:raise ValueError('Frozen strong-PPO registration changed')
    write_json(registered,registration)
    for f in sources:
        dest=root/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
    data=collect_common(root/'common_data',spec)
    for seed in spec['seeds']:
        target=root/f'seed{seed}'/'world_models'
        if not (target/'training.json').exists():train_models(data,target,seed=seed,preset=args.preset)
        hashes={f.name:digest(f) for f in target.glob('*.npz')};manifest=target/'frozen_sha256.json'
        if len(hashes)!=8 or (manifest.exists() and json.loads(manifest.read_text())!=hashes):raise ValueError('World-model artifact mismatch')
        write_json(manifest,hashes)
    jobs=[dict(root=str(root),seed=seed,method=method,spec=spec,preset=args.preset) for seed in spec['seeds'] for method in METHODS]
    np.random.default_rng(12345).shuffle(jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:ledgers=list(executor.map(job,jobs))
    write_json(root/'cost_ledger.json',ledgers)
    rows=[json.loads(line) for seed in spec['seeds'] for method in METHODS
        for line in (root/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines() if line.strip()]
    summarize(rows,root,reference='ppo')
    expected=len(jobs)*(16 if args.preset=='smoke' else 80)
    write_json(root/'completeness.json',dict(expected_episodes=expected,actual_episodes=len(rows),completed_jobs=len(ledgers),
        expected_jobs=len(jobs),verified_complete=len(rows)==expected and len(ledgers)==len(jobs)))
    print(f'Strong PPO artifacts: {root.resolve()}',flush=True)


if __name__=='__main__':main()
