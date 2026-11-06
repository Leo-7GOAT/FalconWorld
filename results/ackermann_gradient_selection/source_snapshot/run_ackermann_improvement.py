"""Validation-only model-gradient selection, then new-seed frozen comparison."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import shutil
from time import perf_counter
import numpy as np
import torch
from stable_baselines3 import PPO
from falconworld.ackermann import PhysicalModel,ResidualModel
from falconworld.ackermann_rl import AckermannTrackingEnv,TrackingRoute,encode_tracking
from falconworld.ackermann_model_gradient import ModelGradientPPO
from run_ackermann_rl import CONFIG,offline,evaluate,sha,write

CANDIDATES={'a':dict(weight=.3,horizon=12,warm_steps=128,interval=4),
            'b':dict(weight=1.,horizon=16,warm_steps=256,interval=4)}
METHODS=('ppo','ppo_compute','smooth_ppo','wm_gradient','physics_gradient')


def warm_roots(seed):
    rng=np.random.default_rng(72_000_000+seed);obs=[]
    for i in range(2048):
        route=TrackingRoute(0. if i%2 else 1.3,float(rng.uniform(-.8,.8)),float(rng.uniform(2,4)))
        x=rng.uniform(0,29)
        s=np.array([x,route.road(x)+rng.uniform(-.5,.5),route.heading(x)+rng.uniform(-.12,.12),
            route.target_speed+rng.uniform(-.5,.5),rng.uniform(-.5,.5),rng.uniform(-.12,.12)])
        previous=np.array([rng.uniform(-.5,.5),rng.uniform(-.12,.12)])
        obs.append(encode_tracking(s,previous,route))
    return np.array(obs)


def evaluation(policy,folder,seed,pairs,stage):
    # Existing evaluator owns raw logs/trajectories. Seed offsets reserve
    # namespaces 88M for validation selection and >=95.3M for final testing.
    rows=evaluate(policy,folder,seed+(30 if stage=='select' else 50),pairs,
                  'validation' if stage=='select' else 'test')
    for r in rows:r['training_seed']=seed
    with (folder/'episodes.jsonl').open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    return rows


def job(arg):
    root=Path(arg['root']);seed=arg['seed'];method=arg['method'];cfg=arg['config'];stage=arg['stage']
    torch.set_num_threads(1)
    folder=root/f'seed{seed}'/method;folder.mkdir()
    mp=root/f'seed{seed}'/'models'
    predictor=PhysicalModel.load(mp/'rich_physical.npz') if method=='physics_gradient' else ResidualModel.load(mp/'residual.npz')
    uses_aux=method not in ('ppo','ppo_compute')
    env=AckermannTrackingEnv(training_seed=seed)
    model=ModelGradientPPO('MlpPolicy',env,world_model=predictor if uses_aux else None,
        aux_weight=cfg['weight'] if uses_aux else 0,aux_horizon=cfg['horizon'],aux_interval=cfg['interval'],
        smooth_only=method=='smooth_ppo',seed=seed,device='cpu',**CONFIG)
    start=perf_counter();model.warm_start(warm_roots(seed),cfg['warm_steps'])
    warm_seconds=perf_counter()-start
    write(folder/'warm.json',dict(steps=model.warm_steps,seconds=warm_seconds,imagined_steps=model.aux_model_steps))
    steps=16384 if stage=='select' else 65536
    start=perf_counter()
    model.learn(total_timesteps=steps-2048)
    if method=='ppo_compute':model.n_epochs=10+cfg['warm_steps']//32
    model.learn(total_timesteps=2048,reset_num_timesteps=False)
    model.n_epochs=10
    training_seconds=perf_counter()-start
    assert env.transition_count==model.num_timesteps==steps
    path=folder/'final_policy.zip';model.save(path)
    frozen=PPO.load(path,device='cpu')
    probe,_=AckermannTrackingEnv().reset(seed=123)
    np.testing.assert_array_equal(model.predict(probe,deterministic=True)[0],frozen.predict(probe,deterministic=True)[0])
    rows=evaluation(frozen,folder/('validation' if stage=='select' else 'test'),seed,4 if stage=='select' else 10,stage)
    ledger=dict(method=method,training_seed=seed,online_steps=steps,warm_steps=model.warm_steps,
        policy_optimizer_steps=model.optimizer_steps,total_optimizer_steps=model.optimizer_steps+model.warm_steps,
        aux_updates=model.aux_updates,imagined_steps=model.aux_model_steps,
        offline_interactions=3000 if method in ('wm_gradient','physics_gradient') else 0,
        warm_seconds=warm_seconds,training_seconds=training_seconds,config=cfg,
        checkpoint_sha256=sha(path),test_episodes=len(rows),
        evaluation_sha256=sha(folder/('validation' if stage=='select' else 'test')/'episodes.jsonl'),
        trajectory_sha256=sha(folder/('validation' if stage=='select' else 'test')/'trajectories.npz'),
        mean_return=float(np.mean([r['return_raw'] for r in rows])),
        completion_rate=float(np.mean([r['completed'] for r in rows])),
        lateral_rmse_m=float(np.mean([r['lateral_rmse_m'] for r in rows])))
    write(folder/'complete.json',ledger)
    print(json.dumps(ledger),flush=True)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('select','final'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    seeds=[0,1] if args.stage=='select' else [3,4,5]
    chosen=None
    if args.stage=='final':
        if args.selection is None:raise ValueError('Frozen validation selection required')
        chosen=json.loads(args.selection.read_text(encoding='utf8'))
    methods=tuple(CANDIDATES) if args.stage=='select' else METHODS
    sources=['run_ackermann_improvement.py','falconworld/ackermann_differentiable.py','falconworld/ackermann_model_gradient.py',
        'falconworld/ackermann_rl.py','falconworld/ackermann.py','run_ackermann_rl.py']
    write(root/'registration.json',dict(stage=args.stage,seeds=seeds,online_steps=16384 if args.stage=='select' else 65536,
        methods=methods,candidates=CANDIDATES,selection=chosen,selection_sha256=sha(args.selection) if chosen else None,
        primary='new test mean raw return, with completion not lower; lateral RMSE and steering rate secondary',
        selection_rule='highest validation completion averaged over both seeds, then highest validation mean raw return',
        namespaces=dict(training=10_000_000,offline=80_000_000,warm_roots=72_000_000,selection_validation=88_000_000,final_test=95_300_000),
        changes='real rollout and GAE kept intact; actor-only differentiable model cost before joint backward; warm and online gradients counted separately',
        sources={s:sha(s) for s in sources},versions=dict(torch=torch.__version__,numpy=np.__version__),
        limitation='simulation; sampled imagined roots; analytic route known; extra offline/model/gradient costs; no guaranteed improvement'))
    for name in sources:
        target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    for seed in seeds:offline(root,seed)
    tasks=[]
    for seed in seeds:
        for method in methods:
            tasks.append(dict(root=str(root),seed=seed,method='wm_gradient' if args.stage=='select' else method,
                config=CANDIDATES[method] if args.stage=='select' else chosen['config'],stage=args.stage,candidate=method))
    # Candidate-specific directories isolate experiments without touching labels.
    if args.stage=='select':
        for key in CANDIDATES:
            candidate_root=root/key;candidate_root.mkdir()
            for seed in seeds:shutil.copytree(root/f'seed{seed}/models',candidate_root/f'seed{seed}/models')
        for task in tasks:task['root']=str(root/task['candidate'])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:ledgers=list(pool.map(job,tasks))
    write(root/'cost_ledger.json',ledgers)
    if args.stage=='select':
        scores=[]
        for key in CANDIDATES:
            rows=[json.loads((root/key/f'seed{s}/wm_gradient/complete.json').read_text()) for s in seeds]
            scores.append(dict(candidate=key,config=CANDIDATES[key],completion=float(np.mean([r['completion_rate'] for r in rows])),
                mean_return=float(np.mean([r['mean_return'] for r in rows]))))
        selected=max(scores,key=lambda r:(r['completion'],r['mean_return']))
        write(root/'selection.json',dict(selected,validation_scores=scores,test_access=False))
        print('SELECTED',json.dumps(selected),flush=True)
    else:
        rows=[]
        for seed in seeds:
            for method in methods:
                rows += [dict(json.loads(line),method=method) for line in (root/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines()]
        write(root/'episodes.json',rows)
        summary=[]
        for method in methods:
            for dist in ('id','ood'):
                rr=[r for r in rows if r['method']==method and r['distribution']==dist]
                summary.append(dict(method=method,distribution=dist,episodes=len(rr),completed=sum(r['completed'] for r in rr),
                    departures=sum(r['road_departure'] for r in rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in ('return_raw','lateral_rmse_m','speed_rmse_mps','steering_rate_rms_radps')}))
        write(root/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
