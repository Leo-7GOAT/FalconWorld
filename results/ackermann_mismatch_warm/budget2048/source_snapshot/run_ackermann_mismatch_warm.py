"""Exploratory follow-up: fixed zero/2048 online budgets, entirely new tests.

Both budgets are reported; no best-checkpoint or scenario selection. Original
16384-step mismatch experiment remains the primary registered experiment.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import torch
from stable_baselines3 import PPO
from falconworld.ackermann import PhysicalModel
from falconworld.ackermann_mismatch import MismatchTrackingEnv,TorchSteering,load_predictor
from falconworld.ackermann_model_gradient import ModelGradientPPO
from run_ackermann_improvement import warm_roots
from run_ackermann_mismatch import evaluate
from run_ackermann_rl import CONFIG,sha,write


def job(args):
    root=Path(args['root']);seed=args['seed'];method=args['method'];torch.set_num_threads(1)
    mp=root/'budget0'/f'seed{seed}'/'models';rich=PhysicalModel.load(mp/'rich.npz')
    env=MismatchTrackingEnv(training_seed=seed,namespace=390_000_000)
    model=ModelGradientPPO('MlpPolicy',env,world_model=rich,aux_weight=1,aux_horizon=16,aux_interval=4,
        seed=seed,device='cpu',**CONFIG)
    if method!='rich':model.world_dynamics=TorchSteering(load_predictor(mp/f'{method}.npz'))
    start=perf_counter();model.warm_start(warm_roots(seed),256);warm_seconds=perf_counter()-start
    ledgers=[]
    for budget in (0,2048):
        folder=root/f'budget{budget}'/f'seed{seed}'/method;folder.mkdir()
        start=perf_counter()
        if budget:model.learn(2048)
        training_seconds=perf_counter()-start
        checkpoint=folder/'final_policy.zip';model.save(checkpoint)
        # Evaluate live deterministic actor; loading a checkpoint here would
        # reseed Torch before the second budget, so loading is checked afterward.
        rows=evaluate(model,folder/'test',seed,namespace=380_000_000)
        ledger=dict(method=method,training_seed=seed,online_steps=budget,warm_steps=256,
            policy_optimizer_steps=model.optimizer_steps,aux_updates=model.aux_updates,
            imagined_steps=model.aux_model_steps,offline_interactions=3000,
            warm_seconds=warm_seconds,training_seconds=training_seconds,
            checkpoint_sha256=sha(checkpoint),test_sha256=sha(folder/'test/episodes.jsonl'),
            trajectory_sha256=sha(folder/'test/trajectories.npz'),test_episodes=len(rows))
        write(folder/'complete.json',ledger);ledgers.append(dict(ledger,budget=budget))
    frozen=PPO.load(checkpoint,device='cpu')
    obs,_=env.reset(seed=0)
    import numpy as np
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],frozen.predict(obs,deterministic=True)[0])
    print('WARM',seed,method,flush=True);return ledgers


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    previous=json.loads((args.source/'registration.json').read_text());methods=['rich','affine','residual','mechanism']
    source_names=['run_ackermann_mismatch_warm.py','run_ackermann_mismatch.py','falconworld/ackermann_mismatch.py',
        'falconworld/ackermann_rl.py','falconworld/ackermann.py','falconworld/ackermann_model_gradient.py',
        'falconworld/ackermann_differentiable.py','run_ackermann_improvement.py']
    for budget in (0,2048):
        folder=root/f'budget{budget}';folder.mkdir()
        reg=dict(previous,methods=methods,online_steps=budget,evaluation_namespace=380_000_000,
            primary='Exploratory follow-up only: report both budgets, new tests, no selection. Does not replace original 16384-step experiment.',
            sources={s:sha(s) for s in source_names})
        write(folder/'registration.json',reg)
        for seed in previous['seeds']:
            shutil.copytree(args.source/f'seed{seed}/models',folder/f'seed{seed}/models')
        for s in source_names:
            dst=folder/'source_snapshot'/s;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,dst)
    jobs=[dict(root=str(root),seed=s,method=m) for s in previous['seeds'] for m in methods]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:ledgers=[r for group in pool.map(job,jobs) for r in group]
    for budget in (0,2048):
        folder=root/f'budget{budget}';write(folder/'cost_ledger.json',[r for r in ledgers if r['budget']==budget]);records=[]
        for seed in previous['seeds']:
            for method in methods:
                records += [dict(json.loads(line),method=method) for line in (folder/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines()]
        write(folder/'episodes.json',records)


if __name__=='__main__':main()
