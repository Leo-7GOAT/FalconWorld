"""Fixed longer-budget pure-PPO reference; no model experience or test selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from run_sb3_rl import job,digest
from falconworld.experiments import write_json
from falconworld.rl_evaluation import summarize


def main():
    root=Path('outputs/rl_sb3/reference_204800');root.mkdir(parents=True,exist_ok=True)
    standard=Path('outputs/rl_sb3/standard')
    sources={str(p):digest(p) for p in [Path(__file__).relative_to(Path.cwd()),Path('run_sb3_rl.py'),
        *[Path('falconworld')/n for n in ('rl_gym.py','rl_env.py','rl_evaluation.py','sb3_worldmodel_ppo.py','ppo_agent.py','data.py','planning.py')]]}
    spec=dict(seeds=[0,1,2],steps=204800,offline_train=12,offline_val=3)
    registered=dict(algorithm='fixed extended pure-PPO reference',spec=spec,methods=['ppo'],sources=sources,
        rationale='65k val seed instability; same default hyperparameters, fixed longer budget, no test-based selection',
        policy_selection='final 204800 only; every 51200 validation monitoring, no early stop or best selection',
        interpretation='additional convergence/data-efficiency reference, not equal-budget model superiority',
        val_gate=dict(completion_min=.9,collision_max=.1,raw_return_min=30),
        hardware='simulation only; no device or real-time deployment')
    path=root/'registration.json'
    if path.exists() and json.loads(path.read_text())!=registered:raise ValueError('Extended-reference registration changed')
    write_json(path,registered)
    for source in sources:
        target=root/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
    if not (root/'common_data').exists():shutil.copytree(standard/'common_data',root/'common_data')
    for seed in spec['seeds']:
        target=root/f'seed{seed}'/'world_models'
        if not target.exists():shutil.copytree(standard/f'seed{seed}'/'world_models',target)
        # These files are an API dependency only; pure PPO never uses model data.
        expected=json.loads((target/'frozen_sha256.json').read_text())
        if any(digest(target/name)!=sha for name,sha in expected.items()):raise ValueError('Predictor dependency changed')
    jobs=[dict(root=str(root),seed=s,method='ppo',spec=spec,preset='standard') for s in spec['seeds']]
    with ProcessPoolExecutor(max_workers=3) as pool:ledgers=list(pool.map(job,jobs))
    write_json(root/'cost_ledger.json',ledgers)
    rows=[json.loads(line) for s in spec['seeds'] for line in (root/f'seed{s}/ppo/test/episodes.jsonl').read_text().splitlines() if line.strip()]
    summarize(rows,root,reference='ppo')
    identities={(r['training_seed'],r['family'],r['condition'],r['scenario_seed']) for r in rows}
    write_json(root/'completeness.json',dict(expected_jobs=3,completed_jobs=len(ledgers),expected_episodes=240,
        actual_episodes=len(rows),unique_episodes=len(identities),verified_complete=len(rows)==len(identities)==240 and len(ledgers)==3))
    print(f'Extended pure PPO reference: {root.resolve()}',flush=True)


if __name__=='__main__':main()
