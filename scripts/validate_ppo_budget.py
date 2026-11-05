"""Fixed 32k->128k pure-PPO budget diagnostic: training ID plus validation only.

No test cases, world-model branches, hyperparameter search, early stop or winner
selection. Continue the optimizer/RNG from the scaled seed0 32k checkpoint.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_rl import real_rollout, LEARNING_REWARD_SCALE
from falconworld.ppo_agent import PPOAgent
from falconworld.rl_env import LongitudinalRLEnv
from falconworld.rl_evaluation import evaluate_policy
from falconworld.experiments import write_json

BASE_STEPS=32768
SEGMENT_STEPS=32768
ADDED_STEPS=98304
THRESHOLDS=dict(completion_rate_min=.9,collision_rate_max=.1,mean_raw_return_min=30.)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_summary(rows):
    result=dict(episodes=len(rows),completion_rate=float(np.mean([r['completed'] for r in rows])),
        collision_rate=float(np.mean([r['collision'] for r in rows])),
        mean_raw_return=float(np.mean([r['return'] for r in rows])),
        progress_m=float(np.mean([r['progress_m'] for r in rows])))
    result['usable_threshold_met']=bool(result['completion_rate']>=THRESHOLDS['completion_rate_min'] and
        result['collision_rate']<=THRESHOLDS['collision_rate_max'] and
        result['mean_raw_return']>=THRESHOLDS['mean_raw_return_min'])
    return result


class LoggedEnv(LongitudinalRLEnv):
    def __init__(self,path):
        super().__init__();self.file=Path(path).open('w',encoding='utf8');self.episode_summaries=[]
    def reset(self,**kwargs):
        self.case=dict(kwargs)
        return super().reset(**kwargs)
    def step(self,index):
        before=self.state;previous=self.previous_u
        result=super().step(index);obs,reward,term,trunc,info=result
        row=dict(seed=int(self.case['seed']),family=self.case['family'],condition=self.case['condition'],
            step=int(info['steps']),state=before.tolist(),next_state=self.state.tolist(),previous_u=previous,
            action_index=int(index),applied_u=info['applied_u'],fallback=info['fallback'],raw_reward=reward,
            learning_reward=reward*LEARNING_REWARD_SCALE,terminated=term,truncated=trunc)
        self.file.write(json.dumps(row,allow_nan=False)+'\n')
        if term or trunc:
            self.episode_summaries.append(dict(self.case,**info['episode']))
            self.file.flush()
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/rl_ppo/standard/seed0/ppo/final_policy.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rl_ppo_budget_validation/seed0_131072')
    args=parser.parse_args();checkpoint=args.checkpoint.resolve();out=args.output.resolve()
    if LEARNING_REWARD_SCALE!=.01:raise ValueError('This fixed diagnostic requires the registered .01 reward scale')
    source_job=checkpoint.parent
    ledger=json.loads((source_job/'complete.json').read_text(encoding='utf8'))
    if ledger['new_real_steps']!=BASE_STEPS or ledger['method']!='ppo' or ledger['training_seed']!=0:
        raise ValueError('Expected seed0 pure PPO at exactly 32768 real training steps')
    if digest(checkpoint)!=ledger['checkpoint_sha256']:raise ValueError('Source checkpoint digest changed')
    registration=json.loads((source_job.parents[1]/'registration.json').read_text(encoding='utf8'))
    if registration['learning_reward_scale']!=.01:raise ValueError('Source must be the scaled experiment')
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be new/empty; interrupted diagnostics are preserved')
    out.mkdir(parents=True,exist_ok=True)
    cases=[dict(seed=7_000_000+fi*1000+i,family=family,condition='id')
        for fi,family in enumerate(('constant','mild','brake','stopgo')) for i in range(2)]
    baseline_path=source_job/'val/step32768/episodes.jsonl'
    baseline_rows=[json.loads(line) for line in baseline_path.read_text(encoding='utf8').splitlines() if line.strip()]
    if {(r['scenario_seed'],r['family'],r['condition']) for r in baseline_rows}!={(r['seed'],r['family'],r['condition']) for r in cases}:
        raise ValueError('Source baseline validation cases differ from the fixed eight cases')
    source_paths=[Path(__file__),ROOT/'run_rl.py',*[ROOT/'falconworld'/name for name in
        ('ppo_agent.py','rl_env.py','rl_evaluation.py','data.py','planning.py')]]
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    preregistration=dict(source_checkpoint=str(checkpoint),source_checkpoint_sha256=digest(checkpoint),
        source_baseline_val_sha256=digest(baseline_path),source_baseline_validation=validation_summary(baseline_rows),
        base_real_steps=BASE_STEPS,added_real_steps=ADDED_STEPS,segment_steps=SEGMENT_STEPS,
        evaluation_totals=[65536,98304,131072],validation_cases=cases,usable_thresholds=THRESHOLDS,
        reward_scale=.01,rollout_steps=512,minibatches_per_rollout=16,batch_size=128,
        training_namespace='5_000_000 + episode_counter; ID only, families cycle',
        policy='run all three segments; no early stopping, no selecting checkpoint, no test access',
        source_sha256=source_hashes,resume='No mid-run resume; preserve interrupted directory and use a fresh output')
    write_json(out/'preregistration.json',preregistration)
    for path in source_paths:
        target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    agent=PPOAgent.load_checkpoint(checkpoint)
    if agent.config.gamma!=.99:raise ValueError('Unexpected inherited gamma')
    initial_updates=agent.gradient_updates;env=LoggedEnv(out/'training_raw.jsonl')
    env.reset(seed=5_000_000,family='constant',condition='id')
    episode_counter=0;curves=[];added=0;training_wall=0.;evaluation_wall=0.;begin=perf_counter()
    try:
        for segment in range(3):
            segment_begin=perf_counter()
            for _ in range(SEGMENT_STEPS//512):
                t=perf_counter()
                # real_rollout uses 4m+seed*100000+counter; seed=10 selects 5m.
                data,episode_counter=real_rollout(agent,env,512,episode_counter,10)
                metrics=agent.update(data,total_minibatches=16,batch_size=128,synthetic_fraction=0.)
                added+=len(data['obs']);training_wall+=perf_counter()-t
            total=BASE_STEPS+added
            checkpoint_path=out/'checkpoints'/f'total{total}.pt'
            agent.save_checkpoint(checkpoint_path,metadata=dict(base_real_steps=BASE_STEPS,added_real_steps=added,
                total_real_steps=total,role='fixed validation-only budget diagnostic; no test'))
            t=perf_counter()
            rows=evaluate_policy(agent,cases,out/'val'/f'total{total}',method='ppo_budget_diagnostic',training_seed=0)
            evaluation_wall+=perf_counter()-t
            summary=validation_summary(rows)
            curve=dict(total_real_steps=total,added_real_steps=added,gradient_updates_total=agent.gradient_updates,
                added_gradient_updates=agent.gradient_updates-initial_updates,training_wall_s=training_wall,
                validation_wall_s=evaluation_wall,elapsed_wall_s=perf_counter()-begin,
                segment_elapsed_s=perf_counter()-segment_begin,completed_training_episodes=len(env.episode_summaries),
                training_collisions=sum(r['collision'] for r in env.episode_summaries),
                training_completions=sum(r['completed'] for r in env.episode_summaries),
                training_mean_raw_return=float(np.mean([r['return'] for r in env.episode_summaries])),
                final_minibatch_diagnostics=metrics,**summary)
            curves.append(curve);write_json(out/'curve.json',curves)
            write_json(out/'training_episodes.json',env.episode_summaries)
            print(json.dumps({k:curve[k] for k in ('total_real_steps','completion_rate','collision_rate','mean_raw_return','usable_threshold_met','elapsed_wall_s')}),flush=True)
        if added!=ADDED_STEPS or agent.gradient_updates-initial_updates!=3072:
            raise AssertionError('Fixed continuation budget was not executed exactly')
        if {str(p.relative_to(ROOT)):digest(p) for p in source_paths}!=source_hashes:
            raise ValueError('Core source changed during the diagnostic')
        write_json(out/'complete.json',dict(status='verified_complete_validation_only',added_real_steps=added,
            total_real_steps=BASE_STEPS+added,added_gradient_updates=agent.gradient_updates-initial_updates,
            training_wall_s=training_wall,validation_wall_s=evaluation_wall,elapsed_wall_s=perf_counter()-begin,
            completed_training_episodes=len(env.episode_summaries),validation_episodes=len(cases)*3,
            thresholds=THRESHOLDS,final_validation=validation_summary(rows),test_episodes=0,
            final_checkpoint_sha256=digest(checkpoint_path),training_raw_sha256=None))
    finally:
        env.file.close()
    complete=json.loads((out/'complete.json').read_text())
    complete['training_raw_sha256']=digest(out/'training_raw.jsonl');write_json(out/'complete.json',complete)
    print(f'Validation-only budget diagnostic: {out}',flush=True)


if __name__=='__main__':main()
