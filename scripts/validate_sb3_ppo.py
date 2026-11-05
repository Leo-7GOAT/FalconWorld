"""One fixed SB3 official-default PPO cross-check; training and validation only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import inspect
import random
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from falconworld.rl_env import LongitudinalRLEnv, encode, REWARD_VERSION
from falconworld.rl_evaluation import evaluate_policy
from falconworld.experiments import write_json

TOTAL_STEPS=204800
SEGMENT_STEPS=51200
REWARD_SCALE=.01
FAMILIES=('constant','mild','brake','stopgo')
DOC_URL='https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GymLongitudinal(gym.Env):
    """Fixed-scale 25D observations and discrete deltas, with raw logging."""
    metadata={'render_modes':[]}
    def __init__(self,namespace,log_path):
        super().__init__()
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(25,),dtype=np.float32)
        self.action_space=spaces.Discrete(5)
        self.inner=LongitudinalRLEnv();self.namespace=namespace;self.episode_counter=-1
        self.raw_file=Path(log_path).open('w',encoding='utf8')
        self.transition_count=0;self.episodes=[];self.probes=[]
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.episode_counter=0 if seed is not None else self.episode_counter+1
        self.case=dict(seed=self.namespace+self.episode_counter,
            family=FAMILIES[self.episode_counter%len(FAMILIES)],condition='id')
        raw,info=self.inner.reset(**self.case)
        return encode(self.inner.history,self.inner.previous_u),info
    def step(self,action):
        state=self.inner.state;previous=self.inner.previous_u
        encoded=encode(self.inner.history,previous)
        if len(self.probes)<32:self.probes.append(encoded.copy())
        _,raw_reward,terminated,truncated,info=self.inner.step(int(action))
        obs=encode(self.inner.history,self.inner.previous_u)
        self.transition_count+=1
        record=dict(self.case,step=self.inner.tick,state=state.tolist(),next_state=self.inner.state.tolist(),
            previous_u=previous,action_index=int(action),applied_u=info['applied_u'],fallback=info['fallback'],
            raw_reward=raw_reward,learning_reward=raw_reward*REWARD_SCALE,
            terminated=terminated,truncated=truncated)
        self.raw_file.write(json.dumps(record,allow_nan=False)+'\n')
        if terminated or truncated:
            self.episodes.append(dict(self.case,**info['episode']))
            self.raw_file.flush()
        # SB3's DummyVecEnv preserves final observation and marks TimeLimit.
        return obs,float(raw_reward*REWARD_SCALE),bool(terminated),bool(truncated),info
    def close(self):
        if not self.raw_file.closed:self.raw_file.close()


class RawValidationPolicy:
    def __init__(self,model):self.model=model
    def act(self,encoded,explore=False):
        if explore:raise ValueError('Validation policy must be deterministic')
        action,_=self.model.predict(encoded,deterministic=True)
        return int(action)


def summary(rows):
    return dict(episodes=len(rows),completion_rate=float(np.mean([r['completed'] for r in rows])),
        collision_rate=float(np.mean([r['collision'] for r in rows])),
        mean_raw_return=float(np.mean([r['return'] for r in rows])),
        progress_m=float(np.mean([r['progress_m'] for r in rows])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/sb3_ppo_default_validation/seed0_204800')
    args=p.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be new/empty; preserve any interrupted attempt')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if (sb3.__version__,gym.__version__)!=('2.9.2','1.4.0'):
        raise ValueError('This registered cross-check requires SB3 2.9.2 / Gymnasium 1.4.0')
    package_root=Path(sb3.__file__).resolve().parent
    package_hashes={str(path.relative_to(package_root)):digest(path) for path in package_root.rglob('*.py')}
    sources=[Path(__file__),*[ROOT/'falconworld'/name for name in ('rl_env.py','rl_evaluation.py','data.py','planning.py')]]
    project_hashes={str(path.relative_to(ROOT)):digest(path) for path in sources}
    cases=[dict(seed=7_000_000+fi*1000+i,family=family,condition='id') for fi,family in enumerate(FAMILIES) for i in range(2)]
    registration=dict(algorithm='stable_baselines3.PPO',official_defaults_reference=DOC_URL,
        constructor='PPO("MlpPolicy", env, device="cpu", seed=0); all other kwargs omitted',
        official_local_signature=str(inspect.signature(PPO.__init__)),
        versions=dict(stable_baselines3=sb3.__version__,gymnasium=gym.__version__,torch=torch.__version__,numpy=np.__version__),
        total_real_steps=TOTAL_STEPS,validation_totals=[51200,102400,153600,204800],
        training_namespace=5_200_000,check_env_namespace=5_900_000,validation_cases=cases,
        reward_version=REWARD_VERSION,training_reward_scale=REWARD_SCALE,validation_reward='raw environment, unscaled',
        policy='fixed four segments, no test access, no early stop, no best checkpoint selection, no hyperparameter tuning',
        interpretation='configuration/implementation cross-check only; not a world-model gain experiment',
        sb3_source_sha256=package_hashes,project_source_sha256=project_hashes)
    write_json(out/'preregistration.json',registration)
    for path in sources:
        target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    begin=perf_counter();checker=GymLongitudinal(5_900_000,out/'check_env_raw.jsonl')
    try:check_env(checker,warn=True)
    finally:checker.close()
    write_json(out/'gym_check.json',dict(status='passed',extra_check_transitions=checker.transition_count,
        namespace=5_900_000,excluded_from_training_budget=True,elapsed_s=perf_counter()-begin))
    env=GymLongitudinal(5_200_000,out/'training_raw.jsonl')
    model=PPO('MlpPolicy',env,device='cpu',seed=0)
    model.set_logger(configure(str(out/'sb3_logs'),['csv','json']))
    actual=dict(n_steps=model.n_steps,batch_size=model.batch_size,n_epochs=model.n_epochs,
        learning_rate=model.learning_rate,gamma=model.gamma,gae_lambda=model.gae_lambda,
        clip_range=model.clip_range(1.),clip_range_vf=model.clip_range_vf,normalize_advantage=model.normalize_advantage,
        ent_coef=model.ent_coef,vf_coef=model.vf_coef,max_grad_norm=model.max_grad_norm,
        target_kl=model.target_kl,use_sde=model.use_sde,sde_sample_freq=model.sde_sample_freq,
        policy_kwargs=model.policy_kwargs,policy_architecture=str(model.policy),device=str(model.device),
        torch_threads=torch.get_num_threads())
    if (actual['n_steps'],actual['batch_size'],actual['n_epochs'],actual['ent_coef'],actual['learning_rate'])!=(2048,64,10,0.,.0003):
        raise AssertionError('Official default configuration unexpectedly differs')
    write_json(out/'actual_configuration.json',actual)
    optimizer_steps=[0]
    def count_step(optimizer,args,kwargs):optimizer_steps[0]+=1
    hook=model.policy.optimizer.register_step_post_hook(count_step)
    curves=[];training_wall=0.;validation_wall=0.;run_begin=perf_counter();final_path=None
    try:
        for segment in range(4):
            t=perf_counter()
            model.learn(total_timesteps=SEGMENT_STEPS,reset_num_timesteps=False)
            training_wall+=perf_counter()-t
            total=(segment+1)*SEGMENT_STEPS
            if model.num_timesteps!=total or env.transition_count!=total:raise AssertionError('SB3 real step budget drifted')
            final_path=out/'checkpoints'/f'total{total}.zip';final_path.parent.mkdir(exist_ok=True)
            model.save(final_path)
            # PPO.load invokes set_random_seed; validation/reload must not reset
            # the still-training policy's global RNG at segment boundaries.
            python_state=random.getstate();numpy_state=np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[]):
                    restored=PPO.load(final_path,device='cpu')
            finally:
                random.setstate(python_state);np.random.set_state(numpy_state)
            probes=np.asarray(env.probes,dtype=np.float32)
            expected=model.predict(probes,deterministic=True)[0];reloaded=restored.predict(probes,deterministic=True)[0]
            np.testing.assert_array_equal(expected,reloaded)
            t=perf_counter()
            rows=evaluate_policy(RawValidationPolicy(restored),cases,out/'val'/f'total{total}',
                method='sb3_default_ppo_validation_only',training_seed=0)
            validation_wall+=perf_counter()-t
            item=dict(total_real_steps=total,optimizer_steps=optimizer_steps[0],sb3_epoch_updates=int(model._n_updates),
                completed_training_episodes=len(env.episodes),training_collisions=sum(e['collision'] for e in env.episodes),
                training_completions=sum(e['completed'] for e in env.episodes),
                training_mean_raw_return=float(np.mean([e['return'] for e in env.episodes])),
                training_wall_s=training_wall,validation_wall_s=validation_wall,
                elapsed_wall_s=perf_counter()-run_begin,checkpoint_sha256=digest(final_path),reload_consistency='32 deterministic probe actions identical',**summary(rows))
            curves.append(item);write_json(out/'curve.json',curves);write_json(out/'training_episodes.json',env.episodes)
            print(json.dumps({k:item[k] for k in ('total_real_steps','completion_rate','collision_rate','mean_raw_return','optimizer_steps','elapsed_wall_s')}),flush=True)
        if optimizer_steps[0]!=32000 or model._n_updates!=1000:raise AssertionError('Unexpected official-default update budget')
        if {str(path.relative_to(ROOT)):digest(path) for path in sources}!=project_hashes:raise ValueError('Project source changed mid-run')
        if {str(path.relative_to(package_root)):digest(path) for path in package_root.rglob('*.py')}!=package_hashes:
            raise ValueError('SB3 package source changed mid-run')
    finally:
        hook.remove();env.close()
    write_json(out/'complete.json',dict(status='verified_complete_validation_only',total_real_steps=env.transition_count,
        rollout_count=TOTAL_STEPS//2048,epochs_per_rollout=10,optimizer_steps=optimizer_steps[0],
        training_wall_s=training_wall,validation_wall_s=validation_wall,elapsed_wall_s=perf_counter()-run_begin,
        completed_training_episodes=len(env.episodes),validation_episodes=len(cases)*4,test_episodes=0,
        final_validation=summary(rows),final_checkpoint_sha256=digest(final_path),training_raw_sha256=digest(out/'training_raw.jsonl'),
        default_cross_check='Different budget/configuration from custom PPO; no world-model benefit claim'))
    print(f'Official-default SB3 validation-only cross-check: {out}',flush=True)


if __name__=='__main__':main()
