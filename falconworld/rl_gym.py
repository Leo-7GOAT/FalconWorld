"""Gymnasium bridge for the fixed longitudinal RL contract (no vehicle I/O).

A configured namespace/training_seed chooses the physical episode stream.
Gym reset(seed=...) restarts that stream and seeds the Gym RNG; automatic
reset(seed=None) advances one episode. The simulator receives explicit physical
seeds, and families cycle in balanced constant/mild/brake/stopgo order.
"""
from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .rl_env import LongitudinalRLEnv, encode

FAMILIES=('constant','mild','brake','stopgo')


class LongitudinalGymEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,*,training_seed=0,seed_namespace=6_000_000,reward_scale=.01,log_path=None):
        super().__init__()
        if isinstance(training_seed,bool) or not isinstance(training_seed,(int,np.integer)) or not 0<=training_seed<10:
            raise ValueError('training_seed must be an integer in [0,9]')
        if isinstance(seed_namespace,bool) or not isinstance(seed_namespace,(int,np.integer)) or seed_namespace<0:
            raise ValueError('seed_namespace must be a nonnegative integer')
        if not np.isfinite(reward_scale) or reward_scale<=0:raise ValueError('reward_scale must be finite and positive')
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(25,),dtype=np.float32)
        self.action_space=spaces.Discrete(5)
        self.inner=LongitudinalRLEnv();self.training_seed=int(training_seed);self.seed_namespace=int(seed_namespace)
        self.reward_scale=float(reward_scale);self.episode_counter=-1;self.transition_count=0
        self.episode_metrics=[];self.case=None;self.last_transition=None
        self.raw_file=None
        if log_path is not None:
            path=Path(log_path);path.parent.mkdir(parents=True,exist_ok=True)
            self.raw_file=path.open('x',encoding='utf8')

    @property
    def history(self):return self.inner.history

    @property
    def state(self):return self.inner.state

    @property
    def previous_u(self):return self.inner.previous_u

    @property
    def episodes(self):return self.episode_metrics

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.episode_counter=0 if seed is not None else self.episode_counter+1
        self.case=dict(seed=self.seed_namespace+self.training_seed*100_000+self.episode_counter,
            family=FAMILIES[self.episode_counter%len(FAMILIES)],condition='id')
        _,info=self.inner.reset(**self.case)
        self.last_transition=None
        info.update(episode_seed=self.case['seed'],family=self.case['family'],condition='id',
                    reward_scale=self.reward_scale)
        return encode(self.inner.history,self.inner.previous_u),info

    def step(self,action):
        if not self.action_space.contains(action):raise ValueError('Expected discrete integer action in [0,4]')
        history=self.inner.history;previous=self.inner.previous_u
        _,raw_reward,terminated,truncated,info=self.inner.step(int(action))
        next_history=self.inner.history
        self.transition_count+=1
        info.update(raw_history_before=history,previous_u_before=previous,raw_history_next=next_history,
                    raw_reward=float(raw_reward),reward_scale=self.reward_scale)
        # Info is diagnostic/model-training state, never extra policy input.
        self.last_transition=dict(raw_history_before=history.copy(),previous_u_before=previous,
            action_index=int(action),applied_u=info['applied_u'],raw_history_next=next_history.copy(),
            raw_reward=float(raw_reward),learning_reward=float(raw_reward*self.reward_scale),
            terminated=bool(terminated),truncated=bool(truncated))
        if self.raw_file is not None:
            record=dict(self.case,step=self.inner.tick,state=history[-1].tolist(),next_state=next_history[-1].tolist(),
                previous_u=previous,action_index=int(action),applied_u=info['applied_u'],fallback=info['fallback'],
                raw_reward=float(raw_reward),learning_reward=float(raw_reward*self.reward_scale),
                terminated=bool(terminated),truncated=bool(truncated))
            self.raw_file.write(json.dumps(record,allow_nan=False)+'\n')
        if terminated or truncated:
            self.episode_metrics.append(dict(self.case,**info['episode']))
            if self.raw_file is not None:self.raw_file.flush()
        # Do not auto-reset here: DummyVecEnv records terminal_observation and
        # TimeLimit.truncated, allowing SB3 to bootstrap the actual final state.
        return encode(next_history,self.inner.previous_u),float(raw_reward*self.reward_scale),bool(terminated),bool(truncated),info

    def close(self):
        if self.raw_file is not None and not self.raw_file.closed:self.raw_file.close()


GymLongitudinal=LongitudinalGymEnv
