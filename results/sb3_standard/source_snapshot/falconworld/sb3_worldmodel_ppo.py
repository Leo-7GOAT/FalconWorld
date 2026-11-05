"""SB3 PPO with fresh pre-update model-branch replacement of a fixed rollout.

The official PPO optimization remains in super().train(). There is no replay,
no additional policy update, and no return recomputation after target mixing.
"""
from __future__ import annotations
import math
from time import perf_counter
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from .ppo_agent import compute_gae
from .rl_env import STATE_SCALE,encode,synthetic_step


def decode_model_observation(encoded):
    obs=np.asarray(encoded,dtype=np.float64)
    if obs.shape!=(25,) or not np.isfinite(obs).all():raise ValueError('Expected one finite encoded 25-vector')
    history=obs[:24].reshape(6,4)*STATE_SCALE
    # float32 encode(2/6) can round-trip to 2.00000006. This restores only the
    # declared command boundary; state values and model futures are not clipped.
    previous_raw=obs[24]*6.
    if not -6.-1e-5<=previous_raw<=2.+1e-5:raise ValueError('Encoded previous command violates limits')
    previous=float(np.clip(previous_raw,-6.,2.))
    if np.any(history[:,1:3]<0) or history[-1,0]<=0:raise ValueError('Invalid model branch root')
    return history.copy(),previous


class WorldModelPPO(PPO):
    def __init__(self,*args,world_model=None,model_fraction=.5,model_horizon=1,model_seed=None,**kwargs):
        if not 0<=model_fraction<=1 or not isinstance(model_horizon,int) or model_horizon<1:
            raise ValueError('Invalid model fraction/horizon')
        self.world_model=world_model;self.model_fraction=float(model_fraction);self.model_horizon=model_horizon
        self.model_reward_scale=.01
        self.model_generated=0;self.model_replaced=0;self.model_train_calls=0;self.model_branch_roots=0;self.model_optimizer_steps=0
        self.model_generation_wall_s=0.
        self.last_model_metadata={}
        super().__init__(*args,**kwargs)
        selected_seed=(self.seed or 0)+910_001 if model_seed is None else int(model_seed)
        self.model_seed=selected_seed;self._model_rng=np.random.default_rng(selected_seed)
        generator=torch.Generator(device='cpu');generator.manual_seed(selected_seed+1000)
        self._model_torch_rng_state=generator.get_state()
        if world_model is not None:
            if self.device.type!='cpu':raise ValueError('Model-branch path currently verified for CPU policies only')
            if not isinstance(self.action_space,spaces.Discrete) or self.action_space.n!=5 or self.observation_space.shape!=(25,):
                raise ValueError('Model branches require the 25-observation / 5-action longitudinal contract')
            if self.target_kl is not None:raise ValueError('Fixed-gradient comparison requires target_kl=None')

    def _optimizer_step_count(self):
        counters=[state['step'] for state in self.policy.optimizer.state.values() if 'step' in state]
        return max((int(step.item()) if hasattr(step,'item') else int(step) for step in counters),default=0)

    def _branch(self,root):
        history,previous=decode_model_observation(root)
        records=[]
        for _ in range(self.model_horizon):
            observation=encode(history,previous)
            tensor,_=self.policy.obs_to_tensor(observation)
            with torch.no_grad():
                action,value,log_prob=self.policy(tensor)
            index=int(action.cpu().numpy().reshape(-1)[0])
            future,reward,terminated,info=synthetic_step(self.world_model,history,previous,index)
            next_previous=float(info['applied_u'])
            next_observation=encode(future,next_previous)
            next_tensor,_=self.policy.obs_to_tensor(next_observation)
            with torch.no_grad():next_value=float(self.policy.predict_values(next_tensor).reshape(-1)[0])
            records.append(dict(observations=observation,actions=index,values=float(value.reshape(-1)[0]),
                log_probs=float(log_prob.reshape(-1)[0]),rewards=float(reward)*self.model_reward_scale,
                next_values=next_value,terminated=bool(terminated)))
            history,previous=future,next_previous
            if terminated:break
        advantages,returns=compute_gae([r['rewards'] for r in records],[r['values'] for r in records],
            [r['next_values'] for r in records],[r['terminated'] for r in records],
            [r['terminated'] for r in records],self.gamma,self.gae_lambda)
        for i,record in enumerate(records):record['advantages']=float(advantages[i]);record['returns']=float(returns[i])
        return records

    def _replace_with_fresh_model_branches(self):
        buffer=self.rollout_buffer
        if not buffer.full or buffer.generator_ready:raise ValueError('Expected a fresh complete rollout before SB3 flattening')
        if buffer.observations.shape!=(buffer.buffer_size,buffer.n_envs,25):raise ValueError('Unexpected rollout observation shape')
        size=buffer.buffer_size*buffer.n_envs;target=int(round(size*self.model_fraction))
        # Freeze the original real root pool before any replacement; synthetic
        # observations must never become roots of a later branch in this round.
        roots=buffer.observations.reshape(size,25).copy()
        episode_starts=buffer.episode_starts.copy();records=[];root_count=0
        self.policy.set_training_mode(False)
        # Keep model sampling independent of subsequent real rollout actions.
        # The policy itself still samples every action/value/log_prob together.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._model_torch_rng_state)
            while len(records)<target:
                root=roots[int(self._model_rng.integers(size))]
                records.extend(self._branch(root));root_count+=1
            self._model_torch_rng_state=torch.get_rng_state()
        generated=len(records);selected=records[:target]
        indices=self._model_rng.choice(size,size=target,replace=False) if target else np.empty(0,dtype=np.int64)
        if target:
            fields=('observations','actions','values','log_probs','advantages','returns')
            for field in fields:
                array=getattr(buffer,field)
                flattened=array.reshape(size,*array.shape[2:])
                values=np.asarray([r[field] for r in selected],dtype=array.dtype)
                if field=='actions':values=values.reshape(target,*array.shape[2:])
                if not np.isfinite(values).all():raise FloatingPointError('Nonfinite generated model data')
                flattened[indices]=values
            observations=buffer.observations.reshape(size,25)[indices]
            actions=buffer.actions.reshape(size,-1)[indices,0]
            with torch.no_grad():
                tensor,_=self.policy.obs_to_tensor(observations)
                values,log_probs,_=self.policy.evaluate_actions(tensor,torch.as_tensor(actions,dtype=torch.long,device=self.device))
            log_error=float(np.max(abs(log_probs.cpu().numpy()-buffer.log_probs.reshape(size)[indices])))
            value_error=float(np.max(abs(values.cpu().numpy().reshape(-1)-buffer.values.reshape(size)[indices])))
            if log_error>1e-5 or value_error>1e-5:raise AssertionError('Synthetic old log_probs/values disagree with frozen behavior policy')
        else:log_error=value_error=0.
        if not np.array_equal(episode_starts,buffer.episode_starts):raise AssertionError('Episode boundary bookkeeping mutated')
        self.model_generated+=generated;self.model_replaced+=target;self.model_branch_roots+=root_count
        self.last_model_metadata=dict(real_root_pool_size=size,generated=generated,replaced=target,branch_roots=root_count,
            horizon=self.model_horizon,fraction=self.model_fraction,reward_scale=self.model_reward_scale,
            max_behavior_logprob_error=log_error,max_behavior_value_error=value_error,
            episode_starts_preserved=True,targets='full branch GAE before final record truncation',
            expected_optimizer_steps=self.n_epochs*math.ceil(size/self.batch_size))

    def train(self):
        before=self._optimizer_step_count();self.model_train_calls+=1
        if self.world_model is not None and self.model_fraction>0:
            start=perf_counter();self._replace_with_fresh_model_branches()
            elapsed=perf_counter()-start;self.model_generation_wall_s+=elapsed
            self.last_model_metadata['generation_wall_s']=elapsed
        # Official SB3 receives the prepared targets as-is; train() never calls
        # compute_returns_and_advantage and no get()/flatten occurred above.
        super().train()
        steps=self._optimizer_step_count()-before;self.model_optimizer_steps+=steps
        if self.world_model is not None and self.model_fraction>0:
            expected=self.last_model_metadata['expected_optimizer_steps']
            if steps!=expected:raise AssertionError(f'Optimizer budget changed: {steps} != {expected}')
            self.last_model_metadata['actual_optimizer_steps']=steps
            self.logger.record('model/generated',self.model_generated)
            self.logger.record('model/replaced',self.model_replaced)

    def model_metadata(self):
        return dict(model_generated=self.model_generated,model_replaced=self.model_replaced,
            model_train_calls=self.model_train_calls,model_branch_roots=self.model_branch_roots,
            model_optimizer_steps=self.model_optimizer_steps,model_fraction=self.model_fraction,
            model_generation_wall_s=self.model_generation_wall_s,
            model_horizon=self.model_horizon,model_reward_scale=self.model_reward_scale,
            model_seed=self.model_seed,last_model_metadata=dict(self.last_model_metadata),
            save_scope='policy evaluation only; world model/RNG excluded, no partial training resume')

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['world_model','_model_rng','_model_torch_rng_state']
