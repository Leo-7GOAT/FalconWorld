"""Compact CPU Double DQN and explicitly mixed real/model replay.

This learns a value function and discrete policy. It is neither SAC nor the
original MBPO algorithm. Environment encoding/reward and rollout generation live
outside this module; a time-limit truncation does not suppress bootstrapping.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import os
from pathlib import Path
import uuid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTION_DELTAS=np.array([-.8,-.4,0.,.4,.8],dtype=np.float64)


@dataclass(frozen=True)
class AgentConfig:
    obs_dim:int=25
    n_actions:int=5
    hidden_sizes:tuple[int,int]=(64,64)
    batch_size:int=128
    gamma:float=.99
    learning_rate:float=3e-4
    target_update_interval:int=250
    gradient_clip:float=10.
    epsilon_start:float=1.
    epsilon_end:float=.05
    epsilon_decay_steps:int=20000
    seed:int=0

    def __post_init__(self):
        dimensions=(self.obs_dim,self.n_actions,self.batch_size,self.target_update_interval,self.epsilon_decay_steps,*self.hidden_sizes)
        if any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in dimensions):raise ValueError('Positive integer dimensions required')
        if len(self.hidden_sizes)!=2:raise ValueError('Expected two hidden layers')
        if not 0<=self.gamma<=1 or not 0<=self.epsilon_end<=self.epsilon_start<=1:raise ValueError('Invalid discount/exploration range')
        if not np.isfinite(self.learning_rate) or self.learning_rate<=0 or not np.isfinite(self.gradient_clip) or self.gradient_clip<=0:
            raise ValueError('Invalid optimizer settings')


class ReplayBuffer:
    """Uniform ring replay. Raw histories are metadata, not extra network inputs."""
    def __init__(self,capacity:int,obs_dim:int=25,raw_history_shape=(6,4),seed:int=0):
        if not isinstance(capacity,int) or capacity<1 or obs_dim<1:raise ValueError('Invalid replay dimensions')
        self.capacity,self.obs_dim,self.raw_history_shape=capacity,obs_dim,tuple(raw_history_shape)
        self.rng=np.random.default_rng(seed);self.size=0;self.position=0
        self.arrays=dict(obs=np.zeros((capacity,obs_dim),np.float32),next_obs=np.zeros((capacity,obs_dim),np.float32),
            action=np.zeros(capacity,np.int64),reward=np.zeros(capacity,np.float32),terminated=np.zeros(capacity,bool),
            truncated=np.zeros(capacity,bool),raw_history=np.zeros((capacity,*self.raw_history_shape),np.float64),
            next_raw_history=np.zeros((capacity,*self.raw_history_shape),np.float64),raw_history_valid=np.zeros(capacity,bool),
            previous_action=np.zeros(capacity,np.float64),next_previous_action=np.zeros(capacity,np.float64))

    def __len__(self):return self.size

    def add(self,obs,action,reward,next_obs,terminated,*,raw_history=None,next_raw_history=None,
            previous_action=None,next_previous_action=None,truncated=False):
        obs,next_obs=np.asarray(obs,dtype=np.float32),np.asarray(next_obs,dtype=np.float32)
        if obs.shape!=(self.obs_dim,) or next_obs.shape!=obs.shape or not np.isfinite(obs).all() or not np.isfinite(next_obs).all():
            raise ValueError('Expected finite encoded observations')
        if not isinstance(action,(int,np.integer)) or action<0 or not np.isfinite(reward):raise ValueError('Invalid transition action/reward')
        raw_values=(raw_history,next_raw_history,previous_action,next_previous_action)
        valid=all(v is not None for v in raw_values)
        if any(v is not None for v in raw_values) and not valid:raise ValueError('Raw rollout metadata must be supplied together')
        extra={}
        if valid:
            for key,value in zip(('raw_history','next_raw_history'),raw_values[:2]):
                value=np.asarray(value,dtype=np.float64)
                if value.shape!=self.raw_history_shape or not np.isfinite(value).all():raise ValueError('Invalid raw history')
                extra[key]=value
            if not np.isfinite(previous_action) or not np.isfinite(next_previous_action):raise ValueError('Invalid previous commands')
        index=self.position
        for key,value in dict(obs=obs,next_obs=next_obs,action=action,reward=reward,terminated=bool(terminated),
                              truncated=bool(truncated),raw_history_valid=valid).items():self.arrays[key][index]=value
        # Reset metadata on overwritten slots to prevent stale raw-state access.
        for key in ('raw_history','next_raw_history','previous_action','next_previous_action'):self.arrays[key][index]=0
        if valid:
            for key,value in extra.items():self.arrays[key][index]=value
            self.arrays['previous_action'][index]=previous_action
            self.arrays['next_previous_action'][index]=next_previous_action
        self.position=(index+1)%self.capacity;self.size=min(self.capacity,self.size+1)

    def sample(self,count:int):
        if not isinstance(count,int) or count<1 or count>self.size:raise ValueError('Replay sample requires 1 <= count <= size')
        indices=self.rng.choice(self.size,size=count,replace=False)
        return {key:value[indices].copy() for key,value in self.arrays.items()}

    def state_dict(self):
        return dict(capacity=self.capacity,obs_dim=self.obs_dim,raw_history_shape=self.raw_history_shape,
            size=self.size,position=self.position,rng_state=self.rng.bit_generator.state,
            arrays={key:value[:self.size].copy() for key,value in self.arrays.items()})

    @classmethod
    def from_state_dict(cls,state):
        buffer=cls(state['capacity'],state['obs_dim'],state['raw_history_shape'])
        buffer.size,buffer.position=int(state['size']),int(state['position'])
        if not 0<=buffer.size<=buffer.capacity or not 0<=buffer.position<buffer.capacity:raise ValueError('Invalid saved replay cursor')
        for key,value in state['arrays'].items():buffer.arrays[key][:buffer.size]=value
        buffer.rng.bit_generator.state=state['rng_state']
        return buffer


class QNetwork(nn.Module):
    def __init__(self,config):
        super().__init__();a,b=config.hidden_sizes
        self.layers=nn.Sequential(nn.Linear(config.obs_dim,a),nn.ReLU(),nn.Linear(a,b),nn.ReLU(),nn.Linear(b,config.n_actions))

    def forward(self,obs):return self.layers(obs)


@torch.no_grad()
def double_dqn_targets(online_next_q,target_next_q,rewards,terminated,gamma):
    """Online argmax chooses actions; frozen target values evaluate those actions."""
    chosen=online_next_q.argmax(dim=1,keepdim=True)
    bootstrap=target_next_q.gather(1,chosen).squeeze(1)
    return rewards+gamma*(1.-terminated.float())*bootstrap


class DoubleDQNAgent:
    def __init__(self,config:AgentConfig=AgentConfig()):
        self.config=config;torch.set_num_threads(1)
        self.rng=np.random.default_rng(config.seed)
        # Seed initialization without changing another agent's global Torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed);self.online=QNetwork(config);self.target=QNetwork(config)
        self.target.load_state_dict(self.online.state_dict());self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=config.learning_rate)
        self.action_steps=0;self.gradient_updates=0
        self.restored_real_buffer=None;self.restored_model_buffer=None;self.checkpoint_metadata={}

    @property
    def epsilon(self):
        c=self.config;fraction=min(1.,self.action_steps/c.epsilon_decay_steps)
        return float(c.epsilon_start+fraction*(c.epsilon_end-c.epsilon_start))

    def act(self,obs,explore=True,epsilon=None):
        values=np.asarray(obs,dtype=np.float32);single=values.ndim==1
        values=np.atleast_2d(values)
        if values.ndim!=2 or values.shape[1]!=self.config.obs_dim or not np.isfinite(values).all():raise ValueError('Invalid encoded observation')
        with torch.no_grad():actions=self.online(torch.from_numpy(values)).argmax(dim=1).numpy().copy()
        if explore:
            eps=self.epsilon if epsilon is None else float(epsilon)
            if not 0<=eps<=1:raise ValueError('epsilon must lie in [0,1]')
            random_mask=self.rng.random(len(values))<eps
            actions[random_mask]=self.rng.integers(0,self.config.n_actions,size=int(random_mask.sum()))
            self.action_steps+=len(values)
        return int(actions[0]) if single else actions

    def update(self,real_buffer,model_buffer=None,synthetic_fraction=0.,batch_size=None):
        size=self.config.batch_size if batch_size is None else batch_size
        if not isinstance(size,int) or size<1 or not 0<=synthetic_fraction<=1:raise ValueError('Invalid batch/mixing fraction')
        synthetic=int(round(size*synthetic_fraction));real=size-synthetic
        if (real and len(real_buffer)<real) or (synthetic and (model_buffer is None or len(model_buffer)<synthetic)):
            return dict(updated=False,skip_reason='insufficient_replay',real_samples=real,synthetic_samples=synthetic,gradient_updates=self.gradient_updates)
        sources=[]
        if real:sources.append(real_buffer.sample(real))
        if synthetic:sources.append(model_buffer.sample(synthetic))
        batch={key:np.concatenate([part[key] for part in sources],axis=0) for key in ('obs','action','reward','next_obs','terminated')}
        if np.any(batch['action']>=self.config.n_actions):raise ValueError('Replay action exceeds network output dimension')
        obs=torch.from_numpy(batch['obs']);next_obs=torch.from_numpy(batch['next_obs'])
        actions=torch.from_numpy(batch['action']);rewards=torch.from_numpy(batch['reward'])
        with torch.no_grad():
            targets=double_dqn_targets(self.online(next_obs),self.target(next_obs),rewards,
                                       torch.from_numpy(batch['terminated']),self.config.gamma)
        prediction=self.online(obs).gather(1,actions[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(prediction,targets)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DQN loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.online.parameters(),self.config.gradient_clip,error_if_nonfinite=True)
        self.optimizer.step();self.gradient_updates+=1
        synced=self.gradient_updates%self.config.target_update_interval==0
        if synced:self.target.load_state_dict(self.online.state_dict())
        return dict(updated=True,loss=float(loss.detach()),td_abs=float((prediction.detach()-targets).abs().mean()),
            gradient_norm_before_clip=float(norm),real_samples=real,synthetic_samples=synthetic,
            batch_size=size,gradient_updates=self.gradient_updates,target_synced=synced)

    def save_checkpoint(self,path,*,real_buffer=None,model_buffer=None,metadata=None):
        """Atomic publish without overwrite; replay snapshots are explicit opt-ins."""
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        payload=dict(format_version=1,config=asdict(self.config),online=self.online.state_dict(),target=self.target.state_dict(),
            optimizer=self.optimizer.state_dict(),rng_state=self.rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
            action_steps=self.action_steps,gradient_updates=self.gradient_updates,
            real_replay=real_buffer.state_dict() if real_buffer is not None else None,
            model_replay=model_buffer.state_dict() if model_buffer is not None else None,
            metadata=metadata or {},resume_scope='agent state plus supplied replay only; environment/runner state not included')
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            torch.save(payload,temporary)
            # A same-volume hard link atomically publishes a completed file and
            # fails if the destination already exists (also on Windows NTFS).
            os.link(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()

    @classmethod
    def load_checkpoint(cls,path,*,load_replay=True):
        """Load only trusted local checkpoints: optimizer/replay contain pickle data."""
        saved=torch.load(Path(path),map_location='cpu',weights_only=False)
        if saved['format_version']!=1:raise ValueError('Unsupported checkpoint format')
        agent=cls(AgentConfig(**saved['config']))
        agent.online.load_state_dict(saved['online']);agent.target.load_state_dict(saved['target'])
        agent.optimizer.load_state_dict(saved['optimizer']);agent.rng.bit_generator.state=saved['rng_state']
        torch.set_rng_state(saved['torch_rng_state']);agent.action_steps=saved['action_steps'];agent.gradient_updates=saved['gradient_updates']
        agent.checkpoint_metadata=saved['metadata']
        if load_replay:
            if saved['real_replay'] is not None:agent.restored_real_buffer=ReplayBuffer.from_state_dict(saved['real_replay'])
            if saved['model_replay'] is not None:agent.restored_model_buffer=ReplayBuffer.from_state_dict(saved['model_replay'])
        return agent
