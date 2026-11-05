"""Official optimization equivalence and fresh-model fixed-buffer invariants."""
import json
import zipfile
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from falconworld.rl_gym import LongitudinalGymEnv
from falconworld.rl_env import encode
from falconworld.models import PhysicsModel
from falconworld.sb3_worldmodel_ppo import WorldModelPPO,decode_model_observation


def config():return dict(n_steps=64,batch_size=16,n_epochs=2,seed=3,device='cpu',verbose=0)


def test_none_path_matches_vanilla_sb3_policy_and_adam_updates_exactly():
    torch.set_num_threads(1)
    reference=PPO('MlpPolicy',LongitudinalGymEnv(),**config());reference.learn(128)
    actual=WorldModelPPO('MlpPolicy',LongitudinalGymEnv(),world_model=None,**config());actual.learn(128)
    for name,weight in reference.policy.state_dict().items():
        torch.testing.assert_close(weight,actual.policy.state_dict()[name],rtol=0,atol=0)
    assert actual.model_generated==actual.model_replaced==0
    assert actual.model_train_calls==2 and actual.model_optimizer_steps==16
    assert actual.model_generation_wall_s==0


def test_h3_gae_finishes_branch_before_truncation_and_preserves_update_budget():
    torch.set_num_threads(1)
    model=WorldModelPPO('MlpPolicy',LongitudinalGymEnv(),
        world_model=PhysicsModel(enriched=[.45,.75,.045],lead='ca'),model_horizon=3,**config())
    # train augmentation must not call the full real-buffer GAE recomputation.
    original=model.rollout_buffer.compute_returns_and_advantage
    calls=[]
    def tracked(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    model.rollout_buffer.compute_returns_and_advantage=tracked
    model.learn(64)
    assert len(calls)==1 # official real rollout collection only, before train()
    metadata=model.model_metadata();last=metadata['last_model_metadata']
    assert metadata['model_replaced']==32
    assert 32<=metadata['model_generated']<=34
    assert last['episode_starts_preserved']
    assert last['max_behavior_logprob_error']<=1e-5
    assert last['actual_optimizer_steps']==8
    assert metadata['model_optimizer_steps']==8
    assert metadata['model_generation_wall_s']>0


def test_decoding_roundoff_and_new_tail_buffer_size_and_vanilla_load(tmp_path):
    torch.set_num_threads(1)
    history=np.tile([30.,5.,6.,-.2],(6,1));encoded=encode(history,2.)
    restored,previous=decode_model_observation(encoded)
    np.testing.assert_allclose(restored,history,atol=1e-6)
    assert previous==2.
    class UnpickleablePhysics(PhysicsModel):
        def __getstate__(self):raise AssertionError('World model must not be pickled')
    model=WorldModelPPO('MlpPolicy',LongitudinalGymEnv(),world_model=UnpickleablePhysics(),model_horizon=1,**config())
    model.learn(64)
    # Match the runner's exact-budget tail: n_steps is changed with a fresh buffer.
    model.n_steps=32
    model.rollout_buffer=RolloutBuffer(32,model.observation_space,model.action_space,device=model.device,
                                     gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=model.n_envs)
    model.learn(32,reset_num_timesteps=False)
    assert model.model_replaced==48 and model.model_generated==48
    assert model.model_optimizer_steps==12
    path=tmp_path/'policy.zip';model.save(path)
    policy=PPO.load(path,device='cpu')
    expected,_=model.predict(encoded,deterministic=True);actual,_=policy.predict(encoded,deterministic=True)
    np.testing.assert_array_equal(expected,actual)
    with zipfile.ZipFile(path) as archive:
        data=json.loads(archive.read('data'))
    assert 'world_model' not in data and '_model_rng' not in data and '_model_torch_rng_state' not in data


def test_terminal_branch_stops_at_first_collision_and_has_no_bootstrap():
    class AlwaysContact:
        def step(self,state,action,history=None):
            future=state.copy();future[0]=-.1;return future
    model=WorldModelPPO('MlpPolicy',LongitudinalGymEnv(),world_model=AlwaysContact(),model_horizon=3,**config())
    branch=model._branch(encode(np.tile([30.,5.,6.,0.],(6,1)),0.))
    assert len(branch)==1 and branch[0]['terminated']
    assert branch[0]['rewards']==-1. # raw contact -100 times training scale .01
    assert branch[0]['returns']==pytest.approx(-1.,abs=1e-6)
