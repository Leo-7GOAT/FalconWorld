"""Frozen-model parity, useful actor gradients and zero-aux PPO equivalence."""
import numpy as np
import torch
from stable_baselines3 import PPO
from falconworld.ackermann import PhysicalModel,ResidualModel
from falconworld.ackermann_rl import AckermannTrackingEnv
from falconworld.ackermann_differentiable import DifferentiableAckermann
from falconworld.ackermann_model_gradient import ModelGradientPPO


def test_differentiable_models_match_fitted_numpy_and_gradients():
    torch.set_num_threads(1)
    rng=np.random.default_rng(4)
    s=rng.uniform([-2,-1,-.3,.5,-1,-.3],[2,1,.3,5,1,.3],(32,6))
    u=rng.uniform([-2,-.4],[1.5,.4],(32,2))
    residual=ResidualModel(PhysicalModel());residual.weights=rng.normal(0,1e-4,(16,3))
    for predictor in (PhysicalModel(),PhysicalModel(rich=True),residual):
        model=DifferentiableAckermann(predictor)
        controls=torch.tensor(u,dtype=torch.float32,requires_grad=True)
        pred=model.step(torch.tensor(s,dtype=torch.float32),controls)
        np.testing.assert_allclose(pred.detach(),predictor.predict_batch(s,u),atol=3e-7,rtol=2e-6)
        pred[:,4:6].square().sum().backward()
        assert torch.isfinite(controls.grad).all() and controls.grad.abs().sum()>0
        assert not list(model.parameters())


def test_aux_off_is_bitwise_official_ppo():
    torch.set_num_threads(1)
    args=dict(seed=8,n_steps=64,batch_size=32,n_epochs=2,device='cpu')
    a=PPO('MlpPolicy',AckermannTrackingEnv(),**args);a.learn(128)
    b=ModelGradientPPO('MlpPolicy',AckermannTrackingEnv(),aux_weight=0,**args);b.learn(128)
    for key,value in a.policy.state_dict().items():assert torch.equal(value,b.policy.state_dict()[key]),key
    assert b.optimizer_steps==8 and b.aux_updates==0


def test_actor_auxiliary_learns_with_frozen_model_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    env=AckermannTrackingEnv();obs,_=env.reset()
    model=ModelGradientPPO('MlpPolicy',env,world_model=PhysicalModel(rich=True),aux_weight=.3,
        aux_horizon=4,aux_interval=2,seed=3,n_steps=64,batch_size=32,n_epochs=2,device='cpu')
    frozen={k:v.clone() for k,v in model.world_dynamics.state_dict().items()}
    loss=model.world_dynamics.loss(model.policy,torch.tensor(np.repeat(obs[None],4,axis=0)),4)
    loss.backward()
    assert model.policy.action_net.weight.grad.abs().sum()>0
    model.warm_start(np.repeat(obs[None],8,axis=0),2)
    model.learn(128)
    assert model.warm_steps==2 and model.optimizer_steps==8 and model.aux_updates==4
    for key,value in frozen.items():assert torch.equal(value,model.world_dynamics.state_dict()[key])
    path=tmp_path/'actor.zip';model.save(path);loaded=PPO.load(path,device='cpu')
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
