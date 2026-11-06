"""Unmodeled mechanics, exact-structure control and frozen neural parity."""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from falconworld.ackermann import PhysicalModel
from falconworld.ackermann_rl import AckermannTrackingEnv
from falconworld.ackermann_mismatch import (mismatch_truth,MismatchTrackingEnv,
    SteeringPhysical,SteeringResidual,TorchSteering,save_predictor,load_predictor)
from falconworld.ackermann_model_gradient import ModelGradientPPO


def test_matched_mechanism_control_and_hidden_mismatch():
    rng=np.random.default_rng(19)
    s=rng.uniform([-2,-1,-.3,.5,-1,-.3],[2,1,.3,5,1,.3],(32,6))
    u=rng.uniform([-2,-.4],[1.5,.4],(32,2))
    model=SteeringPhysical('mechanism');model.params=np.array([.22,.06,1.08,.025,.032,.30,1.])
    np.testing.assert_allclose(model.predict_batch(s,u),mismatch_truth(s,u),atol=1e-12)
    normal=AckermannTrackingEnv();changed=MismatchTrackingEnv()
    a,_=normal.reset(seed=5);b,_=changed.reset(seed=5)
    np.testing.assert_array_equal(a,b)
    normal.step(np.zeros(2));changed.step(np.zeros(2))
    assert normal.state[5]!=changed.state[5]
    assert normal.observation_space.shape==changed.observation_space.shape==(14,)
    check_env(changed)
    # Increasing steering demand must not reverse its response near a dead zone.
    grid=np.tile(np.array([0.,0.,0.,3.,0.,-.2]),(501,1))
    commands=np.column_stack([np.zeros(501),np.linspace(-.5,.5,501)])
    assert np.min(np.diff(mismatch_truth(grid,commands)[:,5]))>=-1e-14


def test_torch_parity_gradients_and_serialization(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(3)
    rng=np.random.default_rng(3)
    s=rng.uniform([-2,-1,-.3,.5,-1,-.3],[2,1,.3,5,1,.3],(32,6))
    u=rng.uniform([-2,-.4],[1.5,.4],(32,2))
    for i,predictor in enumerate((SteeringPhysical('affine'),SteeringPhysical('mechanism'),SteeringResidual(SteeringPhysical()))):
        path=tmp_path/f'{i}.npz';save_predictor(predictor,path);loaded=load_predictor(path)
        np.testing.assert_allclose(predictor.predict_batch(s,u),loaded.predict_batch(s,u),atol=1e-12)
        model=TorchSteering(loaded);controls=torch.tensor(u,dtype=torch.float32,requires_grad=True)
        future=model.step(torch.tensor(s,dtype=torch.float32),controls)
        np.testing.assert_allclose(future.detach(),loaded.predict_batch(s,u),atol=3e-7,rtol=2e-6)
        future[:,5].sum().backward()
        assert torch.isfinite(controls.grad).all() and controls.grad.abs().sum()>0
        assert all(not p.requires_grad for p in model.parameters())


def test_neural_model_auxiliary_does_not_train_predictor(tmp_path):
    torch.set_num_threads(1)
    predictor=SteeringResidual(SteeringPhysical())
    model=ModelGradientPPO('MlpPolicy',MismatchTrackingEnv(),world_model=PhysicalModel(rich=True),
        aux_weight=1,aux_horizon=4,aux_interval=2,seed=2,n_steps=64,batch_size=32,n_epochs=2,device='cpu')
    model.world_dynamics=TorchSteering(predictor)
    before={k:v.clone() for k,v in model.world_dynamics.state_dict().items()}
    model.learn(128)
    for k,v in before.items():assert torch.equal(v,model.world_dynamics.state_dict()[k])
    path=tmp_path/'policy.zip';model.save(path);loaded=PPO.load(path,device='cpu')
    obs,_=MismatchTrackingEnv().reset(seed=0)
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
