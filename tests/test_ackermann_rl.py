"""Joint-control contract, causal model roots, terminal targets and SB3 parity."""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from falconworld.ackermann import PhysicalModel
from falconworld.ackermann_rl import (AckermannTrackingEnv, AckermannWorldModelPPO,
    TrackingRoute, encode_tracking, applied_tracking, tracking_transition, model_tracking_batch, CFG)


def test_joint_gym_contract_reproducible_stream_and_limits():
    env = AckermannTrackingEnv()
    check_env(env)
    first, _ = env.reset(seed=7)
    again, _ = AckermannTrackingEnv().reset(seed=7)
    np.testing.assert_array_equal(first, again)
    obs, _, _, _, info = env.step(np.array([1., -1.], np.float32))
    assert obs.shape == (14,)
    np.testing.assert_allclose(info['applied'], [.5, -.08])
    assert env.action_space.shape == (2,)
    with pytest.raises(ValueError):
        applied_tracking([np.nan, 0], [0, 0])


def test_model_branch_matches_independent_truth_across_routes():
    roots, actions, futures, rewards, terminals = [], [], [], [], []
    for i in range(6):
        env = AckermannTrackingEnv()
        obs, _ = env.reset(options=dict(episode_seed=123+i, family='straight' if i%2 else 'curve'))
        action = np.array([-.2+i*.1, .1-i*.04], np.float32)
        future, reward, done, _, _ = env.step(action)
        roots.append(obs);actions.append(action);futures.append(future);rewards.append(reward);terminals.append(done)
    # Known rich form is used only as a deterministic truth-equivalence fixture.
    future, reward, done = model_tracking_batch(np.array(roots), np.array(actions), PhysicalModel(CFG, True))
    np.testing.assert_allclose(future, futures, atol=2e-7)
    np.testing.assert_allclose(.01*reward, rewards, atol=2e-7)
    np.testing.assert_array_equal(done, terminals)


def test_road_failure_goal_and_timeout_have_distinct_semantics():
    s = np.array([29.9, 0., 0., 3., 0., 0.]);n = s.copy();n[0] = 30.2
    _, terminal, info = tracking_transition(s, n, np.zeros(2), np.zeros(2), TrackingRoute(0.))
    assert terminal and info['success'] and not info['failure']
    n[1] = 5.
    _, terminal, info = tracking_transition(s, n, np.zeros(2), np.zeros(2), TrackingRoute(0.))
    assert terminal and info['failure'] and not info['success']
    env = AckermannTrackingEnv();env.reset();env.steps=159
    _, _, terminal, truncated, _ = env.step(np.zeros(2))
    assert truncated and not terminal
    with pytest.raises(RuntimeError):env.step(np.zeros(2))


def test_no_model_is_bitwise_official_ppo():
    torch.set_num_threads(1)
    kwargs=dict(seed=3, device='cpu', n_steps=64, batch_size=32, n_epochs=2)
    official=PPO('MlpPolicy', AckermannTrackingEnv(), **kwargs)
    official.learn(128)
    wrapped=AckermannWorldModelPPO('MlpPolicy', AckermannTrackingEnv(), **kwargs)
    wrapped.learn(128)
    for name, weight in official.policy.state_dict().items():
        assert torch.equal(weight, wrapped.policy.state_dict()[name]), name
    assert wrapped.optimizer_steps==8 and wrapped.model_replaced==0


def test_continuous_branches_fixed_updates_and_load(tmp_path):
    torch.set_num_threads(1)
    model=AckermannWorldModelPPO('MlpPolicy', AckermannTrackingEnv(), world_model=PhysicalModel(CFG, True),
        seed=4, n_steps=64, batch_size=32, n_epochs=2, device='cpu')
    model.learn(128)
    assert model.optimizer_steps==8 and model.model_replaced==64
    assert model.last_branch['max_logprob_error']<=1e-5
    assert model.last_branch['max_value_error']<=1e-5
    path=tmp_path/'policy.zip';model.save(path)
    loaded=PPO.load(path, device='cpu')
    obs,_=AckermannTrackingEnv().reset(seed=11)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],loaded.predict(obs, deterministic=True)[0])
