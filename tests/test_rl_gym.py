"""Gym wrapper parity, balanced namespaces and SB3 TimeLimit contract."""
import json
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from falconworld.rl_env import LongitudinalRLEnv,encode
from falconworld.rl_gym import LongitudinalGymEnv


def test_sb3_check_env_and_seed_stream_matches_original_step_for_step():
    checker=LongitudinalGymEnv();check_env(checker,warn=True);checker.close()
    env=LongitudinalGymEnv(training_seed=2,seed_namespace=6_000_000)
    obs,info=env.reset(seed=2)
    truth=LongitudinalRLEnv();truth.reset(seed=6_200_000,family='constant',condition='id')
    np.testing.assert_array_equal(obs,encode(truth.history,truth.previous_u))
    for tick in range(120):
        index=tick%5;previous=truth.history.copy();previous_u=truth.previous_u
        expected=truth.step(index);actual=env.step(index)
        np.testing.assert_array_equal(actual[0],encode(truth.history,truth.previous_u))
        assert actual[1]==expected[1]*.01 and actual[2:4]==expected[2:4]
        np.testing.assert_array_equal(actual[4]['raw_history_before'],previous)
        np.testing.assert_array_equal(actual[4]['raw_history_next'],truth.history)
        assert actual[4]['previous_u_before']==previous_u
        if actual[2] or actual[3]:break
    assert env.transition_count==tick+1
    assert len(env.episode_metrics)==1
    for k in range(1,5):
        _,info=env.reset()
        assert info['episode_seed']==6_200_000+k
        assert info['family']==('constant','mild','brake','stopgo')[k%4]
    env.close()


def test_dummy_vec_env_preserves_final_observation_and_time_limit_bootstrap():
    # A shorter limit is used only for this boundary-contract fixture.
    env=LongitudinalGymEnv();env.inner.max_steps=1
    vec=DummyVecEnv([lambda:env]);initial=vec.reset()
    _,reward,done,infos=vec.step(np.array([2]))
    assert done[0] and infos[0]['TimeLimit.truncated'] is True
    final=infos[0]['terminal_observation']
    np.testing.assert_array_equal(final,encode(infos[0]['raw_history_next'],infos[0]['applied_u']))
    assert infos[0]['bootstrap_mask']==1.
    # Auto reset is the next episode, not the transition's terminal state.
    assert env.episode_counter==1 and env.case['family']=='mild'
    assert reward.shape==(1,)
    vec.close()


def test_optional_raw_log_and_episode_statistics_keep_original_units(tmp_path):
    path=tmp_path/'trace.jsonl'
    env=LongitudinalGymEnv(log_path=path);env.inner.max_steps=2
    env.reset(seed=0)
    env.step(2);_,reward,term,trunc,info=env.step(2)
    assert not term and trunc and env.transition_count==2
    assert env.episode_metrics[0]['return']!=reward
    env.close()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==2
    assert all(r['learning_reward']==r['raw_reward']*.01 for r in rows)
    assert env.episode_metrics[0]['return']==sum(r['raw_reward'] for r in rows)
