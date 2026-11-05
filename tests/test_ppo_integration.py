"""Independent integration audit of on-policy PPO sampling and runner budgets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
import run_rl as runner
from falconworld.models import PhysicsModel
from falconworld.ppo_agent import PPOAgent, PPOConfig
from falconworld.rl_env import LongitudinalRLEnv, encode


class ValueIsGapAgent:
    config=SimpleNamespace(gamma=.5,gae_lambda=1.)
    def sample(self, obs):return 2,-float(np.log(5)),self.value(obs)
    def value(self, obs):return float(obs[-5])*50.


class BoundaryEnv:
    """One TimeLimit transition then a two-transition collision episode."""
    def __init__(self):
        self.resets=[];self.episode=0;self.tick=0;self.previous_u=0.
        self.history=np.tile([10.,1.,1.,0.],(6,1))
    def reset(self,**kwargs):
        self.resets.append(kwargs);self.episode+=1;self.tick=0
        self.history=np.tile([100.,1.,1.,0.],(6,1));self.previous_u=0.
    def step(self,action):
        self.tick+=1
        if self.episode==0:gap,reward,term,trunc=20.,1.,False,True
        elif self.tick==1:gap,reward,term,trunc=101.,2.,False,False
        else:gap,reward,term,trunc=0.,3.,True,False
        self.history=np.concatenate((self.history[1:],np.array([[gap,1.,1.,0.]])))
        return None,reward,term,trunc,{}


def test_real_rollout_bootstraps_pre_reset_time_limit_but_never_gae_across_reset():
    env=BoundaryEnv();data,counter=runner.real_rollout(ValueIsGapAgent(),env,3,0,7)
    np.testing.assert_allclose(data['next_values'],[20,101,0],atol=1e-5)
    np.testing.assert_allclose(data['old_values'],[10,100,101],atol=1e-5)
    np.testing.assert_array_equal(data['terminated'],[False,False,True])
    np.testing.assert_array_equal(data['episode_end'],[True,False,True])
    # Scale rewards only; value outputs stay in the critic's current units.
    scale=runner.LEARNING_REWARD_SCALE
    np.testing.assert_allclose(data['advantages'],[scale,3.5*scale-100,3*scale-101],atol=1e-4)
    assert counter==2 and len(env.resets)==2
    assert env.resets[0]['seed']==4_000_000+7*100_000+1
    np.testing.assert_array_equal(data['history'][0],np.tile([10,1,1,0],(6,1)))


def test_behavior_logprob_values_are_frozen_before_update_and_old_data_rejected():
    agent=PPOAgent(PPOConfig(seed=37))
    env=LongitudinalRLEnv();env.reset(seed=40,family='constant',condition='id')
    weights=copy.deepcopy(agent.network.state_dict())
    real,_=runner.real_rollout(agent,env,16,0,37)
    imagined=runner.model_rollout(agent,PhysicsModel(),real,1,np.random.default_rng(9))
    assert len(real['obs'])==len(imagined['obs'])==16
    for key,value in weights.items():torch.testing.assert_close(value,agent.network.state_dict()[key],rtol=0,atol=0)
    saved={k:v.copy() for k,v in real.items()}
    agent._dataset(real);agent._dataset(imagined)  # actual behavior consistency check
    metrics=agent.update(real,total_minibatches=2,batch_size=8,synthetic_fraction=.5,model_dataset=imagined)
    assert metrics['real_samples']==8 and metrics['synthetic_samples']==8
    for key,value in saved.items():np.testing.assert_array_equal(value,real[key])
    with pytest.raises(ValueError,match='current frozen behavior policy'):
        agent.update(real,total_minibatches=1,batch_size=8)
    # Next rollout starts at the last actual state; no previous dataset is supplied.
    fresh,_=runner.real_rollout(agent,env,4,0,37)
    agent._dataset(fresh)
    assert fresh['old_log_prob'].shape==(4,)


@pytest.mark.parametrize('horizon',[1,3])
def test_model_branches_are_independent_and_end_at_collision(horizon):
    roots=np.stack([np.tile([30.,2.,2.,0.],(6,1)),np.tile([40.,2.,2.,0.],(6,1))])
    previous=np.array([.1,.2]);original=roots.copy();seen=[]
    real=dict(obs=np.stack([encode(h,p) for h,p in zip(roots,previous)]),history=roots,previous_u=previous)
    def synthetic(model,h,p,index):
        seen.append((h.copy(),p))
        nxt=h[-1].copy();nxt[0]+=1
        nh=np.concatenate((h[1:],nxt[None]))
        # First root terminates on second step; second never terminates.
        term=bool(nxt[0]==32.)
        return nh,1.,term,{'applied_u':p+.1}
    with patch.object(runner,'synthetic_step',side_effect=synthetic):
        data=runner.model_rollout(ValueIsGapAgent(),object(),real,horizon,np.random.default_rng(0))
    np.testing.assert_array_equal(roots,original)
    if horizon==1:
        assert len(seen)==2 and len(data['obs'])==2
        assert {h[-1,0] for h,p in seen}=={30,40}
    else:
        assert len(seen)==5 and len(data['obs'])==5
        assert {h[-1,0] for h,p in seen}=={30,31,40,41,42}
        assert 32 not in {h[-1,0] for h,p in seen}
    # Separate branch GAE: a horizon-one root never propagates the next root.
    if horizon==1:
        np.testing.assert_allclose(data['advantages'],runner.LEARNING_REWARD_SCALE+.5*data['next_values']-data['old_values'],atol=1e-5)


def test_synthetic_branches_only_call_model_with_raw_state_history_and_applied_action():
    agent=PPOAgent(PPOConfig(seed=2));env=LongitudinalRLEnv()
    env.reset(seed=3,family='constant',condition='id')
    real,_=runner.real_rollout(agent,env,4,0,2)
    calls=[]
    class RawOnly:
        def step(self,state,action,*,history):
            assert state.shape==(4,) and history.shape==(6,4)
            np.testing.assert_array_equal(state,history[-1])
            assert np.isscalar(action) and -6<=action<=2
            calls.append((state.copy(),float(action),history.copy()))
            return PhysicsModel().step(state,action,history=history)
    with patch('falconworld.rl_env.scenario_lead_accel',side_effect=AssertionError('future script leak')):
        imagined=runner.model_rollout(agent,RawOnly(),real,3,np.random.default_rng(3))
    assert len(calls)==len(imagined['obs'])==12


@pytest.mark.parametrize('method,expected_updates,horizon,mix,fixture_steps',[
    ('ppo',4,0,0.,8),('wm_ppo',4,1,.5,8),('physics_ppo',4,1,.5,8),
    ('real_reuse_control',8,0,0.,8),('wm_no_history',4,1,.5,8),('wm_long',4,3,.5,8),
    ('ppo',16,0,0.,512),('ppo_total_budget',4,0,0.,8)])
def test_train_job_fixed_update_budget_and_cost_sources(tmp_path,method,expected_updates,horizon,mix,fixture_steps):
    # This is a tiny eight-real-step integration fixture, not a reduced standard run.
    common=tmp_path/'common_data';common.mkdir()
    (common/'manifest.json').write_text(json.dumps(dict(training_transitions=10,validation_transitions=2)))
    wm=tmp_path/'seed0'/'world_models';wm.mkdir(parents=True)
    (wm/'training.json').write_text(json.dumps(dict(training_seconds=.5)))
    observed=[]
    class SpyAgent(PPOAgent):
        def update(self,data,**kwargs):
            observed.append((len(data['obs']),kwargs['total_minibatches'],kwargs['batch_size'],kwargs['synthetic_fraction'],
                None if kwargs['model_dataset'] is None else len(kwargs['model_dataset']['obs'])))
            return super().update(data,**kwargs)
    models={'history':PhysicsModel(),'physical_enriched':PhysicsModel(),'residual':PhysicsModel()}
    def evaluate_fixture(agent,cases,output,**kwargs):
        output=Path(output);output.mkdir(parents=True)
        rows=[dict(steps=120,**{'return':1.},collision=0,progress_m=1.) for case in cases]
        (output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return rows
    with patch.object(runner,'PPOAgent',SpyAgent),patch.object(runner,'load_models',return_value=models),patch.object(runner,'evaluate_policy',side_effect=evaluate_fixture):
        ledger=runner.train_job(dict(output=str(tmp_path),seed=0,method=method,spec={'steps':fixture_steps},preset='smoke'))
    real_steps=fixture_steps+(12 if method=='ppo_total_budget' else 0)
    assert observed==[(real_steps,expected_updates,128,mix,None if mix==0 else real_steps*horizon)]
    assert ledger['policy_updates']==expected_updates
    assert ledger['synthetic_generated']==(real_steps*horizon if mix else 0)
    assert ledger['new_real_steps']==real_steps and ledger['rollout_horizon']==horizon
    assert ledger['shared_data_budget_interactions']==real_steps+12
    assert ledger['native_total_training_interactions']==real_steps+(12 if mix else 0)
    assert ledger['worldmodel_pretrain_wall_s']==(.5 if mix else 0.)
    assert (tmp_path/'seed0'/method/'final_policy.pt').exists()
    if method=='ppo' and fixture_steps==8:
        job=dict(output=str(tmp_path),seed=0,method=method,spec={'steps':fixture_steps},preset='smoke')
        # Completed jobs must not resample/train/evaluate; artifact corruption is rejected.
        with patch.object(runner,'load_models',side_effect=AssertionError('completed job retrained')):
            restored=runner.train_job(job)
        assert restored==ledger
        trace=tmp_path/'seed0'/method/'test'/'episodes.jsonl'
        saved=trace.read_bytes();trace.write_bytes(saved+b' ')
        with pytest.raises(ValueError,match='artifact changed'):runner.train_job(job)
        trace.write_bytes(saved)
        checkpoint=tmp_path/'seed0'/method/'final_policy.pt'
        checkpoint.write_bytes(checkpoint.read_bytes()+b'corrupt')
        with pytest.raises(ValueError,match='artifact changed'):runner.train_job(job)


def test_training_reward_scale_shared_real_model_but_evaluation_remains_raw(tmp_path):
    from falconworld.rl_evaluation import evaluate_policy
    scale=runner.LEARNING_REWARD_SCALE
    assert scale==.01
    env=BoundaryEnv();real,_=runner.real_rollout(ValueIsGapAgent(),env,1,0,0)
    np.testing.assert_allclose(real['rewards'],[scale])
    def model_transition(model,h,p,index):return h.copy(),1.,False,{'applied_u':p}
    with patch.object(runner,'synthetic_step',side_effect=model_transition):
        imagined=runner.model_rollout(ValueIsGapAgent(),object(),real,1,np.random.default_rng(0))
    np.testing.assert_array_equal(imagined['rewards'],real['rewards'])
    case=dict(seed=7_000_000,family='constant',condition='id')
    env=LongitudinalRLEnv();env.reset(**case);raw_return=0.
    while True:
        _,reward,term,trunc,_=env.step(2);raw_return+=reward
        if term or trunc:break
    class Hold:
        def act(self,obs,explore=False):return 2
    rows=evaluate_policy(Hold(),[case],tmp_path/'raw_validation',method='scale_unit_fixture')
    assert rows[0]['return']==pytest.approx(raw_return)
    assert rows[0]['return']!=pytest.approx(raw_return*scale)
