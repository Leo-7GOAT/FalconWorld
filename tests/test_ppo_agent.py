"""PPO likelihood, GAE boundaries, fresh-rollout budget and checkpoint checks."""
import numpy as np
import pytest
import torch
from torch.distributions import Categorical
from falconworld.ppo_agent import PPOConfig,PPOAgent,compute_gae,clipped_policy_loss


def fresh_data(agent,count=128,seed=1):
    obs=np.random.default_rng(seed).normal(size=(count,25)).astype(np.float32)
    action,log_prob,value=agent.sample(obs)
    advantage=np.where(action==0,1.,-.25).astype(np.float32)
    return dict(obs=obs,actions=action,old_log_prob=log_prob,advantages=advantage,
                returns=value+advantage,old_values=value,policy_version=agent.policy_version)


def test_ratio_clips_beneficial_direction_for_positive_and_negative_advantages():
    new=torch.log(torch.tensor([1.5,.5,1.5,.5])).requires_grad_()
    loss,ratio=clipped_policy_loss(new,torch.zeros(4),torch.tensor([1.,1.,-1.,-1.]),.2)
    torch.testing.assert_close(ratio,torch.tensor([1.5,.5,1.5,.5]))
    assert float(loss.detach())==pytest.approx(.15)
    loss.backward()
    assert new.grad[0]==0 and new.grad[3]==0
    assert new.grad[1]!=0 and new.grad[2]!=0


def test_gae_time_limit_bootstraps_but_does_not_mix_reset_episode():
    args=([1.,2.,3.],[10.,20.,30.],[20.,99.,40.])
    advantage,returns=compute_gae(*args,[False,False,False],[False,True,False],.9,.8)
    np.testing.assert_allclose(advantage,[60.192,71.1,9.],rtol=1e-6)
    np.testing.assert_allclose(returns,[70.192,91.1,39.],rtol=1e-6)
    terminated,_=compute_gae(*args,[False,True,False],[False,True,False],.9,.8)
    np.testing.assert_allclose(terminated,[-3.96,-18.,9.],rtol=1e-6)
    # A collected rollout cut is not a terminal: the final delta still bootstraps.
    cut,_=compute_gae([1.],[2.],[3.],[False],[False],.9,.95)
    np.testing.assert_allclose(cut,[1.7])
    vector,_=compute_gae(np.ones((2,3)),np.zeros((2,3)),np.ones((2,3)),
                         np.zeros((2,3),bool),np.array([[True]*3,[False]*3]),.9,.8)
    np.testing.assert_allclose(vector,np.full((2,3),1.9))


def test_sampled_log_prob_and_value_match_behavior_and_greedy_does_not_advance_rng():
    agent=PPOAgent(PPOConfig(seed=4));obs=np.ones((20,25),np.float32)
    action,log_prob,value=agent.sample(obs)
    with torch.no_grad():
        logits,reference_value=agent.network(torch.from_numpy(obs))
        reference=Categorical(logits=logits).log_prob(torch.from_numpy(action))
    np.testing.assert_array_equal(log_prob,reference.numpy())
    np.testing.assert_array_equal(value,reference_value.numpy())
    before=agent.torch_rng.get_state().clone()
    assert agent.act(obs[0])==agent.act(obs[0])
    assert torch.equal(before,agent.torch_rng.get_state())
    assert isinstance(agent.sample(obs[0])[0],int)
    assert isinstance(agent.value(obs[0]),float)
    np.testing.assert_allclose(agent.value(obs),value)


def test_policy_changes_with_fixed_mixed_budget_and_stale_rollouts_are_rejected():
    agent=PPOAgent(PPOConfig(seed=0));real,model=fresh_data(agent,seed=5),fresh_data(agent,seed=6)
    with torch.no_grad():before=Categorical(logits=agent.network(torch.from_numpy(real['obs']))[0]).probs[:,0].mean().item()
    result=agent.update(real,total_minibatches=8,batch_size=32,synthetic_fraction=.25,model_dataset=model)
    assert result['minibatches']==8 and result['gradient_updates']==8
    assert result['real_samples']==192 and result['synthetic_samples']==64
    with torch.no_grad():after=Categorical(logits=agent.network(torch.from_numpy(real['obs']))[0]).probs[:,0].mean().item()
    assert after>before
    norm=torch.sqrt(sum(p.grad.square().sum() for p in agent.network.parameters()))
    assert norm<=.500001
    with pytest.raises(ValueError,match='Stale'):
        agent.update(real,total_minibatches=1)
    assert agent.gradient_updates==8


def test_checkpoint_greedy_sampling_optimizer_and_next_fresh_update_match(tmp_path):
    agent=PPOAgent(PPOConfig(seed=8));data=fresh_data(agent)
    agent.update(data,total_minibatches=2,batch_size=32)
    fresh=fresh_data(agent,seed=10);path=tmp_path/'ppo.pt'
    agent.save_checkpoint(path,{'split':'train'})
    original=path.read_bytes()
    with pytest.raises(FileExistsError):agent.save_checkpoint(path)
    assert path.read_bytes()==original
    restored=PPOAgent.load_checkpoint(path)
    assert restored.checkpoint_metadata=={'split':'train'}
    np.testing.assert_array_equal(agent.act(fresh['obs']),restored.act(fresh['obs']))
    for a,b in zip(agent.sample(fresh['obs']),restored.sample(fresh['obs'])):np.testing.assert_array_equal(a,b)
    expected=agent.update(fresh,2,32)
    actual=restored.update(fresh,2,32)
    assert expected==actual
    assert all(torch.equal(a,b) for a,b in zip(agent.network.parameters(),restored.network.parameters()))
