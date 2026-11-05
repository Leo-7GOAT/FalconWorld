"""Double-DQN semantics, gradient budget and portable checkpoint invariants."""
import numpy as np
import pytest
import torch
from falconworld.rl_agent import AgentConfig,DoubleDQNAgent,ReplayBuffer,double_dqn_targets


def replay(seed=0,count=16):
    buffer=ReplayBuffer(32,seed=seed)
    rng=np.random.default_rng(seed)
    for i in range(count):
        buffer.add(rng.normal(size=25),i%5,float(i),rng.normal(size=25),i%4==0,truncated=i%4==1)
    return buffer


def test_online_selects_and_target_evaluates_terminated_masks_only():
    online=torch.tensor([[0.,2.],[3.,1.],[3.,1.]])
    target=torch.tensor([[10.,1.],[2.,20.],[2.,20.]])
    rewards=torch.tensor([3.,4.,4.])
    # Third entry represents a time-limit truncation: it is not terminated.
    terminal=torch.tensor([False,True,False])
    values=double_dqn_targets(online,target,rewards,terminal,.5)
    torch.testing.assert_close(values,torch.tensor([3.5,4.,5.]))
    assert not values.requires_grad


def test_actual_gradient_clipping_and_target_update_are_budgeted():
    config=AgentConfig(batch_size=8,target_update_interval=2,gradient_clip=.01)
    agent=DoubleDQNAgent(config);real,model=replay(1),replay(2)
    before=[p.detach().clone() for p in agent.online.parameters()]
    frozen=[p.detach().clone() for p in agent.target.parameters()]
    first=agent.update(real,model,synthetic_fraction=.5)
    assert first['updated'] and first['real_samples']==4 and first['synthetic_samples']==4
    assert first['batch_size']==8 and first['gradient_updates']==1 and not first['target_synced']
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(frozen,agent.target.parameters()))
    assert all(p.grad is None for p in agent.target.parameters())
    gradient_norm=torch.sqrt(sum(p.grad.square().sum() for p in agent.online.parameters()))
    assert gradient_norm<=.010001
    second=agent.update(real,model,synthetic_fraction=.5)
    assert second['gradient_updates']==2 and second['target_synced']
    assert all(torch.equal(a,b) for a,b in zip(agent.online.parameters(),agent.target.parameters()))
    skipped=agent.update(real,None,synthetic_fraction=.5)
    assert not skipped['updated'] and agent.gradient_updates==2


def test_greedy_evaluation_is_deterministic_and_does_not_consume_exploration():
    agent=DoubleDQNAgent(AgentConfig(seed=3,epsilon_decay_steps=10))
    observation=np.ones(25)
    initial_state=agent.rng.bit_generator.state
    assert agent.act(observation,explore=False)==agent.act(observation,explore=False)
    assert agent.action_steps==0 and agent.epsilon==1.
    assert agent.rng.bit_generator.state==initial_state
    agent.act(observation,explore=True)
    assert agent.action_steps==1 and agent.epsilon<1.


def test_raw_replay_metadata_and_time_limit_are_preserved_without_stale_slots():
    buffer=ReplayBuffer(1)
    history=np.arange(24,dtype=float).reshape(6,4)
    buffer.add(np.zeros(25),0,0,np.ones(25),False,truncated=True,raw_history=history,
               next_raw_history=history+1,previous_action=-.4,next_previous_action=-.8)
    sample=buffer.sample(1)
    assert sample['truncated'][0] and not sample['terminated'][0]
    assert sample['raw_history_valid'][0]
    np.testing.assert_array_equal(sample['raw_history'][0],history)
    buffer.add(np.zeros(25),1,0,np.ones(25),True)
    replacement=buffer.sample(1)
    assert not replacement['raw_history_valid'][0]
    assert np.all(replacement['raw_history']==0)
    with pytest.raises(ValueError,match='together'):
        buffer.add(np.zeros(25),0,0,np.zeros(25),False,raw_history=history)


def test_checkpoint_action_rng_replay_optimizer_and_next_update_roundtrip(tmp_path):
    agent=DoubleDQNAgent(AgentConfig(seed=7,batch_size=8,target_update_interval=3))
    real,model=replay(11),replay(12)
    agent.update(real,model,.5);agent.act(np.zeros(25))
    path=tmp_path/'policy.pt'
    agent.save_checkpoint(path,real_buffer=real,model_buffer=model,metadata={'source':'unit-test'})
    saved=path.read_bytes()
    with pytest.raises(FileExistsError):agent.save_checkpoint(path)
    assert path.read_bytes()==saved
    assert not list(tmp_path.glob('*.tmp'))
    restored=DoubleDQNAgent.load_checkpoint(path)
    assert restored.gradient_updates==agent.gradient_updates
    assert restored.checkpoint_metadata=={'source':'unit-test'}
    observations=np.arange(100,dtype=np.float32).reshape(4,25)/100
    np.testing.assert_array_equal(agent.act(observations,explore=False),restored.act(observations,explore=False))
    np.testing.assert_array_equal(agent.act(observations,epsilon=1),restored.act(observations,epsilon=1))
    actual=agent.update(real,model,.5)
    resumed=restored.update(restored.restored_real_buffer,restored.restored_model_buffer,.5)
    assert actual==resumed
    assert all(torch.equal(a,b) for a,b in zip(agent.online.parameters(),restored.online.parameters()))
