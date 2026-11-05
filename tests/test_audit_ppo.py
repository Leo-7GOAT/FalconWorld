"""The PPO audit independently detects scaled evaluation rewards and budget drift."""
import numpy as np
import pytest
from scripts.audit_ppo import ROOT,METHODS,PRESETS,audit,sha,initial_state,raw_reward,reconstruct_episode,expected_updates,case_identities


def constant_trace():
    state=initial_state(0,'id');gap,speed,lead,_=state
    rows=[]
    for step in range(120):
        gap+=.1*(lead-speed)
        reward=float(raw_reward(np.array([gap]),np.array([speed]),np.array([0.]),np.array([0.]))[0])
        rows.append(dict(step=step,time_s=(step+1)*.1,gap_m=gap,speed_mps=speed,lead_speed_mps=lead,
            action_index=2,applied_command_mps2=0.,reward=reward,measured_accel_mps2=0.,jerk_mps3=0.,
            gap_error_m=gap-(4+1.2*speed),fallback='',policy_ms=.1,cycle_ms=.2,
            terminated=False,truncated=step==119))
    return rows


def test_poststep_trace_reconstructs_first_old_state_and_complete_progress():
    rows=constant_trace();metrics,_,_=reconstruct_episode(rows,0,'id')
    initial=initial_state(0,'id')
    assert metrics['progress_m']==pytest.approx(initial[1]*12)
    assert metrics['reference_progress_m']==pytest.approx(initial[2]*12)
    assert metrics['return_value']==pytest.approx(sum(r['reward'] for r in rows))
    assert metrics['jerk_rms_mps3']==0
    assert metrics['completed']==1 and metrics['collision']==0


def test_evaluation_reward_scaled_to_training_unit_is_rejected():
    rows=constant_trace()
    for row in rows:row['reward']*=.01
    with pytest.raises(AssertionError,match='raw units'):
        reconstruct_episode(rows,0,'id')
    assert raw_reward(np.array([-.001]),np.array([5.]),np.array([0.]),np.array([-6.]))[0]==-100.


def test_partial_last_minibatch_and_exact_standard_cases():
    assert expected_updates(4096,'ppo')==128
    assert expected_updates(4096,'real_reuse_control')==256
    assert expected_updates(4097,'ppo_total_budget')==132
    cases=[case_identities(seed,'standard') for seed in (0,1,2)]
    assert all(len(values)==80 for values in cases)
    assert len(set.union(*cases))==240
    assert 7*sum(len(values) for values in cases)==1680


def test_registered_but_not_run_experiment_is_incomplete_not_verified(tmp_path):
    import json
    sources=['run_rl.py',*[f'falconworld/{name}' for name in
        ('rl_env.py','ppo_agent.py','rl_evaluation.py','data.py','models.py','planning.py')]]
    digests={}
    for relative in sources:
        destination=tmp_path/'run/source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes())
        digests[relative]=sha(destination)
    registration=dict(preset='smoke',spec=PRESETS['smoke'],methods=list(METHODS),
                       learning_reward_scale=.01,sources=digests)
    (tmp_path/'run/registration.json').write_text(json.dumps(registration),encoding='utf-8')
    result=audit(tmp_path/'run',tmp_path/'result')
    assert result['status']=='provisional_incomplete'
    assert result['expected_jobs']==7 and result['completed_jobs']==0
    assert result['expected_test_episodes']==112 and result['audited_test_episodes']==0
    assert result['error_count']==0
