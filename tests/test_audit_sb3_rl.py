"""Audit counts optimizer batches, not SB3 epoch counters, including tails."""
from scripts.audit_sb3_rl import expected_budget,budget_segments,case_identities


def test_default_and_reuse_optimizer_counts_are_explicit():
    ordinary=expected_budget(65536,'ppo')
    assert ordinary['train_calls']==32
    assert ordinary['optimizer_steps']==10240
    assert ordinary['epoch_updates']==320
    assert ordinary['replaced']==0
    reuse=expected_budget(65536,'real_reuse_control')
    assert reuse['optimizer_steps']==20480 and reuse['epochs']==20
    modeled=expected_budget(65536,'wm_ppo')
    assert modeled['optimizer_steps']==ordinary['optimizer_steps']
    assert modeled['replaced']==32768


def test_exact_tail_and_replacement_use_each_rollout_not_floor_total():
    steps=65536+1391
    assert budget_segments(steps)[-1]==1391
    total=expected_budget(steps,'ppo_total_budget')
    assert total['train_calls']==33 and total['optimizer_steps']==10460
    modeled=expected_budget(steps,'wm_long')
    assert modeled['replaced']==32768+696 # Python round(1391*.5)
    assert 2*modeled['replaced']!=steps


def test_new_standard_registration_is_exact_and_disjoint():
    values=[case_identities(seed,'standard') for seed in (0,1,2)]
    assert len(set.union(*values))==240
    assert 7*sum(map(len,values))==1680
    assert all(seed>=8_000_000 for cases in values for family,condition,seed in cases)


def test_cli_partial_exit_is_two_and_error_exit_is_one(monkeypatch,tmp_path):
    import pytest
    import scripts.audit_sb3_rl as module
    monkeypatch.setattr('sys.argv',['audit_sb3_rl.py','--run',str(tmp_path)])
    for status,expected in (('provisional_incomplete',2),('failed',1),('verified_complete',0)):
        monkeypatch.setattr(module,'audit',lambda *args,_status=status:dict(status=_status,error_count=0))
        with pytest.raises(SystemExit) as error:module.main()
        assert error.value.code==expected
