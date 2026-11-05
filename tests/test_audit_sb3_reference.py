"""Extended reference has a separate budget and never inherits 65k expectations."""
import pytest
from scripts.audit_sb3_reference import expected_budget,case_identities


def test_reference_exact_data_and_adam_budget():
    budget=expected_budget(204800,'ppo')
    assert budget['train_calls']==100
    assert budget['optimizer_steps']==32000
    assert budget['epoch_updates']==1000
    assert budget['replaced']==0
    assert sum(len(case_identities(seed,'standard')) for seed in (0,1,2))==240


def test_reference_cli_does_not_accept_partial_as_complete(monkeypatch,tmp_path):
    import scripts.audit_sb3_reference as module
    monkeypatch.setattr('sys.argv',['audit_sb3_reference.py','--run',str(tmp_path)])
    for status,code in (('verified_complete',0),('provisional_incomplete',2),('failed',1)):
        monkeypatch.setattr(module,'audit',lambda *args,_status=status:dict(status=_status,error_count=0))
        with pytest.raises(SystemExit) as result:module.main()
        assert result.value.code==code
