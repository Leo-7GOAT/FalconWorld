"""Experimental completeness must validate identities, not only row counts."""
import json
import numpy as np
from falconworld.experiments import summarize, write_json, read_jsonl


def test_incorrect_scenario_is_not_complete(tmp_path):
    write_json(tmp_path/'run_config.json',dict(seeds=[0],budget='equal'))
    seed=tmp_path/'seed0'
    write_json(seed/'closed_equal_config.json',dict(models={},count=1,preset='smoke'))
    rows=[]
    for family_i,family in enumerate(('constant','mild','brake','stopgo')):
        for condition in ('id','ood'):
            rows.append(dict(model='acc',condition=condition,budget='equal',training_seed=0,
                family=family,scenario_seed=900000+1000*family_i,collision=0,completed=1,
                progress_m=1.,gap_rmse_m=1.,jerk_rms_mps3=0.,fallback_count=0,
                cycle_p50_ms=1.,cycle_p95_ms=1.,timeout_rate=0.))
    path=seed/'closed_equal_seed0.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summarize(tmp_path)
    assert json.loads((tmp_path/'completeness.json').read_text())['status']=='verified_complete'
    rows[0]['scenario_seed']+=1
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summarize(tmp_path)
    audit=json.loads((tmp_path/'completeness.json').read_text())
    assert audit['status']=='provisional_incomplete'
    assert len(audit['missing_pairs'])==len(audit['unexpected_pairs'])==1


def test_interrupted_tail_preserves_complete_results(tmp_path):
    path=tmp_path/'metrics.jsonl'
    path.write_text('{"episode":1}\n{"episode":')
    assert read_jsonl(path,repair=True)==[{'episode':1}]
    assert path.with_suffix('.jsonl.incomplete_tail.txt').exists()
    assert read_jsonl(path)==[{'episode':1}]


def test_open_loop_ensemble_keeps_member_history():
    from falconworld.experiments import recursive_prediction
    class Member:
        def __init__(self, bias):self.bias=bias
        def step(self,state,action,history):
            return np.asarray(state)+history[-1]**2+self.bias
    class Ensemble:
        members=[Member(0),Member(1)]
        def step(self,*args,**kwargs):
            raise AssertionError('Must propagate members independently')
    actions=np.zeros(3)
    state=np.ones(4)
    history=np.ones((6,4))
    actual=recursive_prediction(Ensemble(),state,actions,history)
    expected=np.mean([recursive_prediction(m,state,actions,history) for m in Ensemble.members],axis=0)
    np.testing.assert_array_equal(actual,expected)
