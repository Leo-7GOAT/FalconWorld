"""Reporting must not overweight long episodes or label partial logs complete."""
from pathlib import Path
import json
from scripts.build_report import MODELS, episode_mae, longitudinal_audit, read_jsonl, sensor_differences, research_conclusion


def test_episode_aggregation_does_not_weight_windows_as_independent():
    base = dict(model='m', split='test', horizon_s='3', training_seed='0')
    rows = [dict(base, episode='1', error='0') for _ in range(10)]
    rows += [dict(base, episode='2', error='10')]
    episodes, summary = episode_mae(rows, ('error',))
    assert len(episodes) == 2
    assert summary[0]['error'] == 5
    assert summary[0]['episodes'] == 2


def test_partial_and_duplicate_closed_rows_are_not_complete(tmp_path):
    (tmp_path/'run_config.json').write_text(json.dumps(dict(preset='smoke', seeds=[0], budget='equal')))
    folder = tmp_path/'seed0/models'; folder.mkdir(parents=True)
    for name in MODELS:
        (folder/f'{name}.npz').touch()
    assert longitudinal_audit(tmp_path, [])['status'] == 'provisional_incomplete'
    rows = []
    for fi, family in enumerate(('constant', 'mild', 'brake', 'stopgo')):
        for pair in range(2):
            for condition in ('id', 'ood'):
                for model in (*MODELS, 'ensemble_risk', 'acc'):
                    rows.append(dict(model=model, condition=condition, family=family,
                                     training_seed=0, scenario_seed=900000+fi*1000+pair, budget='equal'))
    assert longitudinal_audit(tmp_path, rows)['status'] == 'verified_complete'
    assert longitudinal_audit(tmp_path, rows+[rows[0]])['status'] == 'provisional_incomplete'


def test_only_incomplete_tail_is_ignored(tmp_path):
    path = tmp_path/'partial.jsonl'
    path.write_text('{"a": 1}\n{"a":')
    assert read_jsonl(path) == [dict(a=1)]
    path.write_text('{"a":\n{"a": 1}\n')
    import pytest
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_sensor_pairing_never_crosses_models_or_scenarios():
    base = dict(family='brake', condition='id', scenario_seed='10', training_seed='0')
    metrics = ('collision','completed','progress_m','gap_rmse_m','jerk_rms_mps3',
               'fallback_count','cycle_p95_ms','timeout_rate')
    rows = [dict(base, model=model, observation_mode=mode, **{k:str(value) for k in metrics})
            for model,mode,value in [('a','privileged',10),('a','sensor',12),
                                     ('b','privileged',100),('b','sensor',105)]]
    # An unmatched sensor episode must not borrow another episode's privileged run.
    rows.append(dict(rows[1], scenario_seed='99'))
    pairs, summary = sensor_differences(rows)
    assert len(pairs) == 2
    assert [r['progress_m'] for r in summary] == [2,5]
    assert all(r['paired_episodes']==1 for r in summary)
    import pytest
    with pytest.raises(ValueError, match='Duplicate'):
        sensor_differences(rows+[rows[0]])


def test_provisional_results_cannot_trigger_global_research_conclusion(tmp_path):
    # Missing files also ensure the provisional guard does not read or select them.
    text='\n'.join(research_conclusion(tmp_path/'missing',tmp_path,dict(status='provisional_incomplete')))
    assert '暂不成立' in text
    assert '不根据部分测试选择新主方法' in text
    assert '当前登记结果不支持' not in text
