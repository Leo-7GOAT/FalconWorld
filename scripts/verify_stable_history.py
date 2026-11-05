"""Verify saved val-selected artifacts without training, search, or test access."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from falconworld.stable_history import StableHistoryModel
from scripts.analyze_validation import evaluate_validation,aggregate,COMPONENTS
from scripts.evaluate_stable_history import score


def main():
    out=ROOT/'outputs/stable_history_validation';run=ROOT/'outputs/standard_verified'
    result=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    data=dict(np.load(run/'data/val.npz',allow_pickle=False))
    metrics=[f'{mode}_{c}' for mode in ('recursive','teacher_forced_endpoint') for c in COMPONENTS]
    checks=[]
    for name,kind in (('stable_history','learned_ar2'),('adaptive_damped_ca','adaptive_damped_ca')):
        selected=min((r for r in result['trials'] if r['kind']==kind),key=lambda r:r['val_gap_score_m'])
        model=StableHistoryModel.load(out/f'{name}.npz')
        expected=selected.get('coefficients',[selected.get('gamma'),0.])
        np.testing.assert_array_equal(model.coefficients,expected)
        rows=evaluate_validation({name:model},data,0)
        ep=aggregate(rows,('seed','episode','family','model','horizon_s'),metrics)
        recomputed=score(ep)
        np.testing.assert_allclose(recomputed,result['baseline_scores'][name],rtol=0,atol=1e-12)
        roots=np.roots([1.,-model.coefficients[0],-model.coefficients[1]])
        if kind=='learned_ar2':assert max(abs(roots))<=selected['radius']+1e-12
        checks.append(dict(model=name,selection_matches_val_argmin=True,reloaded_val_score_m=recomputed,
            val_score_abs_difference=abs(recomputed-result['baseline_scores'][name]),
            root_max_abs=float(max(abs(roots))),sha256=hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()))
    evidence=dict(accessed_splits=['val'],search_performed=False,training_performed=False,checks=checks)
    (out/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence))


if __name__=='__main__':main()
