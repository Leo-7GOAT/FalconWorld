"""Same MPC/observations across sensor methods; no actuator ground truth input."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.models import load_models
from falconworld.planning import CEMConfig
from falconworld.experiments import closed_episode,write_csv

import argparse
import hashlib
import json
from falconworld.experiments import read_jsonl,write_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--models',default='outputs/smoke_verified/seed0/models')
p.add_argument('--output',default='outputs/sensor_closed_verified')
args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
fingerprint=dict(models={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(args.models).glob('*.npz')},
    sources={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
             ('scripts/run_sensor_closed.py','falconworld/experiments.py','falconworld/diagnostics.py','falconworld/planning.py','falconworld/data.py','falconworld/models.py')},
    scope='smoke sensor sensitivity, one training seed, two paired episodes per family and distribution')
meta=out/'manifest.json'
if meta.exists() and json.loads(meta.read_text())!=fingerprint:
    raise ValueError('Sensor run fingerprint changed; choose a new output directory')
write_json(meta,fingerprint)
models=load_models(args.models)
if not {'physical_enriched','history','neural'}<=set(models):raise ValueError('Missing models')
raw=out/'metrics.jsonl'
rows=read_jsonl(raw,repair=True)
seen={(r['condition'],r['family'],r['scenario_seed'],r['model'],r['observation_mode']) for r in rows}
for condition in ('id','ood'):
    for fi,family in enumerate(('constant','mild','brake','stopgo')):
        for i in range(2):
            for name in ('physical_enriched','history','neural'):
                for mode in ('privileged','sensor'):
                    key=(condition,family,900000+fi*1000+i,name,mode)
                    if key in seen:continue
                    row=closed_episode(models[name],name,family,900000+fi*1000+i,condition,0,CEMConfig(),80,
                        out/'trajectories'/f'{condition}_{family}_{i}_{name}_{mode}.csv',observation_mode=mode)
                    rows.append(row)
                    with raw.open('a',encoding='utf-8') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
            write_csv(out/'metrics.csv',rows)
            print(condition,family,i,'episodes',len(rows),flush=True)
write_json(out/'completeness.json',dict(expected=96,actual=len(rows),
    status='verified_complete' if len(rows)==96 else 'provisional_incomplete'))
