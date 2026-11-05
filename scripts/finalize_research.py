"""Wait for frozen experiments, then verify the standard reproduction and report.

Run separately from the experiment writers. Never changes models or test settings.
The status file is a durable checkpoint, not a claim of research superiority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    if not path.exists():return {}
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoint(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    out=ROOT/'outputs/research_acceptance'
    out.mkdir(exist_ok=True)
    status=out/'status.json'
    while True:
        longitudinal=read(ROOT/'outputs/standard_verified/completeness.json')
        planar=read(ROOT/'outputs/ackermann/standard_verified/completeness.json')
        ready=(longitudinal.get('status')=='verified_complete' and planar.get('verified_complete') is True)
        progress=dict(stage='ready' if ready else 'waiting_for_frozen_experiments',
                      checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      longitudinal=longitudinal,ackermann=planar,
                      hardware='R550 deployment deferred by user; no network or motion actions')
        progress['longitudinal_recorded']=sum(sum(bool(line.strip()) for line in p.open(encoding='utf-8'))
            for p in (ROOT/'outputs/standard_verified').glob('seed*/closed_*.jsonl'))
        planar_rows=ROOT/'outputs/ackermann/standard_verified/closed_loop.csv'
        progress['ackermann_recorded']=max(0,sum(1 for _ in planar_rows.open(encoding='utf-8'))-1) if planar_rows.exists() else 0
        checkpoint(status,progress)
        if ready:break
        if not args.wait:
            print('Standard experiments incomplete; use --wait or resume the registered experiment commands.')
            return 2
        time.sleep(10)
    # Writers produce completeness immediately before plots. Allow those final
    # atomic writes to settle, and refuse to launch while either writer exists.
    # On this verified Windows host, CIM is available without extra packages.
    if os.name=='nt':
        query="Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match '(run_experiments|run_ackermann)\\.py.*standard_verified' } | Select-Object -ExpandProperty ProcessId"
        while subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip():
            time.sleep(5)
    python=ROOT/'.venv/Scripts/python.exe' if os.name=='nt' else Path(sys.executable)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    progress['stage']='verifying_standard_reproduction'
    checkpoint(status,progress)
    command=[str(python),'reproduce.py','--preset','standard','--workers','4']
    with (out/'standard_reproduction.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    progress.update(stage='local_research_verified' if result.returncode==0 else 'verification_failed',
                    command=command,exit_code=result.returncode,
                    wrapper_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                    for name in ('reproduce.py','reproduce.ps1')})
    checkpoint(status,progress)
    print(json.dumps(dict(stage=progress['stage'],exit_code=result.returncode,status=str(status))))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
