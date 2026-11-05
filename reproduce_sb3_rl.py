"""One-command strong-PPO tests, frozen experiments, independent audit and report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=('smoke','standard','full'),default='smoke')
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();root=Path(__file__).resolve().parent
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[['-m','pytest','-q'],['run_sb3_rl.py','--preset',args.preset,'--workers',str(args.workers)],
        ['scripts/audit_sb3_rl.py','--run',f'outputs/rl_sb3/{args.preset}'],
        ['scripts/report_sb3_rl.py','--input',f'outputs/rl_sb3/{args.preset}']]
    for cmd in commands:
        print('RUN',sys.executable,*cmd,flush=True)
        subprocess.run([sys.executable,*cmd],cwd=root,env=env,check=True)
    print('Strong PPO reproduction verified. Vehicle deployment/realtime remain deferred.',flush=True)


if __name__=='__main__':main()
