"""Run PPO tests, registered training/evaluation, and the evidence report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=('smoke','standard','full'),default='smoke')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for command in (['-m','pytest','-q'],['run_rl.py','--preset',args.preset,'--workers',str(args.workers)],
                    ['scripts/report_rl.py','--input',f'outputs/rl_ppo/{args.preset}']):
        print('RUN',sys.executable,*command,flush=True)
        subprocess.run([sys.executable,*command],cwd=root,env=env,check=True)
    print('PPO local reproduction completed; hardware deployment remains deferred.',flush=True)


if __name__=='__main__':main()
