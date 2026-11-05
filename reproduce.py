"""Validated local entrypoint; external CARLA/ROS2 acceptance has separate commands."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=('smoke','standard','full'),default='smoke')
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    os.chdir(root)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(*arguments):
        print('RUN',sys.executable,*arguments,flush=True)
        subprocess.run([sys.executable,*arguments],cwd=root,env=env,check=True)
    run('-m','pytest','-q')
    output=f'outputs/{a.preset}_verified'
    run('run_experiments.py','--preset',a.preset,'--output',output,'--workers',str(a.workers))
    run('run_ackermann.py','--preset',a.preset,'--output',f'outputs/ackermann/{a.preset}_verified')
    run('scripts/export_models.py','--models',f'{output}/seed0/models',
        '--data',f'{output}/data','--output',f'{output}/onnx')
    run('scripts/run_diagnostics.py','--models',f'{output}/seed0/models',
        '--data',f'{output}/data','--output',f'{output}/diagnostics')
    run('scripts/build_report.py')
    print(f'Local {a.preset} reproduction completed. External acceptance: docs/CARLA.md and docs/R550.md',flush=True)


if __name__=='__main__':main()
