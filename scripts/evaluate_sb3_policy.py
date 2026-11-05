"""Load a frozen local SB3 policy and replay explicit validation/test episodes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from stable_baselines3 import PPO
from run_sb3_rl import EvaluationPolicy
from falconworld.rl_evaluation import evaluate_policy,evaluation_cases,summarize
from falconworld.experiments import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--preset',choices=('smoke','standard'),default='smoke')
    p.add_argument('--split',choices=('val','test'),default='val')
    args=p.parse_args();torch.set_num_threads(1)
    checkpoint=args.checkpoint.resolve();model=PPO.load(checkpoint,device='cpu')
    rows=evaluate_policy(EvaluationPolicy(model),evaluation_cases(args.seed,args.preset,args.split),args.output,
                         method='loaded_policy',training_seed=args.seed)
    summarize(rows,args.output,reference='loaded_policy')
    write_json(args.output/'checkpoint.json',dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        evaluation_only=True,training=False,vehicle_io=False))
    print(f'Loaded-policy evaluation verified: {args.output.resolve()}',flush=True)


if __name__=='__main__':main()
