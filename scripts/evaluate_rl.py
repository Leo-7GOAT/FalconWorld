"""Evaluate a trusted local final PPO checkpoint, or summarize paired runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.rl_evaluation import evaluation_cases,evaluate_policy,summarize,write_training_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--method',default='ppo')
    p.add_argument('--training-seed',type=int,default=0)
    p.add_argument('--preset',choices=('smoke','standard'),default='smoke')
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summarize',type=Path,nargs='+',help='Explicit episodes.jsonl paths; no checkpoint needed')
    p.add_argument('--reference',default='ppo')
    p.add_argument('--cost-ledger',type=Path,help='Runner-recorded real/offline/model steps, updates, wall times')
    p.add_argument('--validation-curves',type=Path,help='JSON list: method/training_seed/online_steps/offline_transitions/val_return')
    args=p.parse_args()
    if args.summarize:
        rows=[json.loads(line) for path in args.summarize for line in path.read_text().splitlines() if line.strip()]
        ledger=json.loads(args.cost_ledger.read_text()) if args.cost_ledger else []
        if isinstance(ledger,dict):ledger=[ledger]
        curves=json.loads(args.validation_curves.read_text()) if args.validation_curves else []
        write_training_report(rows,args.output,reference=args.reference,ledgers=ledger,validation_rows=curves)
        print(json.dumps(dict(output=str(args.output),episodes=len(rows))))
        return
    if args.checkpoint is None:p.error('--checkpoint is required for policy evaluation')
    from falconworld.ppo_agent import PPOAgent
    agent=PPOAgent.load_checkpoint(args.checkpoint)
    rows=evaluate_policy(agent,evaluation_cases(args.training_seed,args.preset,args.split),
        args.output,method=args.method,training_seed=args.training_seed)
    provenance=dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),split=args.split,
        policy_gradient_updates=getattr(agent,'gradient_updates',None),
        policy_action_steps=getattr(agent,'action_steps',None),
        checkpoint_metadata=getattr(agent,'checkpoint_metadata',{}),
        cost_ledger=json.loads(args.cost_ledger.read_text()) if args.cost_ledger else None,
        sample_efficiency_scope='No cost estimate inferred from replay size; missing ledger means costs unverified')
    (args.output/'checkpoint_provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False))
    summarize(rows,args.output)
    print(json.dumps(dict(output=str(args.output),episodes=len(rows),checkpoint_sha256=provenance['checkpoint_sha256'])))


if __name__=='__main__':main()
