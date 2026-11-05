"""No-network/no-actuation log replay. Requires an explicit hardware config."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.adapters import R550Config,R550ShadowAdapter
from falconworld.contracts import Action,Observation


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',default='outputs/r550/shadow.jsonl')
    p.add_argument('--inject-timeout-step',type=int,default=-1)
    args=p.parse_args();adapter=R550ShadowAdapter(R550Config.load(args.config))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with Path(args.input).open(encoding='utf-8-sig',newline='') as source, out.open('w',encoding='utf-8') as dest:
        rows=csv.DictReader(source)
        count,fallback=0,0
        for row in rows:
            timestamp=float(row['timestamp']);speed=float(row['speed'])
            obs=Observation(timestamp,speed)
            action=Action(float(row.get('command',row.get('acceleration',0))),float(row.get('steering',0)))
            now=timestamp+(adapter.config.feedback_timeout_s+0.01 if count==args.inject_timeout_step else 0)
            record=adapter.shadow(obs,action,now)
            dest.write(json.dumps(record)+'\n');count+=1;fallback+=int(record['fallback'])
    print(json.dumps(dict(rows=count,fallback=fallback,output=str(out),mode='shadow',sent=0,acknowledged=0)))

if __name__=='__main__':main()
