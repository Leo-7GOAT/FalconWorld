"""Replay speed-command telemetry through loaded world-model CEM without TX."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.r550_worldmodel import LowSpeedModel,LowSpeedMPC,DT


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',default='outputs/r550/speed_model/replay_shadow.jsonl');args=p.parse_args()
    model=LowSpeedModel.load(args.model);planner=LowSpeedMPC(model)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    previous_speed=0.;previous_time=None;count=0;timeouts=0
    with Path(args.input).open(newline='',encoding='utf-8') as source,out.open('w',encoding='utf-8') as dest:
        reader=csv.DictReader(source)
        required={'timestamp','speed','speed_reference_m_s','target'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Need timestamp/speed/speed_reference_m_s/target; unknown actual commands cannot be invented')
        for row in reader:
            stamp=float(row['timestamp']);speed=float(row['speed'])
            reason=''
            # The row's command generated its current speed observation. This is
            # the actual SENT reference from input telemetry, not our suggestion.
            reference=float(row['speed_reference_m_s'])
            if previous_time is None:
                previous_speed=speed
            elif not DT*.95<=stamp-previous_time<=DT*1.05:
                reason='feedback_period_mismatch'
            begin=time.perf_counter()
            if not reason:command,info=planner.plan([speed,previous_speed,reference],float(row['target']))
            else:command,info=0.,dict(reason=reason)
            elapsed=time.perf_counter()-begin
            if elapsed>DT:command=0.;reason='planner_timeout';timeouts+=1
            dest.write(json.dumps(dict(timestamp=stamp,observed_speed_m_s=speed,
                recorded_speed_reference_m_s=reference,shadow_speed_reference_m_s=command,
                reason=reason or info['reason'],cycle_ms=elapsed*1000,model_origin=model.origin,
                mode='OFFLINE_WORLD_MODEL_SHADOW',sent=False,acknowledged=False))+'\n')
            previous_speed,previous_time=speed,stamp;count+=1
    print(json.dumps(dict(rows=count,timeouts=timeouts,output=str(out),model_origin=model.origin,sent=0)))


if __name__=='__main__':main()
