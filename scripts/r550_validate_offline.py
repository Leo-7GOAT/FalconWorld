"""Verify a historical receive-only JSONL and run stationary shadow replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.adapters import R550Config,R550ShadowAdapter
from falconworld.contracts import Action,Observation


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--output',default='outputs/r550');args=p.parse_args()
    source=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    # Candidate source-code geometry from existing FalconPlan review. No claim
    # of measurement; policy limits here apply only to a non-actuating replay.
    config=R550Config(.144,.3,.5,.3,.3,.5,.1,.2,
        '/UNVERIFIED/cmd_vel','/falcon/r550/rx/velocity',False,
        'FalconPlan hal/r550/config.py candidate .144m; steering/rate are shadow policy, not calibrated')
    adapter=R550ShadowAdapter(config);records=[];decoded=[];invalid=0;nonzero=0
    with source.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row.get('kind')!='state':continue
            payload=row['payload'];raw=bytes.fromhex(payload['raw_frame_hex'])
            bcc=0
            for byte in raw[:22]:bcc^=byte
            if len(raw)!=24 or raw[0]!=0x7b or raw[-1]!=0x7d or bcc!=raw[22]:
                invalid+=1;continue
            vx,vy,wz=struct.unpack('>hhh',raw[2:8]);vx/=1000;vy/=1000;wz/=1000
            if (vx,vy,wz)!=(payload['vx_m_s'],payload['vy_m_s'],payload['wz_rad_s']):
                raise RuntimeError('Stored decoded telemetry differs from independent wire decode')
            timestamp=float(payload['received_monotonic_s'])
            if vx<0:raise ValueError('Forward-only observation cannot silently clamp reverse data')
            nonzero+=int(abs(vx)>0 or abs(wz)>0)
            records.append(adapter.shadow(Observation(timestamp,vx),Action(0),timestamp))
            decoded.append(dict(timestamp=timestamp,speed=vx,command=0,steering=0,
                                source='REAL_HISTORICAL_RECEIVE_ONLY',wz_rad_s=wz))
    with (out/'real_stationary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(decoded[0]));w.writeheader();w.writerows(decoded)
    with (out/'real_stationary_shadow.jsonl').open('w',encoding='utf-8') as f:
        for row in records:f.write(json.dumps(row)+'\n')
    summary=dict(source=str(source),source_sha256=digest,source_bytes=source.stat().st_size,
        valid_frames=len(decoded),invalid_state_frames=invalid,nonzero_velocity_frames=nonzero,
        sent_commands=0,acknowledged_commands=0,mode='offline historical real data, shadow output only',
        clock='host read completion, not MCU sample time',geometry_verified=False,
        speed_response_identification='not possible from stationary data',
        maximum_shadow_speed=max(r['target_speed_m_s'] for r in records),
        note='Source file preserved unchanged; no serial/socket transport imported')
    (out/'real_replay_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    from dataclasses import asdict
    (out/'candidate_shadow_config.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed during replay')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
