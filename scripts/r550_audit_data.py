"""Audit existing recordings without inventing missing action timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    p=argparse.ArgumentParser();p.add_argument('--stationary',required=True);p.add_argument('--dynamic',required=True)
    p.add_argument('--output',default='outputs/r550/data_audit.json');args=p.parse_args()
    stationary=Path(args.stationary);dynamic=Path(args.dynamic)
    source_files=[stationary]+[dynamic/name for name in ['result.json','samples.jsonl','events.jsonl','transmit.accepted.bin','receive.rx.bin']]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    kinds={}
    with stationary.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);kinds[row['kind']]=kinds.get(row['kind'],0)+1
    summary=json.loads((dynamic/'result.json').read_text(encoding='utf-8'))
    samples=[json.loads(line) for line in (dynamic/'samples.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (dynamic/'events.jsonl').read_text().splitlines()]
    wire=(dynamic/'transmit.accepted.bin').read_bytes();commands=[]
    if len(wire)%11:raise ValueError('Partial TX frame')
    for i in range(0,len(wire),11):
        frame=wire[i:i+11];checksum=0
        for b in frame[:9]:checksum^=b
        if frame[0]!=0x7b or frame[-1]!=0x7d or frame[9]!=checksum:
            raise ValueError('Invalid accepted TX wire frame')
        commands.append(tuple(v/1000 for v in struct.unpack('>hhh',frame[3:9])))
    result=dict(source_sha256=hashes,stationary=dict(kinds=kinds,has_logged_actions=False,
        eligible_for_action_conditioned_training=False,reason='receive-only stationary capture has no speed reference or TX records'),
        dynamic=dict(source=str(dynamic),episodes=1,samples=len(samples),
            feedback_timestamp='host collection elapsed time, not MCU sample time',
            nonzero_tx_events=sum(r['kind']=='tx_nonzero' for r in events),
            zero_tx_timestamp_events=sum(r['kind']=='tx_zero' for r in events),
            accepted_tx_frames=len(commands),unique_wire_commands=sorted(set(commands)),
            max_feedback_speed_m_s=max(r['vx'] for r in samples),
            max_abs_feedback_yawrate_rad_s=max(abs(r['wz']) for r in samples),
            disconnect_stop_verified=summary['disconnect_stop_verified'],
            train_validation_test_episode_split_possible=False,
            reason='one bounded pulse; zero TX times absent; no independent multi-level episodes or measured wheel/ground condition'),
        conclusion='No defensible real R550 train/validation/test speed-response dataset. Existing files retained; deployment model remains synthetic/un-calibrated.',
        unlock='Log multiple authorized independent low-speed episodes with monotonic timestamped SENT speed references and observed speeds, verified semantics/geometry/stop behavior; split whole episodes before fitting.')
    for f in source_files:
        if hashlib.sha256(f.read_bytes()).hexdigest()!=hashes[str(f)]:raise RuntimeError('Source changed')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
