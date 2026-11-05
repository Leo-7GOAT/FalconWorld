"""Train/export/validate the explicitly SYNTHETIC low-speed deployment path."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.models import MLP
from falconworld.r550_worldmodel import DT,LowSpeedModel,LowSpeedMPC,export_onnx,physical
from scipy.optimize import least_squares


def generate(seed,steps=240):
    rng=np.random.default_rng(seed);state=np.zeros(3);rows=[]
    for t in range(steps):
        if t%20==0:command=float(rng.uniform(0,.3))
        next_state=physical(state,command,[.50,.25,2.0])[0]
        rows.append((state.copy(),command,next_state.copy()));state=next_state
    return rows


def flatten(episodes):
    rows=[r for episode in episodes for r in episode]
    return np.asarray([r[0] for r in rows]),np.asarray([r[1] for r in rows]),np.asarray([r[2] for r in rows])


def error(model,episodes):
    result={}
    for seconds in [.5,1.,3.]:
        horizon=round(seconds/DT);per_episode=[]
        for episode in episodes:
            squared=[]
            for start in range(0,len(episode)-horizon,horizon):
                state=episode[start][0]
                for j in range(horizon):state=model.step(state,episode[start+j][1])
                squared.append((state[0]-episode[start+horizon-1][2][0])**2)
            per_episode.append(float(np.sqrt(np.mean(squared))))
        result[str(seconds)]=dict(speed_rmse_m_s=float(np.mean(per_episode)),episode_values=per_episode)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/r550/speed_model');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    split={'train':list(range(100,124)),'validation':list(range(200,208)),'test':list(range(300,312))}
    datasets={key:[generate(seed) for seed in seeds] for key,seeds in split.items()}
    s,u,y=flatten(datasets['train']);sv,uv,yv=flatten(datasets['validation'])
    simple=least_squares(lambda p:physical(s,u,[p[0],p[1],0])[:,0]-y[:,0],[.5,.25],bounds=([.1,.1],[2,2])).x
    enriched=least_squares(lambda p:physical(s,u,p)[:,0]-y[:,0],[.5,.25,1],bounds=([.1,.1,0],[2,2,5])).x
    base=LowSpeedModel([*simple,0]);strong=LowSpeedModel(enriched)
    x=np.column_stack([s,u]);xv=np.column_stack([sv,uv])
    net=MLP().fit(x,(y-base.step(s,u))[:,:1],xv,(yv-base.step(sv,uv))[:,:1],seed=17,epochs=80)
    residual=LowSpeedModel(base.parameters,net.arrays(''))
    models={'simple_physics':base,'enriched_physics':strong,'residual_worldmodel':residual}
    for name,model in models.items():model.save(out/(name+'.npz'))
    residual=LowSpeedModel.load(out/'residual_worldmodel.npz');models['residual_worldmodel']=residual
    np.savez(out/'test_data.npz',states=flatten(datasets['test'])[0],commands=flatten(datasets['test'])[1],next_states=flatten(datasets['test'])[2])
    export_onnx(residual,out/'residual_worldmodel.onnx')
    import onnxruntime as ort
    session=ort.InferenceSession(str(out/'residual_worldmodel.onnx'),providers=['CPUExecutionProvider'])
    st,ut,_=flatten(datasets['test'])
    actual=session.run(None,dict(state=st.astype(np.float32),command=ut[:,None].astype(np.float32)))[0]
    discrepancy=float(np.max(np.abs(actual-residual.step(st,ut))))
    if discrepancy>1e-6:raise AssertionError('Low-speed ONNX disagrees with saved numpy model')
    results=dict(prediction={name:error(model,datasets['test']) for name,model in models.items()},onnx_max_abs_difference=discrepancy,closed_loop=[])
    for name,model in models.items():
        for seed in [401,402,403]:
            planner=LowSpeedMPC(model,seed=seed);state=np.zeros(3);trace=[]
            for step in range(240):
                target=.2 if step<80 else (.05 if step<160 else .15)
                start=time.perf_counter();command,info=planner.plan(state,target)
                cycle=time.perf_counter()-start
                fallback=info['fallback'] or cycle>DT
                if fallback:command=0.
                next_state=physical(state,command,[.50,.25,2.0])[0]
                trace.append(dict(step=step,timestamp=(step+1)*DT,speed=next_state[0],target=target,
                    speed_reference_m_s=command,cycle_ms=cycle*1000,fallback=int(fallback),
                    mode='SYNTHETIC_CLOSED_LOOP',sent=False,acknowledged=False))
                state=next_state
            with (out/('closed_%s_%d.csv'%(name,seed))).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
            cycles=[r['cycle_ms'] for r in trace]
            results['closed_loop'].append(dict(model=name,seed=seed,
                speed_rmse_m_s=float(np.sqrt(np.mean([(r['speed']-r['target'])**2 for r in trace]))),
                progress_m=float(sum(r['speed']*DT for r in trace)),cycle_p50_ms=float(np.percentile(cycles,50)),
                cycle_p95_ms=float(np.percentile(cycles,95)),timeout_rate=float(np.mean(np.asarray(cycles)>DT*1000)),
                fallback_count=sum(r['fallback'] for r in trace),sent=0,acknowledged=0))
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    manifest=dict(origin='SYNTHETIC_UNCALIBRATED',dt=DT,split=split,model_seed=17,
        state=['speed_m_s','previous_speed_m_s','previous_SENT_speed_reference_m_s'],action='speed_reference_m_s',
        limits=dict(speed_reference_m_s=[0,.3],reference_slew_m_s_per_step=.015),
        environment_assumptions=dict(accel_tau_s=.5,decel_tau_s=.25,tau_speed_factor_s_m=2.,delay_samples=1),
        initialization='two equal observed speeds, previous sent command zero; require fresh timestamp history',
        safety='offline/ROS shadow only; model origin does not authorize transport',
        actual_R550_calibration=False,training_log=net.log,
        model_sha256=hashlib.sha256((out/'residual_worldmodel.npz').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),onnx_max_abs_difference=discrepancy,prediction=results['prediction']),indent=2))


if __name__=='__main__':main()
