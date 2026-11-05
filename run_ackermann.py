"""Reproduce synthetic small-scale Ackermann experiments (no hardware motion)."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import csv
import platform
import hashlib
import numpy as np
from falconworld.ackermann import (AckermannConfig, Route, PhysicalModel,
    ResidualModel, AckermannCEM, ClassicTracker, generate_dataset, train_models,
    open_loop, open_loop_ood, simulate, write_csv, bounded_action, step_ackermann)
from falconworld.planning import CEMConfig, CEMOptimizer

PRESETS={
    'smoke':dict(seeds=[527],pairs_per_seed=1,horizon=20,candidates=32,iterations=2,budgets=['candidates']),
    'standard':dict(seeds=[527,528,529],pairs_per_seed=10,horizon=25,candidates=48,iterations=2,budgets=['candidates','time']),
    'full':dict(seeds=[527,528,529],pairs_per_seed=30,horizon=30,candidates=96,iterations=3,budgets=['candidates','time']),
}

def bootstrap(values,seed=1):
    a=np.asarray(values,dtype=float)
    if not len(a):return [None,None]
    rng=np.random.default_rng(seed)
    b=np.mean(a[rng.integers(0,len(a),(2000,len(a)))],axis=1)
    return [float(x) for x in np.percentile(b,[2.5,97.5])]


def summarize(rows):
    summaries=[]
    for budget in sorted(set(r['budget'] for r in rows)):
        for family in sorted(set(r['family'] for r in rows)):
            for condition in sorted(set(r['condition'] for r in rows)):
                for controller in sorted(set(r['controller'] for r in rows)):
                    a=[r for r in rows if r['budget']==budget and r['family']==family and r['condition']==condition and r['controller']==controller]
                    if not a:continue
                    item=dict(budget=budget,family=family,condition=condition,controller=controller,episodes=len(a),
                        collisions=sum(r['collision'] for r in a),road_exits=sum(r['road_exit'] for r in a),completed=sum(r['completed'] for r in a))
                    for key in ('progress_m','tracking_rmse_m','speed_rmse_mps','jerk_rms_mps3','fallback_count','cycle_p50_ms','cycle_p95_ms','timeout_rate'):
                        item[key]=float(np.mean([r[key] for r in a]))
                    summaries.append(item)
    effects=[]
    for budget in sorted(set(r['budget'] for r in rows)):
        for baseline in ('calibrated_physics_mpc','rich_physics_mpc'):
            left={(r['seed'],r['family'],r['pair'],r['condition']):r for r in rows if r['budget']==budget and r['controller']==baseline}
            for family in ('straight','curve','variable_speed','obstacle'):
                for condition in sorted(set(r['condition'] for r in rows)):
                    paired=[(r,left[(r['seed'],r['family'],r['pair'],r['condition'])]) for r in rows if r['controller']=='learned_residual_mpc' and r['budget']==budget and r['family']==family and r['condition']==condition]
                    if not paired:continue
                    for metric in ('tracking_rmse_m','speed_rmse_mps','progress_m','jerk_rms_mps3'):
                        delta=[a[metric]-b[metric] for a,b in paired]
                        seed_values={seed:[a[metric]-b[metric] for a,b in paired if a['seed']==seed] for seed in sorted(set(a['seed'] for a,b in paired))}
                        seed_delta=[float(np.mean(v)) for v in seed_values.values()]
                        effects.append(dict(budget=budget,family=family,condition=condition,baseline=baseline,metric=metric,
                            paired_episodes=len(delta),learned_minus_physics=float(np.mean(delta)),bootstrap_episode_ci95=bootstrap(delta),
                            independent_training_fits=len(seed_values),training_seed_means=seed_delta,
                            bootstrap_training_seed_ci95=bootstrap(seed_delta)))
    return summaries,effects


def plots(out,all_logs,rows,open_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,family in zip(axes.ravel(),('straight','curve','variable_speed','obstacle')):
        chosen=[(k,v) for k,v in all_logs.items() if k[1]==family and k[2]==0 and k[3]=='candidates']
        for key,log in chosen:
            ax.plot([r['x_m'] for r in log],[r['y_m'] for r in log],label=key[0])
        route=Route(family)
        x=np.linspace(0,32,300)
        if chosen:
            log=chosen[0][1]
            ax.plot([r['x_m'] for r in log],[r['y_m']-r['tracking_error_m'] for r in log],'k--',label='actual scenario reference')
        else:
            ax.plot(x,route.reference(x),'k--',label='reference')
        for ox,oy,r in route.obstacles:ax.add_patch(plt.Circle((ox,oy),r,color='black',alpha=.5))
        ax.set(title=family,xlabel='rear axle x [m]',ylabel='rear axle y [m]')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Actual closed-loop trajectories (lateral axes enlarged)',fontsize=11)
    fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key,label in zip(axes,('position_error_m','yaw_error_rad','speed_error_mps'),('position [m]','yaw [rad]','speed [m/s]')):
        for model in sorted(set(r['model'] for r in open_rows)):
            h=sorted(set(r['horizon_s'] for r in open_rows))
            y=[np.mean([r[key] for r in open_rows if r['model']==model and r['horizon_s']==t]) for t in h]
            ax.plot(h,y,'o-',label=model)
        ax.set(xlabel='recursive horizon [s]',ylabel='mean absolute '+label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'open_loop.png',dpi=150);plt.close(fig)


def action_branches(models,out,cfg):
    rows=[];initial=np.array([0.,0.,0.,3.,0.,0.])
    for label,target in [('brake_left',[-2.,.2]),('accelerate_right',[1.5,-.2]),('hold',[0.,0.])]:
        truth=initial.copy();pred={m.name:initial.copy() for m in models};previous=np.zeros(2)
        for k in range(30):
            u=bounded_action(target,previous,cfg);truth=step_ackermann(truth,u,cfg)
            for m in models:
                pred[m.name]=m.predict_batch(pred[m.name][None],u[None])[0]
                p=pred[m.name]
                rows.append(dict(branch=label,step=k+1,model=m.name,accel_cmd=float(u[0]),steer_cmd=float(u[1]),
                    true_x=float(truth[0]),true_y=float(truth[1]),true_speed=float(truth[3]),
                    pred_x=float(p[0]),pred_y=float(p[1]),pred_speed=float(p[3]),position_error_m=float(np.linalg.norm(p[:2]-truth[:2]))))
            previous=u
    write_csv(out/'action_branches.csv',rows)
    return rows


def animation(out,all_logs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    chosen=[(key,log) for key,log in all_logs.items() if key[1]=='obstacle' and key[3]=='candidates']
    if not chosen:return
    fig,ax=plt.subplots(figsize=(9,3));ax.set(xlim=(-1,33),ylim=(-2,4),xlabel='rear axle x [m]',ylabel='rear axle y [m]')
    route=Route('obstacle');x=np.linspace(0,32,200);ax.plot(x,route.reference(x),'k--',label='reference')
    for ox,oy,r in route.obstacles:ax.add_patch(plt.Circle((ox,oy),r,color='black',alpha=.6))
    lines=[];dots=[]
    for key,log in chosen:
        line,=ax.plot([],[],label=key[0]);dot,=ax.plot([],[],'o',color=line.get_color());lines.append(line);dots.append(dot)
    title=ax.set_title('Actual closed-loop logs');ax.legend(fontsize=7,loc='lower right');ax.grid(alpha=.2)
    def frame(index):
        for j,(_,log) in enumerate(chosen):
            n=min(index+1,len(log));lines[j].set_data([r['x_m'] for r in log[:n]],[r['y_m'] for r in log[:n]])
            dots[j].set_data([log[n-1]['x_m']],[log[n-1]['y_m']])
        title.set_text(f'Actual closed-loop logs: t={index*.1:.1f} s')
        return lines+dots+[title]
    anim=FuncAnimation(fig,frame,frames=range(0,max(len(v) for _,v in chosen),2),interval=80,blit=True)
    anim.save(out/'obstacle_comparison.gif',writer=PillowWriter(fps=12));plt.close(fig)



def read_csv_numeric(path):
    rows=list(csv.DictReader(Path(path).open(encoding='utf8')))
    for row in rows:
        for key,value in row.items():
            try:row[key]=float(value)
            except (ValueError,TypeError):pass
    return rows


def render_existing(out):
    out=Path(out);rows=read_csv_numeric(out/'closed_loop.csv');pred=read_csv_numeric(out/'open_loop.csv')
    logs={};seed=min(int(r['seed']) for r in rows)
    seedout=out/f'seed_{seed}'
    for family in ('straight','curve','variable_speed','obstacle'):
        for ctrl in ('pure_pursuit_pid','calibrated_physics_mpc','rich_physics_mpc','learned_residual_mpc'):
            path=seedout/f'candidates_{family}_0_{ctrl}.csv'
            if path.exists():logs[(ctrl,family,0,'candidates')]=read_csv_numeric(path)
    plots(out,logs,rows,pred);animation(out,logs)
    pooled={}
    for row in rows:
        key=(row['budget'],row['controller'],row['family'],row['condition'])
        filename=f"{row['budget']}_{row['family']}_{int(row['pair'])}_{row['controller']}.csv"
        log=read_csv_numeric(out/f"seed_{int(row['seed'])}"/filename)
        pooled.setdefault(key,[]).extend(r['latency_ms'] for r in log)
    latency=[]
    for key,values in pooled.items():
        a=np.asarray(values)
        latency.append(dict(zip(('budget','controller','family','condition'),key),cycles=len(a),
            cycle_p50_ms=float(np.percentile(a,50)),cycle_p95_ms=float(np.percentile(a,95)),
            deadline_ms=100.,timeouts=int(np.sum(a>100)),timeout_rate=float(np.mean(a>100))))
    write_csv(out/'latency_summary.csv',latency)
    seed_means=[]
    for seed in sorted(set(r['seed'] for r in rows)):
        for budget in sorted(set(r['budget'] for r in rows)):
            for ctrl in sorted(set(r['controller'] for r in rows)):
                a=[r for r in rows if r['seed']==seed and r['budget']==budget and r['controller']==ctrl]
                if a:seed_means.append(dict(seed=int(seed),budget=budget,controller=ctrl,episodes=len(a),
                    tracking_rmse_m=float(np.mean([r['tracking_rmse_m'] for r in a])),
                    speed_rmse_mps=float(np.mean([r['speed_rmse_mps'] for r in a])),
                    completion_rate=float(np.mean([r['completed'] for r in a]))))
    write_csv(out/'training_seed_means.csv',seed_means)
    summary,effects=summarize(rows);write_csv(out/'summary.csv',summary)
    (out/'paired_effects.json').write_text(json.dumps(effects,indent=2),encoding='utf8')
    worst=[]
    for budget in sorted(set(r['budget'] for r in rows)):
        for ctrl in sorted(set(r['controller'] for r in rows)):
            a=[r for r in rows if r['budget']==budget and r['controller']==ctrl]
            item=dict(max(a,key=lambda r:r['tracking_rmse_m']))
            item['selection']='highest episode tracking RMSE for this controller and budget; not necessarily a collision'
            item['raw_log']=f"seed_{int(item['seed'])}/{item['budget']}_{item['family']}_{int(item['pair'])}_{item['controller']}.csv"
            worst.append(item)
    (out/'failure_cases.json').write_text(json.dumps(worst,indent=2),encoding='utf8')
    ood=[];manifests=[]
    import hashlib
    for seed in sorted(set(int(r['seed']) for r in rows)):
        folder=out/f'seed_{seed}'
        models=[PhysicalModel.load(folder/'physical.npz'),PhysicalModel.load(folder/'rich_physical.npz'),ResidualModel.load(folder/'residual.npz')]
        with np.load(folder/'dataset.npz',allow_pickle=False) as data:
            prediction=open_loop_ood(models,data,folder,models[0].cfg)
        for r in prediction:r['seed']=seed
        ood.extend(prediction)
        files={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('physical.npz','rich_physical.npz','residual.npz','dataset.npz')}
        manifests.append(dict(seed=seed,model_schema='ackermann-state6-action2-v1',model_types=[m.name for m in models],sha256=files,config=asdict(models[0].cfg)))
    write_csv(out/'open_loop_ood.csv',ood)
    (out/'model_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preset',choices=PRESETS,default='smoke');parser.add_argument('--output',default=None);parser.add_argument('--pairs',type=int,default=None);parser.add_argument('--skip-time',action='store_true');parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();spec=dict(PRESETS[args.preset]);out=Path(args.output or 'outputs/ackermann/'+args.preset);out.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        render_existing(out);return
    if args.pairs is not None:spec['pairs_per_seed']=args.pairs
    if args.skip_time:spec['budgets']=['candidates']
    registration={'preset':args.preset,'spec':spec,'primary':'paired episode tracking RMSE, completion and collision; speed RMSE and progress guard against stopping',
        'split':'whole episodes 0:36 train,36:45 validation,45:54 frozen test; no test-based tuning',
        'tuning':'ridge from [1e-6,1e-4,.01] chosen by validation error; controller gains fixed before tests',
        'model_observation':'privileged [rear axle x,y,yaw,speed,actuator acceleration,road-wheel steer]; no hidden condition',
        'synthetic_assumptions':'small-scale geometry is generic, not R550; speed-dependent lag, acceleration asymmetry and yaw attenuation',
        'compute':'candidate budget identical within row; time budget candidates calibrated by independent full-MPC profile to 1.2 times the slowest equal-candidate median; realized full-cycle distributions reported',
        'deadline_ms':100,'host':platform.platform(),'numpy':np.__version__}
    registration['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path('falconworld/ackermann.py'),Path('falconworld/planning.py'))}
    registration_path=out/'experiment_registration.json'
    if registration_path.exists() and json.loads(registration_path.read_text(encoding='utf8'))!=registration:
        raise ValueError('Frozen source/configuration changed; use a fresh output directory')
    registration_path.write_text(json.dumps(registration,indent=2),encoding='utf8')
    for source in registration['source_sha256']:
        original=Path(source)
        relative=Path('run_ackermann.py') if original.is_absolute() else original
        target=out/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(original.read_bytes())
    cfg=AckermannConfig();rows=read_csv_numeric(out/'closed_loop.csv') if (out/'closed_loop.csv').exists() else []
    keys={(int(r['seed']),r['budget'],r['family'],int(r['pair']),r['controller']) for r in rows}
    if len(keys)!=len(rows):raise ValueError('Duplicate closed-loop episode identities')
    open_rows=[];all_logs={};profiles=[];started=perf_counter()
    for seed in spec['seeds']:
        seedout=out/f'seed_{seed}';seedout.mkdir(exist_ok=True)
        if (seedout/'dataset.npz').exists():
            manifest=json.loads((seedout/'split_manifest.json').read_text(encoding='utf8'))
            if hashlib.sha256((seedout/'dataset.npz').read_bytes()).hexdigest()!=manifest['sha256']:
                raise ValueError('Dataset fingerprint mismatch')
            with np.load(seedout/'dataset.npz',allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
        else:data,_=generate_dataset(seedout,cfg,seed)
        if all((seedout/name).exists() for name in ('physical.npz','rich_physical.npz','residual.npz')):
            models=[PhysicalModel.load(seedout/'physical.npz'),PhysicalModel.load(seedout/'rich_physical.npz'),ResidualModel.load(seedout/'residual.npz')]
        else:models=train_models(data,seedout,cfg)
        artifacts={name:hashlib.sha256((seedout/name).read_bytes()).hexdigest()
            for name in ('dataset.npz','physical.npz','rich_physical.npz','residual.npz')}
        frozen=seedout/'artifacts_sha256.json'
        if frozen.exists() and json.loads(frozen.read_text())!=artifacts:
            raise ValueError('Frozen model/data artifact changed')
        frozen.write_text(json.dumps(artifacts,indent=2),encoding='utf8')
        action_branches(models,seedout,cfg)
        pred=open_loop(models,data,seedout,cfg)
        for r in pred:r['seed']=seed
        open_rows.extend(pred)
        initial=np.array([0.,.1,0.,2.4,0.,0.])
        measured={};matched_candidates={};matching_profiles={}
        def profile_cycle(model,candidates,repeats=3):
            p=AckermannCEM(model,CEMOptimizer(CEMConfig(horizon=spec['horizon'],candidates=candidates,iterations=spec['iterations'],elites=min(8,candidates))))
            elapsed=[]
            for i in range(repeats+1):
                t=perf_counter();p.action(initial,0.,Route('curve'),np.zeros(2))
                if i:elapsed.append((perf_counter()-t)*1000)
            return float(np.median(elapsed))
        profile_path=out/'profiles.json'
        saved_profiles=json.loads(profile_path.read_text(encoding='utf8')) if profile_path.exists() else []
        frozen_profile=next((r for r in saved_profiles if r['seed']==seed),None)
        if frozen_profile is None:
            if any(int(r['seed'])==seed for r in rows):
                raise ValueError('Existing episodes require their original frozen validation profile')
            for model in models:
                measured[model.name]=profile_cycle(model,spec['candidates'])
            target_time=max(measured.values())*1.2
            if 'time' in spec['budgets']:
                # Tune computational counts using an independent profiling state.
                # Fixed overhead makes proportional candidate scaling inaccurate.
                for model in models:
                    cache={spec['candidates']:measured[model.name]};lo,hi=8,2048
                    for _ in range(7):
                        mid=(lo+hi)//2;elapsed=profile_cycle(model,mid)
                        cache[mid]=elapsed
                        if elapsed<target_time:lo=mid+1
                        else:hi=mid-1
                        if hi<lo:break
                    chosen=min(cache,key=lambda n:abs(cache[n]-target_time))
                    matched_candidates[model.name]=chosen
                    matching_profiles[model.name]={'chosen_candidates':chosen,'median_ms':cache[chosen],
                        'relative_target_error':cache[chosen]/target_time-1,'measurements':cache}
            frozen_profile=dict(seed=seed,cycle_ms=measured,target_ms=target_time,time_matching=matching_profiles)
            saved_profiles.append(frozen_profile)
            temporary=profile_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(saved_profiles,indent=2),encoding='utf8');temporary.replace(profile_path)
        else:
            measured=frozen_profile['cycle_ms'];target_time=frozen_profile['target_ms']
            matching_profiles=frozen_profile['time_matching']
            matched_candidates={name:profile['chosen_candidates'] for name,profile in matching_profiles.items()}
        for row in rows:
            if int(row['seed'])!=seed or row['controller']=='pure_pursuit_pid':continue
            expected_count=spec['candidates'] if row['budget']=='candidates' else matched_candidates[row['controller'].removesuffix('_mpc')]
            if int(row['candidates'])!=expected_count:raise ValueError('Episode candidate count differs from frozen validation profile')
        profiles.append(frozen_profile)
        rng=np.random.default_rng(seed+1000)
        scenarios=[]
        for family in ('straight','curve','variable_speed','obstacle'):
            for pair in range(spec['pairs_per_seed']):
                condition=1.3 if pair%2 else 1.
                route=Route(family,phase=float(rng.uniform(-.25,.25)))
                initial=np.array([0.,rng.uniform(-.15,.15),rng.uniform(-.025,.025),rng.uniform(2.2,2.8),0.,0.])
                scenarios.append((family,pair,condition,route,initial))
        for budget in spec['budgets']:
            for family,pair,condition,route,initial in scenarios:
                controllers=[ClassicTracker(cfg)]
                for model in models:
                    candidates=spec['candidates'] if budget=='candidates' else matched_candidates[model.name]
                    optimizer=CEMOptimizer(CEMConfig(horizon=spec['horizon'],candidates=candidates,iterations=spec['iterations'],elites=min(8,candidates)))
                    controllers.append(AckermannCEM(model,optimizer))
                for ctrl in controllers:
                    identity=(seed,budget,family,pair,ctrl.name)
                    if identity in keys:continue
                    metric,log=simulate(ctrl,route,initial,cfg=cfg,condition=condition)
                    metric.update(seed=seed,pair=pair,budget=budget,candidates=0 if isinstance(ctrl,ClassicTracker) else ctrl.optimizer.config.candidates)
                    rows.append(metric);keys.add(identity)
                    name=f'{budget}_{family}_{pair}_{ctrl.name}'
                    write_csv(seedout/(name+'.csv'),log)
                    if pair==0 and seed==spec['seeds'][0]:all_logs[(ctrl.name,family,pair,budget)]=log
                write_csv(out/'closed_loop.csv',rows)
                print(f'{args.preset} seed={seed} {budget} {family} pair={pair} runs={len(rows)} elapsed={perf_counter()-started:.1f}s',flush=True)
    write_csv(out/'open_loop.csv',open_rows)
    summary,effects=summarize(rows);write_csv(out/'summary.csv',summary)
    (out/'paired_effects.json').write_text(json.dumps(effects,indent=2),encoding='utf8')
    render_existing(out)
    expected={(seed,budget,family,pair,ctrl) for seed in spec['seeds'] for budget in spec['budgets']
        for family in ('straight','curve','variable_speed','obstacle') for pair in range(spec['pairs_per_seed'])
        for ctrl in ('pure_pursuit_pid','calibrated_physics_mpc','rich_physics_mpc','learned_residual_mpc')}
    complete=keys==expected
    (out/'completeness.json').write_text(json.dumps(dict(expected=len(expected),actual=len(rows),unique=len(keys),
        missing=[list(k) for k in sorted(expected-keys)],unexpected=[list(k) for k in sorted(keys-expected)],
        verified_complete=complete),indent=2),encoding='utf8')
    if not complete:raise ValueError('Experiment incomplete; inspect completeness.json')
    (out/'run_metadata.json').write_text(json.dumps({'elapsed_s':perf_counter()-started,'episodes':len(rows),'training_seeds':spec['seeds'],'model_types':['calibrated constant physics','rich calibrated physics','learned nonlinear-basis residual'],'status':'verified local synthetic simulation'},indent=2),encoding='utf8')
    print(json.dumps({'output':str(out),'runs':len(rows),'elapsed_s':perf_counter()-started}),flush=True)

if __name__=='__main__':main()
