"""Preregistered synthetic friction mismatch; refit strong physical controls."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil,json
from time import perf_counter
import numpy as np
import torch
from stable_baselines3 import PPO
from falconworld.ackermann import PhysicalModel,bounded_action
from falconworld.ackermann_rl import applied_tracking,CFG,MAX_STEPS
from falconworld.ackermann_mismatch import (mismatch_truth,MismatchTrackingEnv,SteeringPhysical,
    SteeringResidual,TorchSteering,save_predictor,load_predictor)
from falconworld.ackermann_model_gradient import ModelGradientPPO
from run_ackermann_rl import CONFIG,sha,write
from run_ackermann_improvement import warm_roots

METHODS=('ppo','rich','affine','residual','mechanism')
NAMESPACES={'train':310_000_000,'val':320_000_000,'test':330_000_000,'ood':340_000_000}


def collect(seed,split,episodes):
    all_s=[];all_u=[];all_n=[];ids=[]
    for eid in range(episodes):
        episode_seed=NAMESPACES[split]+seed*100_000+eid;rng=np.random.default_rng(episode_seed)
        s=np.array([0,0,rng.uniform(-.2,.2),rng.uniform(.5,6),rng.uniform(-.5,.5),rng.uniform(-.3,.3)])
        previous=np.zeros(2)
        for step in range(100):
            if step%8==0:request=rng.uniform([-2.5,-.45],[1.8,.45])
            u=bounded_action(request,previous,CFG);n=mismatch_truth(s,u,1.3 if split=='ood' else 1.)
            all_s.append(s.copy());all_u.append(u.copy());all_n.append(n.copy());ids.append(episode_seed)
            s,previous=n,u
    return np.array(all_s),np.array(all_u),np.array(all_n),np.array(ids)


def prepare(root,seed):
    torch.set_num_threads(1);folder=root/f'seed{seed}'/'models';folder.mkdir(parents=True)
    train=collect(seed,'train',24);val=collect(seed,'val',6)
    rich=PhysicalModel(CFG,True).fit(*train[:3]);rich.save(folder/'rich.npz')
    affine=SteeringPhysical('affine').fit(*train[:3]);save_predictor(affine,folder/'affine.npz')
    mechanism=SteeringPhysical('mechanism').fit(*train[:3]);save_predictor(mechanism,folder/'mechanism.npz')
    start=perf_counter();residual=SteeringResidual(affine).fit(train[:3],val[:3],seed=seed,epochs=200)
    seconds=perf_counter()-start;save_predictor(residual,folder/'residual.npz')
    arrays={}
    for split,data in [('train',train),('val',val)]:
        for key,value in zip(('state','action','next','episode'),data):arrays[split+'_'+key]=value
    np.savez_compressed(folder/'data.npz',**arrays)
    scores={name:float(np.mean(np.abs(m.predict_batch(val[0],val[1])[:,5]-val[2][:,5])))
            for name,m in [('rich',rich),('affine',affine),('residual',residual),('mechanism',mechanism)]}
    write(folder/'manifest.json',dict(training_episodes=24,validation_episodes=6,training_transitions=2400,
        validation_transitions=600,validation_steering_mae_rad=scores,residual_fit_seconds=seconds,
        residual_training_log=residual.log,model_sha256={p.name:sha(p) for p in folder.glob('*.npz')}))
    print('VALIDATION',seed,json.dumps(scores),flush=True)


def predictions(root,seeds):
    rows=[]
    for seed in seeds:
        folder=root/f'seed{seed}'/'models'
        predictors={'rich':PhysicalModel.load(folder/'rich.npz'),**{k:load_predictor(folder/f'{k}.npz') for k in ('affine','residual','mechanism')}}
        for split in ('test','ood'):
            s,u,n,ids=collect(seed,split,10)
            np.savez_compressed(folder/f'{split}_prediction_data.npz',state=s,action=u,next=n,episode=ids)
            # Non-overlapping windows; episode means retained as sampling units.
            for name,model in predictors.items():
                for eid in np.unique(ids):
                    ss,uu,nn=s[ids==eid],u[ids==eid],n[ids==eid]
                    for h in (1,5,10,30):
                        errors=[]
                        for start in range(0,100-h+1,h):
                            pred=ss[start:start+1].copy()
                            for k in range(h):pred=model.predict_batch(pred,uu[start+k:start+k+1])
                            truth=nn[start+h-1]
                            errors.append([abs(pred[0,5]-truth[5]),np.linalg.norm(pred[0,:2]-truth[:2])])
                        means=np.mean(errors,axis=0)
                        rows.append(dict(training_seed=seed,split=split,model=name,episode=int(eid),horizon_steps=h,
                            steering_mae_rad=float(means[0]),position_error_m=float(means[1])))
    write(root/'prediction_episodes.json',rows)
    summary=[]
    for split in ('test','ood'):
        for h in (1,5,10,30):
            for name in ('rich','affine','residual','mechanism'):
                rr=[r for r in rows if r['split']==split and r['horizon_steps']==h and r['model']==name]
                summary.append(dict(split=split,horizon_steps=h,model=name,episodes=len(rr),
                    steering_mae_rad=float(np.mean([r['steering_mae_rad'] for r in rr])),
                    position_error_m=float(np.mean([r['position_error_m'] for r in rr]))))
    write(root/'prediction_summary.json',summary)


def evaluate(policy,folder,seed,namespace=350_000_000):
    folder.mkdir();env=MismatchTrackingEnv();records=[];traces={}
    for ci,condition in enumerate((1.,1.3)):
        for fi,family in enumerate(('straight','curve')):
            for pair in range(10):
                eid=namespace+seed*100_000+ci*10_000+fi*1000+pair
                obs,_=env.reset(options=dict(episode_seed=eid,family=family,condition=condition))
                total=0.;ey=[];ev=[];rate=[];trace=[]
                for _ in range(MAX_STEPS):
                    action,_=policy.predict(obs,deterministic=True)
                    obs,_,terminated,truncated,info=env.step(action);total+=info['raw_reward']
                    ey.append(info['lateral_error_m']);ev.append(info['speed_error_mps'])
                    rate.append((info['applied'][1]-info['previous'][1])/.1)
                    trace.append(np.r_[info['state'],info['next_state'],info['previous'],action,info['applied'],
                        info['raw_reward'],info['failure'],info['success'],terminated,truncated])
                    if terminated or truncated:break
                traces[str(eid)]=np.array(trace)
                records.append(dict(training_seed=seed,episode_seed=eid,family=family,condition=condition,
                    distribution='id' if ci==0 else 'ood',steps=len(trace),completed=info['success'],
                    road_departure=info['failure'],timeout=bool(truncated),return_raw=total,
                    lateral_rmse_m=float(np.sqrt(np.mean(np.square(ey)))),
                    speed_rmse_mps=float(np.sqrt(np.mean(np.square(ev)))),
                    steering_rate_rms_radps=float(np.sqrt(np.mean(np.square(rate))))))
    np.savez_compressed(folder/'trajectories.npz',**traces)
    with (folder/'episodes.jsonl').open('w',encoding='utf8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    return records


def job(arg):
    root=Path(arg['root']);seed=arg['seed'];name=arg['method'];torch.set_num_threads(1)
    folder=root/f'seed{seed}'/name;folder.mkdir();mp=root/f'seed{seed}'/'models'
    rich=PhysicalModel.load(mp/'rich.npz')
    model=ModelGradientPPO('MlpPolicy',MismatchTrackingEnv(training_seed=seed,namespace=360_000_000),
        world_model=rich if name!='ppo' else None,aux_weight=1 if name!='ppo' else 0,
        aux_horizon=16,aux_interval=4,seed=seed,device='cpu',**CONFIG)
    if name not in ('ppo','rich'):model.world_dynamics=TorchSteering(load_predictor(mp/f'{name}.npz'))
    start=perf_counter();model.warm_start(warm_roots(seed),256);warm_seconds=perf_counter()-start
    start=perf_counter();model.learn(16384);train_seconds=perf_counter()-start
    assert model.num_timesteps==model.env.get_attr('transition_count')[0]==16384
    path=folder/'final_policy.zip';model.save(path);frozen=PPO.load(path,device='cpu')
    rows=evaluate(frozen,folder/'test',seed)
    ledger=dict(method=name,training_seed=seed,online_steps=16384,warm_steps=model.warm_steps,
        policy_optimizer_steps=model.optimizer_steps,aux_updates=model.aux_updates,
        imagined_steps=model.aux_model_steps,offline_interactions=3000 if name!='ppo' else 0,
        warm_seconds=warm_seconds,training_seconds=train_seconds,checkpoint_sha256=sha(path),
        test_sha256=sha(folder/'test/episodes.jsonl'),trajectory_sha256=sha(folder/'test/trajectories.npz'),
        completion_rate=float(np.mean([r['completed'] for r in rows])),mean_return=float(np.mean([r['return_raw'] for r in rows])))
    write(folder/'complete.json',ledger);print('CLOSED',json.dumps(ledger),flush=True);return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','evaluate'),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/ackermann_rl/friction_mismatch'));p.add_argument('--workers',type=int,default=5)
    args=p.parse_args();root=args.output;seeds=[0,1,2]
    if args.stage=='prepare':
        root.mkdir(parents=True,exist_ok=False)
        sources=['run_ackermann_mismatch.py','falconworld/ackermann_mismatch.py','falconworld/ackermann_rl.py',
            'falconworld/ackermann_differentiable.py','falconworld/ackermann_model_gradient.py',
            'falconworld/ackermann.py','run_ackermann_improvement.py','run_ackermann_rl.py']
        write(root/'registration.json',dict(seeds=seeds,methods=METHODS,online_steps=16384,
            primary='ID prediction steering MAE and ID closed-loop raw return with completion not lower; OOD secondary',
            control='rich7 refit; affine gain/offset refit; residual on same affine; exact friction-structure physical refit',
            plant_version='v2-monotone-smooth-deadzone',
            interpretation='Synthetic monotone dead-zone mismatch, not measured R550. V1 nonmonotone prototype archived/excluded. Residual cannot be assumed superior to matched-structure physical control.',
            frozen_hyperparameters=dict(aux_weight=1,horizon=16,warm_steps=256,interval=4,mlp_epochs=200),
            sources={s:sha(s) for s in sources},namespaces=NAMESPACES,versions=dict(torch=torch.__version__,numpy=np.__version__)))
        for source in sources:
            target=root/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for seed in seeds:prepare(root,seed)
    else:
        registration=json.loads((root/'registration.json').read_text())
        for source,digest in registration['sources'].items():
            if sha(source)!=digest:raise ValueError('Registered source changed before test: '+source)
        predictions(root,seeds)
        tasks=[dict(root=str(root),seed=s,method=m) for s in seeds for m in METHODS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:ledgers=list(pool.map(job,tasks))
        write(root/'cost_ledger.json',ledgers)
        records=[]
        for seed in seeds:
            for method in METHODS:
                records += [dict(json.loads(line),method=method) for line in (root/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines()]
        write(root/'episodes.json',records);summary=[]
        for method in METHODS:
            for dist in ('id','ood'):
                rr=[r for r in records if r['method']==method and r['distribution']==dist]
                summary.append(dict(method=method,distribution=dist,episodes=len(rr),completed=sum(r['completed'] for r in rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in ('return_raw','lateral_rmse_m','speed_rmse_mps','steering_rate_rms_radps')}))
        write(root/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
