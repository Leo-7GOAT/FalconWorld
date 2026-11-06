"""Independent trajectory/dynamics/reward/Adam audit (no simulator imported)."""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    registration=json.loads((root/'registration.json').read_text(encoding='utf8'))
    mismatch='frozen_hyperparameters' in registration
    gradient=registration.get('stage') in ('final','efficiency') or mismatch
    spec=registration.get('spec',dict(seeds=registration.get('seeds'),steps=registration.get('online_steps'),pairs=10))
    errors=[];episodes=0;steps=0;identities=set();max_error=0.
    def require(condition, label):
        if not condition:errors.append(label)
    if gradient:
        require("REWARD_VERSION = 'tracking-v2-bounded-survival'" in
            (root/'source_snapshot/falconworld/ackermann_rl.py').read_text(encoding='utf8'),'wrong_reward_version')
    else:
        require(registration['reward_version']=='tracking-v2-bounded-survival','wrong_reward_version')
    for name,digest in registration['sources'].items():
        require(sha(root/'source_snapshot'/name)==digest, 'source_snapshot:'+name)
    for seed in spec['seeds']:
        mf=root/f'seed{seed}/models';manifest=json.loads((mf/'manifest.json').read_text())
        require(sha(mf/'data.npz')==manifest.get('dataset_sha256',manifest['model_sha256'].get('data.npz')),'offline_data_hash')
        for name,digest in manifest['model_sha256'].items():require(sha(mf/name)==digest,'model_hash')
        if mismatch:
            with np.load(mf/'data.npz',allow_pickle=False) as data:
                train=set(map(int,np.unique(data['train_episode'])));val=set(map(int,np.unique(data['val_episode'])))
        else:
            train={e['seed'] for e in manifest['episodes'] if e['split']=='train'}
            val={e['seed'] for e in manifest['episodes'] if e['split']=='validation'}
        require(not train&val and len(train)==24 and len(val)==6,'offline_episode_split')
        for method in registration['methods']:
            folder=root/f'seed{seed}'/method;ledger=json.loads((folder/'complete.json').read_text())
            require(sha(folder/'final_policy.zip')==ledger['checkpoint_sha256'],'policy_hash')
            require(sha(folder/'test/episodes.jsonl')==ledger['evaluation_sha256' if gradient and not mismatch else 'test_sha256'],'test_hash')
            require(sha(folder/'test/trajectories.npz')==ledger['trajectory_sha256'],'trajectory_hash')
            expected_online=20480 if method=='ppo_total_budget' else spec['steps']
            expected_steps=expected_online//2048*320
            require(ledger['online_steps']==expected_online,'online_budget')
            if gradient:
                cfg=registration['frozen_hyperparameters'] if mismatch else registration['selection']['config']
                warm=0 if method in ('ppo','ppo_compute','ppo_total_budget') else cfg['warm_steps']
                extra=cfg['warm_steps'] if method=='ppo_compute' else 0
                require(ledger['warm_steps']==warm and ledger['policy_optimizer_steps']==expected_steps+extra,'gradient_budget')
                if not mismatch:require(ledger['total_optimizer_steps']==expected_steps+extra+warm,'total_gradient_budget')
                expected_aux=0 if method in ('ppo','ppo_compute','ppo_total_budget') else expected_steps//cfg['interval']
                require(ledger['aux_updates']==expected_aux,'aux_update_count')
                uses_dynamics=method in ('wm_gradient','physics_gradient') or (mismatch and method!='ppo')
                expected_imagined=(warm*64+expected_aux*16)*cfg['horizon'] if uses_dynamics else 0
                require(ledger['imagined_steps']==expected_imagined,'imagined_step_count')
                expected_adam={expected_steps+extra} if not warm else {n for n in (expected_steps,expected_steps+warm) if n>0}
            else:
                require(ledger['online_steps']==spec['steps'] and ledger['optimizer_steps']==expected_steps,'ledger_budget')
                require(ledger['synthetic_replaced']==(0 if method=='ppo' else spec['steps']//2),'model_replacement_count')
                expected_adam={expected_steps}
            with zipfile.ZipFile(folder/'final_policy.zip') as archive:
                data=json.loads(archive.read('data'))
                require(data['num_timesteps']==expected_online,'zip_online_steps')
                optim=torch.load(BytesIO(archive.read('policy.optimizer.pth')),weights_only=False,map_location='cpu')
                require({int(v['step'].item()) for v in optim['state'].values() if 'step' in v}==expected_adam,'actual_adam_count')
                require(not any(k in data for k in ('world_model','branch_rng','branch_torch_state','world_dynamics')),'external_model_saved')
            records=[json.loads(line) for line in (folder/'test/episodes.jsonl').read_text().splitlines()]
            require(len(records)==4*spec['pairs'],'episode_count')
            with np.load(folder/'test/trajectories.npz',allow_pickle=False) as traces:
                for record in records:
                    eid=record['episode_seed'];key=(seed,method,eid)
                    require(key not in identities,'duplicate_test_case');identities.add(key)
                    require(eid not in train|val,'test_leakage')
                    trace=traces[str(eid)];episodes+=1;steps+=len(trace)
                    require(len(trace)==record['steps'] and trace.shape[1]==23,'trace_shape')
                    s,n,prev,raw,u=trace[:,:6],trace[:,6:12],trace[:,12:14],trace[:,14:16],trace[:,16:18]
                    require(np.isfinite(trace).all(),'nonfinite_trace')
                    require(np.allclose(s[1:],n[:-1],atol=1e-12) and np.allclose(prev[1:],u[:-1],atol=1e-12),'trace_continuity')
                    desired=np.column_stack([2.5*np.clip(raw[:,0],-1,1)-.5,.5*np.clip(raw[:,1],-1,1)])
                    expected_u=np.clip(desired,prev-[.5,.08],prev+[.5,.08])
                    require(np.allclose(u,expected_u,atol=1e-12),'command_limits_or_slew')
                    condition=record['condition'];v=s[:,3]
                    ta=np.where(u[:,0]>=s[:,4],.35+.04*v,.25+.02*v)*condition
                    td=(.22+.06*v)*condition
                    expected=np.column_stack([s[:,0]+.1*v*np.cos(s[:,2]),s[:,1]+.1*v*np.sin(s[:,2]),
                        (s[:,2]+.1*v*np.tan(s[:,5])/(1+.015*condition*v*v)+np.pi)%(2*np.pi)-np.pi,
                        np.maximum(0,v+.1*s[:,4]),np.clip(s[:,4]+.1*(u[:,0]-s[:,4])/ta,-3,2),
                        np.clip(s[:,5]+.1*(u[:,1]-s[:,5])/td,-.5,.5)])
                    if mismatch:
                        if registration.get('plant_version')!='v2-monotone-smooth-deadzone':
                            raise ValueError('Excluded nonmonotone prototype is not an accepted mismatch experiment')
                        effort=1.08*u[:,1]+.025-s[:,5]
                        width=.032*(1+.30*(v/5)**2)
                        force=effort-width*np.tanh(effort/width)
                        expected[:,5]=np.clip(s[:,5]+.1/td*force,-.5,.5)
                    require(np.allclose(n,expected,atol=1e-10),'independent_dynamics')
                    rng=np.random.default_rng(eid);phase=rng.uniform(-.8,.8)
                    vref=rng.uniform(4,5) if condition!=1 else rng.uniform(2,4)
                    amp=(1.8 if condition!=1 else 1.3) if record['family']=='curve' else 0.
                    road=lambda x:amp*(np.sin(x/9+phase)-np.sin(phase))
                    ey=n[:,1]-road(n[:,0]);ep=(n[:,2]-np.arctan(amp/9*np.cos(n[:,0]/9+phase))+np.pi)%(2*np.pi)-np.pi
                    ev=n[:,3]-vref
                    # Independent swept footprint using rectangular corner offsets.
                    fail=[]
                    offsets=np.array([[-.75,-.35],[-.75,.35],[.75,-.35],[.75,.35]])
                    for old,new in zip(s,n):
                        yaw_delta=(new[2]-old[2]+np.pi)%(2*np.pi)-np.pi
                        parts=max(2,int(np.ceil((np.linalg.norm(new[:2]-old[:2])+abs(yaw_delta)*1.5)/.07)))
                        t=np.linspace(0,1,parts+1)
                        poses=old+t[:,None]*(new-old);poses[:,2]=old[2]+t*yaw_delta
                        co,si=np.cos(poses[:,2]),np.sin(poses[:,2])
                        xx=poses[:,0,None]+.4*co[:,None]+co[:,None]*offsets[:,0]-si[:,None]*offsets[:,1]
                        yy=poses[:,1,None]+.4*si[:,None]+si[:,None]*offsets[:,0]+co[:,None]*offsets[:,1]
                        fail.append(bool(np.any(np.abs(yy-road(xx))>=3.5) or new[0]<-2))
                    fail=np.array(fail);success=(n[:,0]>=30)&~fail
                    cost=.5*ey**2+.15*ep**2+.2*ev**2+.005*np.sum(u*u,axis=1)+.01*np.sum((u-prev)**2,axis=1)
                    reward=np.where(fail,-20.,1-np.minimum(2,cost)+.1*np.clip(n[:,0]-s[:,0],-1,1)+50*success)
                    delta=float(np.max(abs(reward-trace[:,18])));max_error=max(max_error,delta)
                    require(delta<1e-9,'raw_reward')
                    require(np.array_equal(fail,trace[:,19].astype(bool)) and np.array_equal(success,trace[:,20].astype(bool)),'terminal_geometry')
                    require(not np.any(trace[:-1,21:23]) and bool(trace[-1,21] or trace[-1,22]),'terminal_boundaries')
                    require(bool(trace[-1,22])==(len(trace)==160 and not (fail[-1] or success[-1])),'timeout_bootstrap_boundary')
                    require(abs(sum(reward)-record['return_raw'])<1e-8,'episode_return')
                    require(abs(np.sqrt(np.mean(ey*ey))-record['lateral_rmse_m'])<1e-10,'lateral_rmse')
                    require(bool(success[-1])==record['completed'] and bool(fail[-1])==record['road_departure'],'completion_label')
    expected=len(spec['seeds'])*len(registration['methods'])*4*spec['pairs']
    require(episodes==expected,'global_case_count')
    namespace=registration.get('evaluation_namespace',350_000_000) if mismatch else 99_000_000 if registration.get('stage')=='efficiency' else 95_000_000 if gradient else 90_000_000
    expected_keys={(seed,method,namespace+seed*100_000+ci*10_000+fi*1000+pair)
        for seed in spec['seeds'] for method in registration['methods'] for ci in (0,1) for fi in (0,1) for pair in range(spec['pairs'])}
    require(identities==expected_keys,'heldout_case_identities')
    return dict(status='verified_complete' if not errors else 'failed',episodes=episodes,steps=steps,
        expected_episodes=expected,error_count=len(errors),errors=errors,max_raw_reward_error=max_error,
        scope='Independent truth dynamics, swept road footprint, rewards, termination, hashes, splits, zip/Adam budgets. No training sample trace; model branch individual samples not independently recoverable.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    result=audit(args.run);(args.run/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result));raise SystemExit(0 if result['status']=='verified_complete' else 1)


if __name__=='__main__':main()
