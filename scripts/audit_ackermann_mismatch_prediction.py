"""Independent NumPy forward audit of frozen physics and neural predictions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def step(s,u,model,name):
    p=model['params'] if name=='rich' else model['base_params']
    v=s[:,3];a=s[:,4];d=s[:,5]
    ta=np.where(u[:,0]>=a,p[0]+p[1]*v,p[2]+p[3]*v)
    td=p[4]+p[5]*v
    n=np.column_stack([s[:,0]+.1*v*np.cos(s[:,2]),s[:,1]+.1*v*np.sin(s[:,2]),
        (s[:,2]+.1*v*np.tan(d)/(1+p[6]*v*v)+np.pi)%(2*np.pi)-np.pi,
        np.maximum(0,v+.1*a),np.clip(a+.1*(u[:,0]-a)/ta,-3,2),np.clip(d+.1*(u[:,1]-d)/td,-.5,.5)])
    if name!='rich':
        p=model['params'];force=p[2]*u[:,1]+p[3]-d
        if str(model['kind'])=='mechanism':
            width=p[4]*(1+p[5]*(v/5)**2)
            force-=width*np.tanh(force/(p[6]*width))
        n[:,5]=np.clip(d+.1/(p[0]+p[1]*v)*force,-.5,.5)
        if bool(model['residual']):
            features=np.column_stack([v/6,(u[:,1]-d)/.1,u[:,1]/.5])
            x=((features-model['xmean'])/model['xscale']).astype(np.float32)
            for layer in (0,2):x=np.tanh(x@model[f'net_{layer}.weight'].T+model[f'net_{layer}.bias'])
            r=(x@model['net_4.weight'].T+model['net_4.bias'])[:,0]*float(model['yscale'])
            n[:,5]=np.clip(n[:,5]+r,-.5,.5)
    return n


def audit(root):
    registration=json.loads((root/'registration.json').read_text())
    if registration.get('plant_version')!='v2-monotone-smooth-deadzone':raise ValueError('Excluded plant')
    logged=json.loads((root/'prediction_episodes.json').read_text())
    lookup={(r['training_seed'],r['split'],r['model'],r['episode'],r['horizon_steps']):r for r in logged}
    errors=[];max_error=0.;count=0
    for seed in registration['seeds']:
        folder=root/f'seed{seed}'/'models'
        with np.load(folder/'data.npz') as data:
            used=set(data['train_episode'])|set(data['val_episode'])
        for split in ('test','ood'):
            with np.load(folder/f'{split}_prediction_data.npz') as data:
                s,u,n,ids=[data[k].copy() for k in ('state','action','next','episode')]
            if used&set(ids):errors.append('split_leakage')
            for name in ('rich','affine','residual','mechanism'):
                with np.load(folder/f'{name}.npz',allow_pickle=False) as f:model={k:f[k].copy() for k in f.files}
                for eid in np.unique(ids):
                    ss,uu,nn=s[ids==eid],u[ids==eid],n[ids==eid]
                    for h in (1,5,10,30):
                        starts=np.array(list(range(0,100-h+1,h)))
                        pred=ss[starts].copy()
                        for k in range(h):pred=step(pred,uu[starts+k],model,name)
                        truth=nn[starts+h-1]
                        delta=np.array([np.mean(abs(pred[:,5]-truth[:,5])),np.mean(np.linalg.norm(pred[:,:2]-truth[:,:2],axis=1))])
                        row=lookup[(seed,split,name,int(eid),h)];count+=1
                        difference=max(abs(delta[0]-row['steering_mae_rad']),abs(delta[1]-row['position_error_m']))
                        max_error=max(max_error,float(difference))
                        if difference>2e-6:errors.append(dict(kind='prediction_reconstruction',seed=seed,model=name,horizon=h,difference=float(difference)))
    if count!=len(logged) or len(lookup)!=len(logged):errors.append('row_identity_count')
    report=dict(status='verified_complete' if not errors else 'failed',episode_model_horizon_rows=count,
        error_count=len(errors),errors=errors,max_metric_difference=max_error,
        scope='Reimplemented physics and neural forward in NumPy; no model/simulator modules imported; complete heldout episode means rebuilt.')
    (root/'prediction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    report=audit(args.run);print(json.dumps(report));raise SystemExit(report['error_count']!=0)
