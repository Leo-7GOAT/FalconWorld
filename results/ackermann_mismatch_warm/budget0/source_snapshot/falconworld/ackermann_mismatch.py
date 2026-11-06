"""Synthetic unmodeled steering friction: fit physical families and residual.

No real-car identification claim. All predictors are fit with identical data;
the mechanism-aware physical family is an explicit matched-structure control.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
import torch
from torch import nn
from .ackermann import PhysicalModel,step_ackermann
from .ackermann_rl import AckermannTrackingEnv,CFG
from .ackermann_differentiable import DifferentiableAckermann


def mismatch_truth(state, action, condition=1.):
    s,u=np.asarray(state),np.asarray(action)
    n=step_ackermann(s,u,CFG,condition=condition)
    v,d=s[...,3],s[...,5]
    effort=1.08*u[...,1]+.025-d
    width=.032*(1+.30*(v/5)**2)
    # Monotone smooth dead zone: F(e)=e-w*tanh(e/w), F'(e)>=0.
    # Resistance never reverses the effort and vanishes at equilibrium.
    force=effort-width*np.tanh(effort/width)
    n[...,5]=np.clip(d+.1/((.22+.06*v)*condition)*force,-.5,.5)
    return n


class MismatchTrackingEnv(AckermannTrackingEnv):
    def advance(self,state,action):
        return mismatch_truth(state,action,self.condition)


class SteeringPhysical:
    def __init__(self,kind='affine',base=None):
        self.kind=kind;self.base=base or PhysicalModel(CFG,True)
        self.params=np.array([.3,.04,1.,0.,.02,.1,1.3])

    def predict_batch(self,s,u):
        s,u=np.asarray(s),np.asarray(u);n=self.base.predict_batch(s,u)
        p=self.params;v,d=s[...,3],s[...,5]
        force=p[2]*u[...,1]+p[3]-d
        if self.kind=='mechanism':
            width=p[4]*(1+p[5]*(v/5)**2)
            force=force-width*np.tanh(force/(p[6]*width))
        n[...,5]=np.clip(d+.1/(p[0]+p[1]*v)*force,-.5,.5)
        return n

    def fit(self,s,u,n):
        self.base.fit(s,u,n)
        count=7 if self.kind=='mechanism' else 4
        initial=np.array([.3,.04,1.,0.,.02,.1,1.3])
        low=np.array([.12,0,.7,-.1,.001,0,1.]);high=np.array([1,.2,1.4,.1,.08,1,3.])
        def loss(p):
            self.params[:count]=p
            return self.predict_batch(s,u)[:,5]-n[:,5]
        result=least_squares(loss,initial[:count],bounds=(low[:count],high[:count]),max_nfev=150)
        self.params[:count]=result.x
        return self


class SteeringResidual:
    def __init__(self,base):
        self.base=base;self.net=nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.xmean=np.zeros(3);self.xscale=np.ones(3);self.yscale=.01

    @staticmethod
    def features(s,u):
        return np.column_stack([s[:,3]/6,(u[:,1]-s[:,5])/.1,u[:,1]/.5])

    def fit(self,train,val,seed=0,epochs=200):
        torch.manual_seed(seed)
        self.net=nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        s,u,n=train;sv,uv,nv=val
        x=self.features(s,u);xv=self.features(sv,uv)
        self.xmean=x.mean(0);self.xscale=np.maximum(x.std(0),.05)
        y=n[:,5:6]-self.base.predict_batch(s,u)[:,5:6]
        yv=nv[:,5:6]-self.base.predict_batch(sv,uv)[:,5:6]
        self.yscale=max(float(y.std()),.001)
        x=torch.tensor((x-self.xmean)/self.xscale,dtype=torch.float32)
        xv=torch.tensor((xv-self.xmean)/self.xscale,dtype=torch.float32)
        y=torch.tensor(y/self.yscale,dtype=torch.float32);yv=torch.tensor(yv/self.yscale,dtype=torch.float32)
        optim=torch.optim.Adam(self.net.parameters(),lr=.003,weight_decay=1e-5)
        best=float('inf');weights=None;stale=0;self.log=[]
        generator=torch.Generator().manual_seed(seed+91)
        for epoch in range(epochs):
            for ids in torch.randperm(len(x),generator=generator).split(256):
                loss=(self.net(x[ids])-y[ids]).square().mean()
                optim.zero_grad();loss.backward();optim.step()
            if epoch%5==0 or epoch==epochs-1:
                with torch.no_grad():score=float((self.net(xv)-yv).square().mean())
                self.log.append(dict(epoch=epoch,validation_mse=score))
                if score<best:best=score;weights={k:v.clone() for k,v in self.net.state_dict().items()};stale=0
                else:stale+=1
                if stale>=10:break
        self.net.load_state_dict(weights)
        self.net.eval()
        for p in self.net.parameters():p.requires_grad_(False)
        return self

    def predict_batch(self,s,u):
        n=self.base.predict_batch(s,u)
        x=(self.features(s,u)-self.xmean)/self.xscale
        with torch.no_grad():r=self.net(torch.tensor(x,dtype=torch.float32)).numpy()[:,0]*self.yscale
        n[:,5]=np.clip(n[:,5]+r,-.5,.5)
        return n


class TorchSteering(DifferentiableAckermann):
    def __init__(self,predictor):
        physical=predictor.base if isinstance(predictor,SteeringResidual) else predictor
        super().__init__(physical.base)
        self.kind=physical.kind
        self.register_buffer('steering_params',torch.tensor(physical.params,dtype=torch.float32))
        self.residual=isinstance(predictor,SteeringResidual)
        if self.residual:
            self.net=predictor.net
            for p in self.net.parameters():p.requires_grad_(False)
            self.register_buffer('xmean',torch.tensor(predictor.xmean,dtype=torch.float32))
            self.register_buffer('xscale',torch.tensor(predictor.xscale,dtype=torch.float32));self.yscale=predictor.yscale

    def step(self,s,u):
        n=super().step(s,u);v,d=s[:,3],s[:,5];p=self.steering_params
        force=p[2]*u[:,1]+p[3]-d
        if self.kind=='mechanism':
            width=p[4]*(1+p[5]*(v/5)**2)
            force=force-width*torch.tanh(force/(p[6]*width))
        delta=(d+.1/(p[0]+p[1]*v)*force).clamp(-.5,.5)
        if self.residual:
            x=torch.stack([v/6,(u[:,1]-d)/.1,u[:,1]/.5],dim=-1)
            delta=(delta+self.net((x-self.xmean)/self.xscale).flatten()*self.yscale).clamp(-.5,.5)
        return torch.cat([n[:,:5],delta[:,None]],dim=-1)


def save_predictor(model,path):
    physical=model.base if isinstance(model,SteeringResidual) else model
    data=dict(kind=physical.kind,params=physical.params,base_params=physical.base.params,
              residual=isinstance(model,SteeringResidual))
    if isinstance(model,SteeringResidual):
        data.update(xmean=model.xmean,xscale=model.xscale,yscale=np.array(model.yscale))
        for k,v in model.net.state_dict().items():data['net_'+k]=v.numpy()
    np.savez_compressed(path,**data)


def load_predictor(path):
    with np.load(path,allow_pickle=False) as f:
        physical=SteeringPhysical(str(f['kind']));physical.params=f['params'].copy();physical.base.params=f['base_params'].copy()
        if not bool(f['residual']):return physical
        model=SteeringResidual(physical);model.xmean=f['xmean'].copy();model.xscale=f['xscale'].copy();model.yscale=float(f['yscale'])
        model.net.load_state_dict({k:torch.tensor(f['net_'+k]) for k in model.net.state_dict()})
        for p in model.net.parameters():p.requires_grad_(False)
        model.net.eval();return model
