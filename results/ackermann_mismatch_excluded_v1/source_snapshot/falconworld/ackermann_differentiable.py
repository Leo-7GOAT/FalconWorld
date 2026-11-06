"""Frozen differentiable copy of fitted Ackermann predictors, training only.

All fitted parameters are buffers, not optimized policy parameters. It supports
both ridge residual and rich physical controls; no truth dynamics are called.
"""
import numpy as np
import torch
from torch import nn
from .ackermann_rl import SCALE


class DifferentiableAckermann(nn.Module):
    def __init__(self, model):
        super().__init__()
        base = model.base if hasattr(model, 'base') else model
        self.rich = base.rich
        self.register_buffer('params', torch.as_tensor(base.params.copy(), dtype=torch.float32))
        self.register_buffer('weights', torch.as_tensor(model.weights.copy() if hasattr(model, 'weights') else np.zeros((16,3)), dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(SCALE, dtype=torch.float32))

    def step(self, s, u):
        v,a,d = s[:,3],s[:,4],s[:,5]
        p = self.params
        if self.rich:
            ga = 1/torch.where(u[:,0]>=a,p[0]+p[1]*v,p[2]+p[3]*v)
            gd = 1/(p[4]+p[5]*v)
            slip = 1/(1+p[6]*v*v)
        else:
            ga,gd,slip = p
        da,dd = u[:,0]-a,u[:,1]-d
        z = v/6; up=(da>=0).to(s.dtype); yaw=v*torch.tan(d)
        features = torch.stack([torch.ones_like(v),da,da*z,da*z*z,da*z**3,
            da*up,da*up*z,da*up*z*z,da*up*z**3,dd,dd*z,dd*z*z,dd*z**3,
            yaw,yaw*z*z,yaw*z**4],dim=-1)
        residual = features@self.weights
        angle=s[:,2]+.1*v*torch.tan(d)*slip+residual[:,2]
        return torch.stack([s[:,0]+.1*v*torch.cos(s[:,2]),s[:,1]+.1*v*torch.sin(s[:,2]),
            torch.atan2(torch.sin(angle),torch.cos(angle)),torch.clamp(v+.1*a,min=0),
            torch.clamp(a+.1*ga*da+residual[:,0],-3,2),
            torch.clamp(d+.1*gd*dd+residual[:,1],-.5,.5)],dim=-1)

    def encode(self, s, previous, route):
        amp,phase,_=route.unbind(-1)
        reference=amp*(torch.sin(s[:,0]/9+phase)-torch.sin(phase))
        heading=torch.atan(amp/9*torch.cos(s[:,0]/9+phase))
        error=torch.atan2(torch.sin(s[:,2]-heading),torch.cos(s[:,2]-heading))
        look=amp*(torch.sin((s[:,0]+3)/9+phase)-torch.sin(phase))
        return torch.cat([torch.cat([s,previous,route],dim=-1)/self.scale,
            torch.stack([(s[:,1]-reference)/2,error/np.pi,(look-s[:,1])/3],dim=-1)],dim=-1)

    def loss(self, policy, observations, horizon=12, smooth_only=False):
        raw=observations[:,:11]*self.scale
        s,previous,route=raw[:,:6],raw[:,6:8],raw[:,8:11]
        losses=[]
        for _ in range(1 if smooth_only else horizon):
            obs=self.encode(s,previous,route)
            mean=policy.get_distribution(obs).distribution.mean
            normalized=mean.clamp(-1,1)
            requested=torch.stack([2.5*normalized[:,0]-.5,.5*normalized[:,1]],dim=-1)
            change=torch.tensor([.5,.08],device=s.device)
            u=torch.maximum(torch.minimum(requested,previous+change),previous-change)
            # Raw command penalty has a gradient even when a rate limiter clips.
            smooth=.04*((requested-previous)/torch.tensor([2.5,.5],device=s.device)).square().sum(-1)
            magnitude=.001*mean.square().sum(-1)
            if smooth_only:
                return (smooth+magnitude).mean()
            n=self.step(s,u)
            amp,phase,target=route.unbind(-1)
            ey=n[:,1]-amp*(torch.sin(n[:,0]/9+phase)-torch.sin(phase))
            heading=torch.atan(amp/9*torch.cos(n[:,0]/9+phase))
            ep=torch.atan2(torch.sin(n[:,2]-heading),torch.cos(n[:,2]-heading))
            ev=n[:,3]-target
            # Smooth, unsaturated tracking objective supplies credit through
            # actuator delay. This is an auxiliary objective, not PPO's reward.
            losses.append(.5*ey.square()+.15*ep.square()+.2*ev.square()+smooth+magnitude)
            s,previous=n,u
        return torch.stack(losses).mean()
