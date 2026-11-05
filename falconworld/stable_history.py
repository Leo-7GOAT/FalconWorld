"""Validation-only candidate: stable, zero-intercept correction to leader CA.

The exogenous leader head reads only leader speed history. Ego dynamics remain
the train-identified enriched physical model. This does not replace frozen runs.
"""
from pathlib import Path
import json
import numpy as np
from .models import PhysicsModel, _batch, _history


def project_stable(coefficients, radius=.999):
    if not 0 < radius < 1:
        raise ValueError('Root radius must lie inside the unit circle')
    roots=np.roots([1.,-coefficients[0],-coefficients[1]])
    roots=roots*np.minimum(1.,radius/np.maximum(abs(roots),1e-15))
    polynomial=np.real_if_close(np.poly(roots))
    if np.iscomplexobj(polynomial):raise ValueError('Projection lost real coefficients')
    return -polynomial[1:]


class StableHistoryModel:
    def __init__(self,physics,coefficients,name='stable_history'):
        self.physics,self.coefficients,self.name=physics,np.asarray(coefficients,dtype=float),name
        if self.coefficients.shape!=(2,) or not np.isfinite(self.coefficients).all():
            raise ValueError('Expected two finite AR coefficients')

    def step(self,states,actions,history=None):
        s,u,single=_batch(states,actions)
        h=_history(s,history)
        differences=np.diff(h[:,-3:,2],axis=1)
        # CA + learned correction: last + (c0-1)*last + c1*previous.
        delta=differences[:,::-1]@self.coefficients
        out=self.physics.step(s,u,h)
        out[:,2]=np.maximum(0.,s[:,2]+np.clip(delta,-.6,.2))
        return out[0] if single else out

    def save(self,path):
        np.savez_compressed(path,coefficients=self.coefficients,
            specification=np.array(json.dumps(dict(name=self.name,physics=self.physics.specification()))))

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as arrays:
            spec=json.loads(str(arrays['specification']))
            p=spec['physics']
            return cls(PhysicsModel(p['name'],p['tau'],p['enriched'],p['lead']),arrays['coefficients'],spec['name'])


def fit_history(train,physics,ridge=1e-3,radius=.999):
    h=np.asarray(train['history'])
    x=np.diff(h[:,-3:,2],axis=1)[:,::-1]
    y=train['next_state'][:,2]-train['state'][:,2]
    scale=max(float(np.trace(x.T@x))/2,1e-12)
    coefficients=np.linalg.solve(x.T@x+ridge*scale*np.eye(2),x.T@y)
    return StableHistoryModel(physics,project_stable(coefficients,radius))
