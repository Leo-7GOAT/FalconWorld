"""Small-scale Ackermann simulator and action-conditioned prediction models.

All geometry is SI and is a simulation assumption, not R550 calibration.
State: rear-axle x/y [m], yaw [rad], speed [m/s], actuator acceleration
[m/s²], road-wheel angle [rad]. Actuator states are privileged observations.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from time import perf_counter
import json
import hashlib
import csv
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class AckermannConfig:
    dt: float = .1
    wheelbase: float = 1.0
    length: float = 1.5
    width: float = .7
    center_offset: float = .4
    accel_min: float = -3.0
    accel_max: float = 2.0
    steer_max: float = .5
    steer_rate: float = .8
    jerk_max: float = 5.0
    tau_accel: float = .45
    tau_steer: float = .4

    def __post_init__(self):
        if not all(np.isfinite(list(asdict(self).values()))):
            raise ValueError('nonfinite vehicle configuration')
        if min(self.dt, self.wheelbase, self.length, self.width,
               self.steer_max, self.steer_rate, self.jerk_max,
               self.tau_accel, self.tau_steer) <= 0:
            raise ValueError('dimensions, dt, response and limits must be positive')
        if not self.accel_min < 0 < self.accel_max or self.steer_max >= np.pi / 2:
            raise ValueError('invalid action limits')
        if self.dt > min(self.tau_accel, self.tau_steer):
            raise ValueError('explicit actuator integration requires dt <= tau')


def wrap_angle(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def validate_state(state):
    s = np.asarray(state, dtype=float)
    if s.shape[-1:] != (6,) or not np.all(np.isfinite(s)) or np.any(s[..., 3] < 0):
        raise ValueError('expected finite [...,6] state with nonnegative speed')
    return s


def bounded_action(action, previous, cfg=AckermannConfig()):
    u = np.asarray(action, dtype=float)
    if u.shape[-1:] != (2,) or not np.all(np.isfinite(u)):
        raise ValueError('action must be finite acceleration/road-wheel angle')
    previous=np.asarray(previous,dtype=float)
    if previous.shape[-1:]!=(2,) or not np.all(np.isfinite(previous)):
        raise ValueError('previous action must be finite [...,2]')
    if np.any(previous < [cfg.accel_min,-cfg.steer_max]) or np.any(previous > [cfg.accel_max,cfg.steer_max]):
        raise ValueError('previous action outside configured limits')
    u = np.clip(u, [cfg.accel_min, -cfg.steer_max], [cfg.accel_max, cfg.steer_max])
    step = np.array([cfg.jerk_max, cfg.steer_rate]) * cfg.dt
    return np.clip(u, np.asarray(previous) - step, np.asarray(previous) + step)


def constrain_sequences(actions, previous, cfg):
    out = np.asarray(actions, dtype=float).copy()
    p = np.broadcast_to(previous, (len(out), 2)).copy()
    for h in range(out.shape[1]):
        out[:, h] = bounded_action(out[:, h], p, cfg)
        p = out[:, h]
    return out


def _advance(s, u, cfg, gain_a, gain_d, slip):
    n = np.array(s, dtype=float, copy=True)
    n[..., 0] += cfg.dt * s[..., 3] * np.cos(s[..., 2])
    n[..., 1] += cfg.dt * s[..., 3] * np.sin(s[..., 2])
    n[..., 2] = wrap_angle(s[..., 2] + cfg.dt * s[..., 3] * np.tan(s[..., 5]) / cfg.wheelbase * slip)
    n[..., 3] = np.maximum(0, s[..., 3] + cfg.dt * s[..., 4])
    n[..., 4] = np.clip(s[..., 4] + cfg.dt * gain_a * (u[..., 0] - s[..., 4]), cfg.accel_min, cfg.accel_max)
    n[..., 5] = np.clip(s[..., 5] + cfg.dt * gain_d * (u[..., 1] - s[..., 5]), -cfg.steer_max, cfg.steer_max)
    return n


def step_ackermann(state, action, cfg=AckermannConfig(), *, condition=1.0):
    """Independent reference dynamics; condition is hidden from predictors.

    Speed-dependent response and yaw attenuation are explicit simulator
    assumptions. condition=1 is ID; 1.3 is held-out slower-response OOD.
    The caller must supply already rate-limited commands and owns termination.
    """
    s = validate_state(state)
    if not np.isfinite(condition) or condition <= 0 or cfg.dt > .22 * condition:
        raise ValueError('condition requires a finite positive, integration-stable response')
    if np.any(np.abs(s[...,5]) > cfg.steer_max) or np.any(s[...,4] < cfg.accel_min) or np.any(s[...,4] > cfg.accel_max):
        raise ValueError('actuator state outside configured limits')
    u = np.asarray(action, dtype=float)
    if u.shape[-1:] != (2,) or not np.all(np.isfinite(u)):
        raise ValueError('invalid action')
    u = np.clip(u, [cfg.accel_min, -cfg.steer_max], [cfg.accel_max, cfg.steer_max])
    v = s[..., 3]
    tau_a = np.where(u[..., 0] >= s[..., 4], .35 + .04 * v, .25 + .02 * v) * condition
    tau_d = (.22 + .06 * v) * condition
    slip = 1 / (1 + .015 * condition * v * v)
    # Keep environment update separate from _advance (prediction implementation).
    n = s.copy()
    n[..., 0] = s[..., 0] + cfg.dt * v * np.cos(s[..., 2])
    n[..., 1] = s[..., 1] + cfg.dt * v * np.sin(s[..., 2])
    n[..., 2] = wrap_angle(s[..., 2] + cfg.dt * v * np.tan(s[..., 5]) / cfg.wheelbase * slip)
    n[..., 3] = np.maximum(0, v + cfg.dt * s[..., 4])
    n[..., 4] = np.clip(s[..., 4] + cfg.dt / tau_a * (u[..., 0] - s[..., 4]), cfg.accel_min, cfg.accel_max)
    n[..., 5] = np.clip(s[..., 5] + cfg.dt / tau_d * (u[..., 1] - s[..., 5]), -cfg.steer_max, cfg.steer_max)
    return n


class PhysicalModel:
    """Train-only least-squares calibration, constant or rich mechanisms."""
    def __init__(self, cfg=AckermannConfig(), rich=False):
        self.cfg = cfg
        self.rich = rich
        self.params = np.array([.35, .04, .25, .02, .22, .06, .015]) if rich else np.array([1/.45, 1/.4, 1.])
        self.name = 'rich_physics' if rich else 'calibrated_physics'
        self.calls = self.seconds = 0

    def fit(self, s, u, n):
        if not self.rich:
            for k, si, ui in [(0, 4, 0), (1, 5, 1)]:
                x = (u[:, ui] - s[:, si]) * self.cfg.dt
                self.params[k] = np.clip(np.dot(x, n[:, si] - s[:, si]) / max(np.dot(x, x), 1e-12), .1, 1/self.cfg.dt)
            x = self.cfg.dt * s[:, 3] * np.tan(s[:, 5]) / self.cfg.wheelbase
            y = wrap_angle(n[:, 2] - s[:, 2])
            self.params[2] = np.clip(np.dot(x, y) / max(np.dot(x, x), 1e-12), .3, 1.)
        else:
            def residual(p):
                v = s[:, 3]
                a = 1 / np.where(u[:, 0] >= s[:, 4], p[0]+p[1]*v, p[2]+p[3]*v)
                d = 1 / (p[4]+p[5]*v)
                pred = _advance(s, u, self.cfg, a, d, 1/(1+p[6]*v*v))
                return np.concatenate([(pred[:,4]-n[:,4]), 4*(pred[:,5]-n[:,5]), 4*wrap_angle(pred[:,2]-n[:,2])])
            self.params = least_squares(residual, np.array([.4,.03,.3,.03,.3,.04,.01]),
                bounds=([.12,0,.12,0,.12,0,0],[1,.2,1,.2,1,.2,.1]), max_nfev=100).x
        return self

    def predict_batch(self, s, u):
        start = perf_counter()
        s = np.asarray(s)
        if self.rich:
            p, v = self.params, s[...,3]
            a = 1 / np.where(u[...,0] >= s[...,4], p[0]+p[1]*v, p[2]+p[3]*v)
            d = 1 / (p[4]+p[5]*v)
            slip = 1 / (1+p[6]*v*v)
        else:
            a,d,slip = self.params
        n = _advance(s,u,self.cfg,a,d,slip)
        self.calls += len(np.atleast_2d(s))
        self.seconds += perf_counter()-start
        return n

    def save(self, path):
        np.savez(path, params=self.params, rich=self.rich, config=json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            m = cls(AckermannConfig(**json.loads(str(f['config']))), bool(f['rich']))
            m.params = f['params'].copy()
        return m


def residual_features(s, u):
    """Compact nonlinear basis conditioned on candidate commands and speed.

    Output weights are learned by ridge regression. This is a learned residual
    basis model, not a fully trained hidden-layer neural network.
    """
    v = s[...,3] / 6.
    da = u[...,0]-s[...,4]
    dd = u[...,1]-s[...,5]
    yaw = s[...,3]*np.tan(s[...,5])
    up = (da>=0).astype(float)
    return np.stack([np.ones_like(v), da, da*v, da*v*v, da*v**3,
        da*up, da*up*v, da*up*v*v, da*up*v**3,
        dd, dd*v, dd*v*v, dd*v**3, yaw, yaw*v*v, yaw*v**4], axis=-1)


class ResidualModel:
    def __init__(self, base, ridge=1e-4):
        self.base, self.ridge, self.cfg = base, ridge, base.cfg
        self.weights = np.zeros((16,3))
        self.name = 'learned_residual'
        self.calls = self.seconds = 0

    def fit(self, s,u,n):
        p = self.base.predict_batch(s,u)
        target = n[:,[4,5,2]]-p[:,[4,5,2]]
        target[:,2] = wrap_angle(target[:,2])
        x = residual_features(s,u)
        self.weights = np.linalg.solve(x.T@x+self.ridge*np.eye(x.shape[1]), x.T@target)
        return self

    def predict_batch(self,s,u):
        start=perf_counter()
        n=self.base.predict_batch(s,u)
        r=residual_features(s,u)@self.weights
        n[...,4]=np.clip(n[...,4]+r[...,0],self.cfg.accel_min,self.cfg.accel_max)
        n[...,5]=np.clip(n[...,5]+r[...,1],-self.cfg.steer_max,self.cfg.steer_max)
        n[...,2]=wrap_angle(n[...,2]+r[...,2])
        self.calls+=len(np.atleast_2d(s)); self.seconds+=perf_counter()-start
        return n

    def save(self,path):
        np.savez(path,weights=self.weights,params=self.base.params,ridge=self.ridge,
                 config=json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            base=PhysicalModel(AckermannConfig(**json.loads(str(f['config']))))
            base.params=f['params'].copy()
            m=cls(base,float(f['ridge'])); m.weights=f['weights'].copy()
        return m


@dataclass(frozen=True)
class Route:
    family: str
    phase: float = 0.
    goal: float = 30.
    road_width: float = 7.

    def road(self,x):
        return 1.3*np.sin(np.asarray(x)/9+self.phase)-1.3*np.sin(self.phase) if self.family=='curve' else np.zeros_like(np.asarray(x))

    def reference(self,x):
        x=np.asarray(x)
        return 1.8*np.exp(-((x-15)/4)**2) if self.family=='obstacle' else self.road(x)

    def heading(self,x):
        return np.arctan((self.reference(np.asarray(x)+.02)-self.reference(np.asarray(x)-.02))/.04)

    def speed(self,t):
        return 2.8+.8*np.sin(.6*np.asarray(t)+self.phase) if self.family=='variable_speed' else np.zeros_like(np.asarray(t))+3.

    @property
    def obstacles(self):
        return ((15.,0.,.65),) if self.family=='obstacle' else ()


def body_corners(s,cfg=AckermannConfig()):
    s=np.asarray(s); yaw=s[...,2]
    offsets=np.array([[-cfg.length/2,-cfg.width/2],[-cfg.length/2,cfg.width/2],
                      [cfg.length/2,-cfg.width/2],[cfg.length/2,cfg.width/2]])
    center=s[...,:2]+cfg.center_offset*np.stack([np.cos(yaw),np.sin(yaw)],axis=-1)
    x=center[...,0,None]+np.cos(yaw)[...,None]*offsets[:,0]-np.sin(yaw)[...,None]*offsets[:,1]
    y=center[...,1,None]+np.sin(yaw)[...,None]*offsets[:,0]+np.cos(yaw)[...,None]*offsets[:,1]
    return np.stack([x,y],axis=-1)


def clearance(s,route,cfg=AckermannConfig()):
    """Signed road margin and exact circle-to-oriented-rectangle clearance."""
    s=np.asarray(s); yaw=s[...,2]
    corners=body_corners(s,cfg)
    road=np.min(route.road_width/2-np.abs(corners[...,1]-route.road(corners[...,0])),axis=-1)
    obstacle=np.full(s.shape[:-1],np.inf)
    cx=s[...,0]+cfg.center_offset*np.cos(yaw); cy=s[...,1]+cfg.center_offset*np.sin(yaw)
    for ox,oy,radius in route.obstacles:
        dx,dy=ox-cx,oy-cy
        lx=dx*np.cos(yaw)+dy*np.sin(yaw); ly=-dx*np.sin(yaw)+dy*np.cos(yaw)
        qx=np.abs(lx)-cfg.length/2; qy=np.abs(ly)-cfg.width/2
        distance=np.hypot(np.maximum(qx,0),np.maximum(qy,0))+np.minimum(np.maximum(qx,qy),0)-radius
        obstacle=np.minimum(obstacle,distance)
    return road,obstacle


def swept_collision(old,new,route,cfg):
    """Check the swept pose with interpolation finer than 0.1 vehicle width."""
    distance=float(np.linalg.norm(np.asarray(new)[:2]-np.asarray(old)[:2]))
    rotation=abs(float(wrap_angle(new[2]-old[2])))
    count=max(2,int(np.ceil((distance+rotation*cfg.length)/(.1*cfg.width))))
    alpha=np.linspace(0,1,count+1)[:,None]
    states=np.asarray(old)[None,:]+alpha*(np.asarray(new)-np.asarray(old))[None,:]
    states[:,2]=old[2]+alpha[:,0]*wrap_angle(new[2]-old[2])
    road,obs=clearance(states,route,cfg)
    return bool(np.any(obs<=0)),bool(np.any(road<=0))


class ClassicTracker:
    name='pure_pursuit_pid'
    def __init__(self,cfg=AckermannConfig()):
        self.cfg=cfg
    def action(self,s,t,route,previous):
        look=1.5+.65*s[3]
        tx=s[0]+look
        angle=wrap_angle(np.arctan2(route.reference(tx)-s[1],look)-s[2])
        steer=np.arctan2(2*self.cfg.wheelbase*np.sin(angle),look)
        accel=1.6*(float(route.speed(t))-s[3])-.25*s[4]
        return bounded_action(np.array([accel,steer]),previous,self.cfg),{'solve_failed':False}


class AckermannCEM:
    """Two-dimensional objective using the shared project CEM implementation."""
    def __init__(self,model,optimizer):
        self.model,self.optimizer,self.cfg=model,optimizer,model.cfg
        if abs(optimizer.config.dt-self.cfg.dt)>1e-12:
            raise ValueError("planner and model dt must match")
        self.name=model.name+'_mpc'
        self.previous_plan=None
        self.tick=0

    def action(self,s,t,route,previous):
        horizon=self.optimizer.config.horizon
        def objective(raw):
            actions=raw
            states=np.broadcast_to(s,(len(actions),6)).copy()
            cost=np.zeros(len(actions)); last=np.broadcast_to(previous,(len(actions),2)).copy()
            for k in range(horizon):
                states=self.model.predict_batch(states,actions[:,k])
                lateral=states[:,1]-route.reference(states[:,0])
                heading=wrap_angle(states[:,2]-route.heading(states[:,0]))
                road,obstacle=clearance(states,route,self.cfg)
                cost+=5*lateral*lateral+2*heading*heading+1.2*(states[:,3]-route.speed(t+(k+1)*self.cfg.dt))**2
                cost+=.06*actions[:,k,0]**2+.5*actions[:,k,1]**2
                cost+=.15*(actions[:,k,0]-last[:,0])**2+4*(actions[:,k,1]-last[:,1])**2
                cost+=2e4*(road<=0)+2e4*(obstacle<=0)+300*np.maximum(.5-road,0)**2+500*np.maximum(.5-obstacle,0)**2
                last=actions[:,k]
            cost+=10*(states[:,1]-route.reference(states[:,0]))**2-2*states[:,0]
            return cost
        plan,info=self.optimizer.optimize(objective,
            np.array([self.cfg.accel_min,-self.cfg.steer_max]),
            np.array([self.cfg.accel_max,self.cfg.steer_max]),previous,seed=917+self.tick,
            initial=self.previous_plan,slew=np.array([self.cfg.jerk_max,self.cfg.steer_rate])*self.cfg.dt)
        self.tick+=1
        if plan is None or not np.isfinite(info["cost"]):
            return bounded_action([self.cfg.accel_min,previous[1]],previous,self.cfg), {"solve_failed":True}
        plan=constrain_sequences(plan[None],previous,self.cfg)[0]
        self.previous_plan=np.concatenate([plan[1:],plan[-1:]],axis=0)
        return plan[0],info


def generate_dataset(out,cfg=AckermannConfig(),seed=527):
    """Entire episodes assigned before normalization or model fitting."""
    rng=np.random.default_rng(seed); records=[]; manifest=[]
    for eid in range(54):
        split='train' if eid<36 else 'validation' if eid<45 else 'test'
        s=np.array([0.,0.,rng.uniform(-.2,.2),rng.uniform(.1,6),rng.uniform(-1,1),rng.uniform(-.3,.3)])
        previous=np.zeros(2); states=[]; actions=[]; nxt=[]
        target=previous.copy()
        for k in range(100):
            if k%8==0:
                target=np.array([rng.uniform(-2.5,1.8),rng.uniform(-.45,.45)])
            u=bounded_action(target,previous,cfg)
            n=step_ackermann(s,u,cfg)
            states.append(s.copy()); actions.append(u.copy()); nxt.append(n.copy())
            s,previous=n,u
        records.append((np.array(states),np.array(actions),np.array(nxt),split))
        manifest.append({'episode':eid,'split':split,'steps':100})
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    arrays={}
    for split in ('train','validation','test'):
        rows=[r for r in records if r[3]==split]
        for j,key in enumerate(('state','action','next')):
            arrays[f'{split}_{key}']=np.concatenate([r[j] for r in rows])
    arrays['episode_states']=np.array([r[0] for r in records])
    arrays['episode_actions']=np.array([r[1] for r in records])
    arrays['episode_next']=np.array([r[2] for r in records])
    np.savez_compressed(out/'dataset.npz',**arrays)
    digest=hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()
    (out/'split_manifest.json').write_text(json.dumps({'seed':seed,'sha256':digest,'episodes':manifest},indent=2),encoding='utf8')
    return arrays,manifest


def train_models(data,out,cfg=AckermannConfig()):
    args=[data['train_'+x] for x in ('state','action','next')]
    base=PhysicalModel(cfg).fit(*args)
    rich=PhysicalModel(cfg,True).fit(*args)
    # Ridge coefficient selected using validation one-step error, never test.
    best=None; selection=[]
    for ridge in (1e-6,1e-4,.01):
        m=ResidualModel(base,ridge).fit(*args)
        p=m.predict_batch(data['validation_state'],data['validation_action'])
        err=float(np.mean((p[:,[4,5,2]]-data['validation_next'][:,[4,5,2]])**2))
        selection.append({'ridge':ridge,'validation_mse':err})
        if best is None or err<best[0]: best=(err,m)
    residual=best[1]
    out=Path(out)
    base.save(out/'physical.npz');rich.save(out/'rich_physical.npz');residual.save(out/'residual.npz')
    (out/'calibration.json').write_text(json.dumps({'constant':base.params.tolist(),'rich':rich.params.tolist(),'ridge_selection':selection,'chosen_ridge':residual.ridge},indent=2),encoding='utf8')
    for m,loaded in ((base,PhysicalModel.load(out/'physical.npz')),(rich,PhysicalModel.load(out/'rich_physical.npz')),(residual,ResidualModel.load(out/'residual.npz'))):
        np.testing.assert_allclose(m.predict_batch(data['validation_state'],data['validation_action']),loaded.predict_batch(data['validation_state'],data['validation_action']),atol=1e-12)
    return [base,rich,residual]


def open_loop(models,data,out,cfg):
    rows=[]
    # Four independent windows per held-out episode; episode-level errors retained.
    for eid in range(45,54):
        for model in models:
            for start in (0,20,40,60):
                s=data['episode_states'][eid,start].copy()
                for k in range(30):
                    s=model.predict_batch(s[None],data['episode_actions'][eid,start+k][None])[0]
                    if k+1 in (1,5,10,30):
                        truth=data['episode_next'][eid,start+k]
                        rows.append({'episode':eid,'window':start,'model':model.name,'horizon_s':(k+1)*cfg.dt,
                            'position_error_m':float(np.linalg.norm(s[:2]-truth[:2])),
                            'yaw_error_rad':float(abs(wrap_angle(s[2]-truth[2]))),
                            'speed_error_mps':float(abs(s[3]-truth[3])),
                            'steer_error_rad':float(abs(s[5]-truth[5]))})
    write_csv(Path(out)/'open_loop.csv',rows)
    return rows



def open_loop_ood(models,data,out,cfg,condition=1.3):
    """Frozen test action scripts replayed in slower-response OOD dynamics.

    Hidden condition generates labels only. Each forecast begins with an
    observed OOD current state and recursively predicts without condition.
    """
    rows=[]
    for eid in range(45,54):
        states=[];future=[];s=data['episode_states'][eid,0].copy()
        for u in data['episode_actions'][eid]:
            states.append(s.copy());s=step_ackermann(s,u,cfg,condition=condition);future.append(s.copy())
        for model in models:
            for start in (0,20,40,60):
                p=states[start].copy()
                for k in range(30):
                    p=model.predict_batch(p[None],data['episode_actions'][eid,start+k][None])[0]
                    if k+1 in (1,5,10,30):
                        truth=future[start+k]
                        rows.append({'episode':eid,'window':start,'model':model.name,'horizon_s':(k+1)*cfg.dt,
                            'position_error_m':float(np.linalg.norm(p[:2]-truth[:2])),
                            'yaw_error_rad':float(abs(wrap_angle(p[2]-truth[2]))),
                            'speed_error_mps':float(abs(p[3]-truth[3])),
                            'steer_error_rad':float(abs(p[5]-truth[5])), 'condition':condition})
    write_csv(Path(out)/'open_loop_ood.csv',rows)
    return rows


def write_csv(path,rows):
    if not rows:return
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temporary.replace(path)



class AckermannSafetySupervisor:
    """Independent rate-limited emergency override, shared across controllers.

    A geometric last resort with explicit logs, not a safety guarantee. Models
    can fail before this reserve is sufficient; swept simulation still checks.
    """
    def __init__(self,cfg=AckermannConfig()):
        self.cfg=cfg

    def filter(self,s,u,previous,route):
        cfg=self.cfg
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(u)):
            return bounded_action([cfg.accel_min,previous[1]],previous,cfg),'invalid_observation_or_action'
        for ox,oy,r in route.obstacles:
            along=(ox-s[0])*np.cos(s[2])+(oy-s[1])*np.sin(s[2])
            lateral=abs(-(ox-s[0])*np.sin(s[2])+(oy-s[1])*np.cos(s[2]))
            stopping=s[3]**2/(2*abs(cfg.accel_min))+.5*s[3]+cfg.length/2+r
            if 0<along<stopping and lateral<cfg.width/2+r+.1 and along<3.:
                return bounded_action([cfg.accel_min,u[1]],previous,cfg),'obstacle_emergency'
        return bounded_action(u,previous,cfg),''


def simulate(controller,route,initial,*,cfg=AckermannConfig(),condition=1.,max_steps=160,deadline_ms=100.):
    s=np.array(initial,dtype=float);previous=np.zeros(2);logs=[];latencies=[];failures={};collision=False;road_exit=False;completed=False
    model=getattr(controller,'model',None)
    infer_start=0 if model is None else model.seconds
    for k in range(max_steps):
        t=k*cfg.dt;start=perf_counter();reason=''
        try:
            u,info=controller.action(s,t,route,previous)
            if not np.all(np.isfinite(u)) or info.get('solve_failed',False):raise ValueError('solver result invalid')
        except (ValueError,FloatingPointError,RuntimeError) as exc:
            u=bounded_action([cfg.accel_min,previous[1]],previous,cfg);reason='solve_failure'
        u,safety_reason=AckermannSafetySupervisor(cfg).filter(s,u,previous,route)
        reason=reason or safety_reason
        elapsed=(perf_counter()-start)*1000
        if elapsed>deadline_ms:
            reason=reason or 'deadline'
            u=bounded_action([cfg.accel_min,previous[1]],previous,cfg)
        latencies.append(elapsed)
        if reason:failures[reason]=failures.get(reason,0)+1
        n=step_ackermann(s,u,cfg,condition=condition)
        collision,road_exit=swept_collision(s,n,route,cfg)
        actual_a=(n[3]-s[3])/cfg.dt
        logs.append({'step':k,'time_s':t,'x_m':float(n[0]),'y_m':float(n[1]),'yaw_rad':float(n[2]),'speed_mps':float(n[3]),
            'accel_cmd_mps2':float(u[0]),'steer_cmd_rad':float(u[1]),'actual_accel_mps2':float(actual_a),
            'tracking_error_m':float(n[1]-route.reference(n[0])),'speed_error_mps':float(n[3]-route.speed(t+cfg.dt)),
            'latency_ms':elapsed,'fallback':reason})
        s,previous=n,u
        completed=bool(s[0]>=route.goal and not collision and not road_exit)
        if collision or road_exit or completed:break
    jerk=np.diff([x['actual_accel_mps2'] for x in logs])/cfg.dt
    metrics={'controller':controller.name,'family':route.family,'condition':condition,'collision':int(collision),
        'road_exit':int(road_exit),'completed':int(completed),'progress_m':float(s[0]-initial[0]),
        'duration_s':len(logs)*cfg.dt,'tracking_rmse_m':float(np.sqrt(np.mean([x['tracking_error_m']**2 for x in logs]))),
        'speed_rmse_mps':float(np.sqrt(np.mean([x['speed_error_mps']**2 for x in logs]))),
        'jerk_rms_mps3':float(np.sqrt(np.mean(jerk*jerk))) if len(jerk) else 0.,
        'fallback_count':sum(failures.values()),'fallback_reasons':json.dumps(failures,sort_keys=True),
        'model_total_ms':0. if model is None else (model.seconds-infer_start)*1000,
        'cycle_p50_ms':float(np.percentile(latencies,50)),'cycle_p95_ms':float(np.percentile(latencies,95)),
        'timeout_rate':float(np.mean(np.array(latencies)>deadline_ms))}
    return metrics,logs
