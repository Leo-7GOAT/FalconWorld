"""Explicit bridges between sensor-facing contracts and NumPy world models."""
import numpy as np
from .contracts import Action, Observation
from .models import estimate_sensor_state


def observed_history(history, dt=.1):
    if not np.isclose(dt,.1):
        raise ValueError('This model is trained for dt=0.1s; resample observations first')
    if not history.observations:
        raise ValueError('At least one timestamped observation required')
    obs=history.observations[-6:]
    if any(o.gap is None or o.lead_speed is None for o in obs):
        raise ValueError('Following requires measured gap and leader speed; use speed-only adapter otherwise')
    if len(obs)>1 and not np.allclose(np.diff([o.timestamp for o in obs]),dt,rtol=.05,atol=1e-5):
        raise ValueError('Sampling interval differs from trained model; resampling/latency handling required')
    rows=[]
    for k,o in enumerate(obs):
        raw=np.array([o.gap,o.speed,o.lead_speed])
        causal=np.array([[p.gap,p.speed,p.lead_speed] for p in obs[:k+1]])
        rows.append(estimate_sensor_state(raw,causal if k else None))
    rows=np.asarray(rows)
    return np.concatenate((np.repeat(rows[:1],6-len(rows),axis=0),rows),axis=0)


class SensorWorldModel:
    def __init__(self, model):
        self.model=model

    def predict(self, history, actions):
        from .experiments import recursive_prediction
        states=observed_history(history)
        if any(abs(a.steering)>1e-8 for a in actions):
            raise ValueError('Longitudinal model cannot predict steering')
        trajectory=recursive_prediction(self.model,states[-1],[a.acceleration for a in actions],states)
        t=history.observations[-1].timestamp
        return [Observation(t+(k+1)*.1,float(s[1]),gap=float(s[0]),lead_speed=float(s[2])) for k,s in enumerate(trajectory)]


class SensorPlanner:
    def __init__(self, planner):
        self.planner=planner
        self.last_info=None

    def plan(self, observation, history):
        if not history.observations or history.observations[-1]!=observation:
            raise ValueError('History must end at current observation')
        states=observed_history(history)
        command,self.last_info=self.planner.command(states[-1],states)
        return Action(command,0.)
