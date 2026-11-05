"""Validate that the analysis distinguishes true one-step checks from recursion."""
import numpy as np
from falconworld.models import PhysicsModel
from scripts.analyze_validation import evaluate_validation


def test_known_accelerating_leader_error_accumulates_only_in_recursive_gap():
    state=np.array([20.,2.,2.,0.]);states=[];targets=[];histories=[]
    history=np.tile(state,(6,1))
    for _ in range(30):
        states.append(state.copy());histories.append(history.copy())
        future=state.copy()
        future[0]+=.1*(state[2]-state[1])
        future[2]+=.1
        targets.append(future.copy());state=future
        history=np.concatenate((history[1:],state[None]))
    arrays=dict(state=np.asarray(states),next_state=np.asarray(targets),history=np.asarray(histories),
                action=np.zeros(30),episode=np.zeros(30,dtype=int),family=np.repeat('mild',30))
    rows=evaluate_validation({'cv':PhysicsModel()},arrays,seed=0)
    three=next(r for r in rows if r['horizon_s']==3.)
    assert abs(three['recursive_gap_m']-4.35)<1e-10
    assert three['teacher_forced_endpoint_gap_m']==0
    assert abs(three['teacher_forced_endpoint_lead_mps']-.1)<1e-10
    assert abs(three['recursive_lead_mps']-3.)<1e-10
