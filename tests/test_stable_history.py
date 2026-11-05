"""Regression for the repair's specific promises, not driving safety claims."""
import numpy as np
from falconworld.models import PhysicsModel
from falconworld.stable_history import StableHistoryModel,project_stable


def test_projection_roots_are_inside_registered_radius():
    for coefficients in ([2.5,-1.4],[.4,.8],[-2.,-.8]):
        projected=project_stable(coefficients,.95)
        assert np.max(abs(np.roots([1.,-projected[0],-projected[1]])))<=.9500000001
        assert np.isfinite(projected).all()


def test_zero_history_does_not_invent_leader_acceleration_and_roundtrips(tmp_path):
    physics=PhysicsModel(enriched=[.45,.75,.045],lead='ca')
    model=StableHistoryModel(physics,[1.0589810675612303,-.09372386369657686])
    path=tmp_path/'model.npz';model.save(path)
    restored=StableHistoryModel.load(path)
    state=np.array([30.,5.,5.,0.]);history=np.tile(state,(6,1))
    for _ in range(100):
        next_state=model.step(state,0.,history)
        np.testing.assert_array_equal(next_state,restored.step(state,0.,history))
        np.testing.assert_array_equal(next_state,state)
        history=np.concatenate((history[1:],next_state[None]))
        state=next_state
    # Batched load/inference includes nonzero history and distinct candidate actions.
    history[:,2]+=np.linspace(.4,0,6)
    batch=np.tile(state,(3,1))
    np.testing.assert_array_equal(model.step(batch,[-4.,0.,2.],history),
                                  restored.step(batch,[-4.,0.,2.],history))
