"""Independent audit must handle stopping and the unlogged terminal state."""
import pytest
from scripts.audit_longitudinal import reconstruct,expected_identities


def point(step,gap,speed,actuator,measured,fallback=''):
    return dict(step=step,time_s=step*.1,gap_m=gap,speed_mps=speed,lead_mps=0.,
                actuator_mps2=actuator,command_mps2=-6.,measured_accel_mps2=measured,
                cycle_s=.11 if fallback else .01,inference_s=.005,fallback=fallback)


def test_stopping_jerk_uses_velocity_differences_not_latent_acceleration():
    log=[point(0,10.,.15,-1.,-1.),point(1,9.985,.05,-1.,-.5),
         point(2,9.98,0.,-1.,0.,'deadline_exceeded')]
    metrics,cycles,reasons=reconstruct(log,.1,3)
    assert metrics['jerk_rms_mps3']==pytest.approx(5.)
    assert metrics['final_speed_mps']==0
    assert metrics['progress_m']==pytest.approx(.02)
    assert metrics['timeout_rate']==pytest.approx(1/3)
    assert metrics['fallback_rate']==pytest.approx(1/3)
    assert reasons['deadline_exceeded']==1
    assert metrics['completed']==0 # stationary progress cannot meet 2.5m minimum


def test_collision_and_gap_error_include_reconstructed_last_future_state():
    metrics,_,_=reconstruct([point(0,.05,1.,-6.,-6.)],.1,80)
    assert metrics['final_gap_m']==pytest.approx(-.05)
    assert metrics['min_gap_m']==pytest.approx(-.05)
    assert metrics['collision']==1
    assert metrics['completed']==0
    assert metrics['gap_rmse_m']==pytest.approx(4.53)
    assert metrics['jerk_rms_mps3']==0


def test_audit_rejects_forged_measured_acceleration_and_counts_full_pairs():
    with pytest.raises(AssertionError):
        reconstruct([point(0,10.,0.,-6.,-6.)],.1,80)
    expected=expected_identities(dict(preset='smoke',seeds=[0],budget='both'))
    assert len(expected)==304
    assert len({key[:-1] for key in expected})==32
