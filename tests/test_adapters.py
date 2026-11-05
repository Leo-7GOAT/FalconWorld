import math
import unittest
from falconworld.contracts import Action,History,Observation,SafetySupervisor
from falconworld.adapters import R550Config,R550ShadowAdapter,carla_control


def config():
    # Synthetic test fixture, explicitly NOT R550 measured calibration.
    return R550Config(.25,.4,.6,1.,.7,1.5,.1,.25,'/test/drive','/test/odom')


class AdapterTests(unittest.TestCase):
    def test_si_to_carla_direction_and_bounds(self):
        self.assertEqual(carla_control(Action(99,.2),.4),dict(throttle=1.,brake=0.,steer=-.5))
        self.assertEqual(carla_control(Action(-99,-99),.4),dict(throttle=0.,brake=1.,steer=1.))

    def test_history_uses_real_velocity_difference_at_stop(self):
        h=History();h.append(Observation(1,.1));h.append(Observation(1.1,0),Action(-4))
        self.assertAlmostEqual(h.acceleration(),-1)
        with self.assertRaises(ValueError):h.append(Observation(1.1,0))

    def test_safety_freshness_deadline_and_gap(self):
        s=SafetySupervisor()
        self.assertEqual(s.check(Observation(1,1),Action(1),1.3).reason,'stale_or_future_observation')
        self.assertEqual(s.check(Observation(1,1),Action(1),1,1).reason,'planner_timeout')
        self.assertEqual(s.check(Observation(1,2,gap=.3),Action(1),1).reason,'stopping_distance')
        self.assertFalse(s.check(Observation(1,1),Action(99,99),1).triggered)

    def test_invalid_timing_and_conversion_cannot_bypass_limits(self):
        for value in (float('nan'), float('inf'), 0.):
            with self.assertRaises(ValueError):SafetySupervisor(max_age=value)
            with self.assertRaises(ValueError):carla_control(Action(0),value)
        self.assertEqual(SafetySupervisor().check(Observation(1,0),Action(0),1,-.1).reason,
                         'planner_timeout')

    def test_shadow_no_transport_and_timeout_stop(self):
        a=R550ShadowAdapter(config())
        a.shadow(Observation(1,.2),Action(1,.4),1)
        row=a.shadow(Observation(1.1,.2),Action(1,.4),1.1)
        self.assertLessEqual(row['target_speed_m_s'],.27+1e-9)
        self.assertLessEqual(row['steering_rad'],.06+1e-9)
        self.assertFalse(row['sent']);self.assertFalse(row['acknowledged'])
        stop=a.shadow(Observation(1.2,.2),Action(1),2)
        self.assertEqual(stop['target_speed_m_s'],0)
        self.assertEqual(stop['reason'],'stale_or_future_observation')

    def test_unverified_config_and_invalid_observation(self):
        self.assertFalse(config().verified)
        with self.assertRaises(ValueError):Observation(0,float('nan'))
        with self.assertRaises(ValueError):Action(float('inf'))
        with self.assertRaises(ValueError):R550Config(.25,.4,.6,1,.7,1.5,.1,.25,'a','b',True,'')

    def test_new_feedback_does_not_hide_a_sampling_gap(self):
        a=R550ShadowAdapter(config())
        a.shadow(Observation(1,.2),Action(1),1)
        stop=a.shadow(Observation(3,.2),Action(1),3)
        self.assertEqual(stop['reason'],'feedback_gap')
        self.assertEqual(stop['target_speed_m_s'],0)

    def test_carla_observation_coordinates_timestamp_and_restore(self):
        from types import SimpleNamespace as S
        from falconworld.adapters import CarlaAdapter
        original=S(synchronous_mode=False)
        class World:
            def __init__(self):self.settings=original
            def get_settings(self):
                return S(**self.settings.__dict__)
            def apply_settings(self,settings):self.settings=settings
            def get_snapshot(self):return S(timestamp=S(elapsed_seconds=2.),frame=9)
        class Vehicle:
            def get_physics_control(self):return S(wheels=[S(max_steer_angle=40.)])
            def get_transform(self):return S(location=S(x=4.,y=3.),rotation=S(yaw=90.))
            def get_velocity(self):return S(x=3.,y=4.)
        w=World()
        with CarlaAdapter(w,Vehicle()) as adapter:
            obs=adapter.observe()
            self.assertEqual((obs.x,obs.y,obs.speed,obs.timestamp,obs.frame),(4.,-3.,5.,2.,9))
            self.assertAlmostEqual(obs.yaw,-math.pi/2)
            self.assertTrue(w.settings.synchronous_mode)
        self.assertFalse(w.settings.synchronous_mode)


if __name__=='__main__':unittest.main()
