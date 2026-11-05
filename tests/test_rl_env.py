"""RL observation/action/reward contracts and real/model transition parity."""
import unittest
from unittest.mock import patch
import numpy as np
from falconworld.data import step_truth
from falconworld.rl_env import (LongitudinalRLEnv, observation, decode_observation,
    encode, apply_action, reward_transition, reward_components, synthetic_step)


class ConstantTruthModel:
    def step(self, state, action, history):
        return step_truth(state, action, 0., 'id')


class RLEnvTests(unittest.TestCase):
    def test_raw_observation_roundtrip_and_fixed_scale_without_ood_clipping(self):
        h=np.tile([100.,40.,60.,-6.],(6,1))
        raw=observation(h,-6.)
        self.assertEqual(raw.shape,(25,))
        restored,u=decode_observation(raw)
        np.testing.assert_array_equal(restored,h);self.assertEqual(u,-6.)
        scaled=encode(h,-6.)
        np.testing.assert_allclose(scaled[:4],[2.,2.,3.,-1.]);self.assertEqual(scaled[-1],-1.)
        batch=encode(np.stack([h,h]),[-6.,2.]);self.assertEqual(batch.shape,(2,25))
        self.assertEqual(batch.dtype,np.float32)

    def test_action_delta_box_and_common_supervisor_override(self):
        safe=np.array([50.,10.,10.,0.])
        u,info=apply_action(safe,1.8,4);self.assertEqual(u,2.);self.assertFalse(info['fallback'])
        u,info=apply_action(safe,-5.8,0);self.assertEqual(u,-6.)
        unsafe=np.array([1.,10.,10.,0.])
        u,info=apply_action(unsafe,0.,4)
        self.assertEqual(info['requested_action_mps2'],.8)
        self.assertEqual(u,-6.);self.assertEqual(info['fallback'],'emergency_gap')

    def test_reward_exact_formula_bounds_collision_and_batch(self):
        s=np.array([16.,10.,10.,0.])
        self.assertAlmostEqual(reward_transition(s,s,0.,0.),.96)
        extreme=s.copy();extreme[0]=1.e6;extreme[1]=100.
        self.assertEqual(reward_transition(s,extreme,-6.,2.),-1.)
        collision=s.copy();collision[0]=0.
        self.assertEqual(reward_transition(s,collision,0.,0.),-100.)
        self.assertEqual(reward_components(s,collision,0.,0.)['collision_cost'],100.)
        np.testing.assert_allclose(reward_transition(s,np.stack([s,collision]),0.,0.),[.96,-100.])

    def test_truth_reproducibility_all_scenario_families_and_conditions(self):
        for family in ('constant','mild','brake','stopgo'):
            for condition in ('id','ood'):
                a,b=LongitudinalRLEnv(),LongitudinalRLEnv()
                oa,ia=a.reset(seed=123,family=family,condition=condition)
                ob,ib=b.reset(seed=123,family=family,condition=condition)
                np.testing.assert_array_equal(oa,ob)
                for k in range(120):
                    ra=a.step(k%5);rb=b.step(k%5)
                    np.testing.assert_array_equal(ra[0],rb[0]);self.assertEqual(ra[1:4],rb[1:4])
                    self.assertNotIn('params',ra[4])
                    if ra[2] or ra[3]:break
                self.assertLessEqual(a.tick,120)
                self.assertTrue(ra[2] or ra[3])

    def test_synthetic_matches_truth_and_never_reads_scenario_script(self):
        env=LongitudinalRLEnv();env.reset(seed=0,family='constant',condition='id')
        model=ConstantTruthModel()
        for index in (4,4,0,0,1,2,3,4):
            h,previous=env.history,env.previous_action
            original_history=h.copy()
            with patch('falconworld.rl_env.scenario_lead_accel',side_effect=AssertionError('script leak')):
                nh,reward,terminated,info=synthetic_step(model,h,previous,index)
            obs,truth_reward,truth_term,trunc,truth_info=env.step(index)
            np.testing.assert_allclose(nh,env.history,atol=0.)
            np.testing.assert_array_equal(h,original_history)
            self.assertEqual(reward,truth_reward);self.assertEqual(terminated,truth_term)
            self.assertEqual(info['applied_u'],truth_info['applied_u'])
            self.assertFalse(trunc)
            self.assertEqual(info['transition_source'],'model')

    def test_time_limit_bootstraps_and_terminal_blocks_subsequent_step(self):
        env=LongitudinalRLEnv();env.reset(seed=0,family='constant',condition='id')
        for _ in range(120):obs,reward,term,trunc,info=env.step(2)
        self.assertFalse(term);self.assertTrue(trunc);self.assertEqual(info['bootstrap_mask'],1.)
        self.assertEqual(info['episode']['length'],120)
        self.assertTrue(info['episode']['completed'])
        self.assertGreater(info['episode']['progress_m'],0)
        with self.assertRaises(RuntimeError):env.step(2)
        h=env.history;h[-1,0]=9999
        self.assertNotEqual(env.state[0],9999)

    def test_contact_is_terminal_not_truncation_and_keeps_negative_gap(self):
        h=np.tile([.01,10.,0.,0.],(6,1))
        nh,reward,term,info=synthetic_step(ConstantTruthModel(),h,0.,2)
        self.assertTrue(term);self.assertEqual(reward,-100.)
        self.assertLess(nh[-1,0],0.);self.assertTrue(info['collision'])
        with self.assertRaises(ValueError):synthetic_step(ConstantTruthModel(),nh,info['applied_u'],2)

    def test_stopping_jerk_uses_speed_difference_not_negative_actuator(self):
        h=np.tile([30.,.01,0.,-6.],(6,1))
        nh,reward,term,info=synthetic_step(ConstantTruthModel(),h,-6.,2)
        self.assertEqual(nh[-1,1],0.)
        self.assertAlmostEqual(info['measured_accel_mps2'],-.1)
        self.assertAlmostEqual(info['jerk_mps3'],-1.)

    def test_invalid_inputs_and_model_outputs_fail_explicitly(self):
        h=np.tile([20.,3.,3.,0.],(6,1))
        for index in (-1,5,1.,True):
            with self.assertRaises(ValueError):apply_action(h[-1],0.,index)
        with self.assertRaises(ValueError):encode(h,np.nan)
        with self.assertRaises(ValueError):encode(h[:5],0.)
        with self.assertRaises(ValueError):LongitudinalRLEnv().reset(seed=1,family='brake',condition='linear')
        class Invalid:
            def step(self,*args,**kwargs):return [np.nan,0,0,0]
        with self.assertRaises(ValueError):synthetic_step(Invalid(),h,0.,2)
        with self.assertRaises(RuntimeError):LongitudinalRLEnv().step(0)


if __name__=='__main__':unittest.main()
