"""Independent invariants and integration checks for the synthetic 2-D task."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from falconworld.ackermann import (AckermannConfig, PhysicalModel, ResidualModel,
    Route, AckermannCEM, ClassicTracker, bounded_action, constrain_sequences,
    step_ackermann, clearance, swept_collision, generate_dataset, train_models,
    simulate, wrap_angle)
from falconworld.planning import CEMOptimizer, CEMConfig


class AckermannTests(unittest.TestCase):
    def setUp(self):
        self.cfg=AckermannConfig()
        self.s=np.array([0.,0.,0.,2.,0.,0.])

    def test_old_state_euler_and_si_units(self):
        s=np.array([0.,0.,0.,2.,1.,.2])
        n=step_ackermann(s,[2.,.3])
        self.assertAlmostEqual(n[0],.2)
        self.assertAlmostEqual(n[3],2.1)
        self.assertAlmostEqual(n[2],.1*2*np.tan(.2)/(1+.015*4))
        np.testing.assert_array_equal(s,[0.,0.,0.,2.,1.,.2])

    def test_state_and_configuration_validation(self):
        with self.assertRaises(ValueError):AckermannConfig(wheelbase=0)
        with self.assertRaises(ValueError):AckermannConfig(steer_max=2)
        with self.assertRaises(ValueError):step_ackermann([0,0,0,-1,0,0],[0,0])
        with self.assertRaises(ValueError):step_ackermann(self.s,[np.nan,0])
        with self.assertRaises(ValueError):step_ackermann(self.s,[0,0],condition=0)
        with self.assertRaises(ValueError):AckermannCEM(PhysicalModel(),CEMOptimizer(CEMConfig(dt=.2)))

    def test_standstill_preserves_braking_actuator_and_prevents_reverse(self):
        s=np.array([0.,0.,0.,.01,-3.,0.])
        n=step_ackermann(s,[-3.,0.])
        self.assertEqual(n[3],0.)
        self.assertEqual(n[4],-3.)
        self.assertAlmostEqual((n[3]-s[3])/.1,-.1)

    def test_action_box_and_interstep_slew(self):
        a=bounded_action([200.,200.],[0.,0.])
        np.testing.assert_allclose(a,[.5,.08])
        seq=constrain_sequences(np.full((4,30,2),100.),[0.,0.],self.cfg)
        self.assertLessEqual(seq[:,:,1].max(),.5)
        self.assertTrue(np.all(np.abs(np.diff(seq,axis=1))<=[.5+.000001,.08+.000001]))

    def test_previous_action_cannot_bypass_limits_or_propagate_nan(self):
        for previous in ([10.,0.],[0.,1.],[np.nan,0.],[0.]):
            with self.assertRaises(ValueError):bounded_action([0.,0.],previous)

    def test_solver_exception_uses_logged_fallback(self):
        class Broken:
            name='broken'
            def action(self,*args):raise FloatingPointError('numeric failure')
        metrics,logs=simulate(Broken(),Route('straight'),self.s,max_steps=3)
        self.assertEqual(metrics['fallback_count'],3)
        self.assertTrue(all(r['fallback']=='solve_failure' for r in logs))
        self.assertTrue(all(r['accel_cmd_mps2']<=0 for r in logs))

    def test_circle_to_oriented_rectangle_and_road_corners(self):
        route=Route('obstacle')
        s=np.array([14.6,0.,0.,0.,0.,0.])
        road,obs=clearance(s,route)
        self.assertLess(obs,0)
        s[1]=2.;self.assertGreater(clearance(s,route)[1],0)
        s[1]=3.4;self.assertLess(clearance(s,route)[0],0)
        s=np.array([14.6,0.,np.pi/2,0.,0.,0.]);self.assertLess(clearance(s,route)[1],0)

    def test_swept_collision_catches_tunneling(self):
        a=np.array([12.,0.,0.,20.,0.,0.]);b=a.copy();b[0]=18.
        collision,road=swept_collision(a,b,Route('obstacle'),self.cfg)
        self.assertTrue(collision);self.assertFalse(road)

    def test_same_current_state_actions_branch_only_after_actuator_delay(self):
        u0=np.array([-2.,-.3]);u1=np.array([1.5,.3])
        a=step_ackermann(self.s,u0);b=step_ackermann(self.s,u1)
        np.testing.assert_allclose(a[:4],b[:4])
        self.assertNotEqual(a[4],b[4]);self.assertNotEqual(a[5],b[5])
        for _ in range(20):a=step_ackermann(a,u0);b=step_ackermann(b,u1)
        self.assertGreater(np.linalg.norm(a[:2]-b[:2]),1)

    def test_training_split_and_model_serialization(self):
        with tempfile.TemporaryDirectory() as d:
            data,manifest=generate_dataset(d)
            ids={split:{r['episode'] for r in manifest if r['split']==split} for split in ('train','validation','test')}
            self.assertFalse(ids['train']&ids['test']);self.assertFalse(ids['validation']&ids['test'])
            models=train_models(data,d)
            pred=models[1].predict_batch(data['test_state'],data['test_action'])
            # Identifiable noise-free rich physics recovers the true simulator.
            np.testing.assert_allclose(pred,data['test_next'],atol=1e-7)
            for m in models:self.assertTrue(np.all(np.isfinite(m.predict_batch(data['test_state'],data['test_action']))))
            loaded=ResidualModel.load(Path(d)/'residual.npz')
            np.testing.assert_allclose(loaded.weights,models[2].weights)
            self.assertEqual(len(ids['test']),9)

    def test_recursive_prediction_uses_own_state(self):
        model=PhysicalModel()
        u=np.array([[1.,.2]])
        first=model.predict_batch(self.s[None],u)
        second=model.predict_batch(first,u)
        repeated=model.predict_batch(self.s[None],u)
        self.assertGreater(second[0,0],repeated[0,0])
        self.assertGreater(second[0,3],first[0,3])

    def test_cem_objective_and_closed_loop_regression(self):
        cfg=CEMConfig(horizon=12,candidates=24,iterations=2,elites=6)
        ctrl=AckermannCEM(PhysicalModel(),CEMOptimizer(cfg))
        u,info=ctrl.action(self.s,0.,Route('straight'),np.zeros(2))
        self.assertTrue(np.isfinite(info['cost']))
        self.assertLessEqual(abs(u[1]),.08+1e-9)
        route=Route('straight',goal=3.)
        metric,log=simulate(ctrl,route,self.s,max_steps=40,deadline_ms=1000)
        self.assertTrue(metric['completed'])
        self.assertEqual(metric['collision'],0)
        self.assertGreater(len(log),1)

    def test_partial_resume_reuses_frozen_time_profile_and_preserves_logs(self):
        import run_ackermann as runner
        from unittest.mock import patch
        import hashlib
        spec=dict(seeds=[527],pairs_per_seed=1,horizon=3,candidates=8,iterations=1,budgets=['candidates','time'])
        actual_simulate=runner.simulate
        calls=[0]
        def short_run(*args,**kwargs):
            calls[0]+=1
            if calls[0]==5:raise RuntimeError('simulated interruption after committed scene')
            return actual_simulate(*args,**kwargs,max_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            argv=['run_ackermann.py','--preset','smoke','--output',folder]
            with patch.dict(runner.PRESETS,smoke=spec),patch('sys.argv',argv),patch.object(runner,'render_existing'),patch.object(runner,'simulate',short_run):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):runner.main()
            profile=(Path(folder)/'profiles.json').read_bytes()
            logs={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(folder).glob('seed_*/candidates_*.csv')}
            self.assertEqual(len(logs),4)
            calls[0]=100
            original_action=runner.AckermannCEM.action
            def reject_reprofile(ctrl,state,*args):
                if np.array_equal(state,np.array([0.,.1,0.,2.4,0.,0.])):
                    raise AssertionError('resume attempted to measure validation profile again')
                return original_action(ctrl,state,*args)
            with patch.dict(runner.PRESETS,smoke=spec),patch('sys.argv',argv),patch.object(runner,'render_existing'),patch.object(runner,'simulate',short_run),patch.object(runner.AckermannCEM,'action',reject_reprofile):runner.main()
            self.assertEqual(profile,(Path(folder)/'profiles.json').read_bytes())
            for p,digest in logs.items():self.assertEqual(digest,hashlib.sha256(p.read_bytes()).hexdigest())
            audit=json.loads((Path(folder)/'completeness.json').read_text())
            self.assertTrue(audit['verified_complete']);self.assertEqual(audit['actual'],32)

    def test_goal_crossing_does_not_override_collision_or_road_exit(self):
        for route,state in ((Route('obstacle',goal=14.1),np.array([14.,0.,0.,2.,0.,0.])),
                            (Route('straight',goal=.1),np.array([0.,3.5,0.,2.,0.,0.]))):
            metrics,_=simulate(ClassicTracker(),route,state,max_steps=1)
            self.assertTrue(metrics['collision'] or metrics['road_exit'])
            self.assertEqual(metrics['completed'],0)

    def test_collision_stops_episode_and_reports_completion_separately(self):
        route=Route('obstacle')
        s=np.array([14.,0.,0.,2.,0.,0.])
        metrics,log=simulate(ClassicTracker(),route,s)
        self.assertEqual(metrics['collision'],1)
        self.assertEqual(metrics['completed'],0)
        self.assertLess(len(log),160)

if __name__=='__main__':unittest.main()
