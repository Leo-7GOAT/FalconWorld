import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from falconworld.rl_evaluation import evaluation_cases,evaluate_policy,summarize
from falconworld.rl_env import observation


class GreedyAgent:
    def act(self, encoded, explore=False):
        assert explore is False
        assert encoded.shape == (25,) and np.isfinite(encoded).all()
        return 2


class StopBoundaryEnv:
    dt=.1
    def reset(self, **case):
        self.history=np.repeat([[10.,.1,0.,-6.]],6,axis=0)
        self.t=0
        return observation(self.history,0.),{}
    def step(self,index):
        self.t+=1
        new=np.array([10. if self.t==1 else -1.,0.,0.,-6.])
        self.history=np.concatenate((self.history[1:],new[None]),axis=0)
        return observation(self.history,-6.),1.,self.t==2,False,{'fallback':'test_stop'}


class EvaluationTests(unittest.TestCase):
    def test_fixed_split_and_seed_isolation(self):
        test=evaluation_cases(0,'standard')
        self.assertEqual(len(test),80)
        ids=lambda rows:{r['seed'] for r in rows}
        self.assertFalse(ids(test)&ids(evaluation_cases(1,'standard')))
        self.assertFalse(ids(test)&ids(evaluation_cases(0,'standard','val')))
        self.assertEqual(len(evaluation_cases(0)),16)

    def test_actual_velocity_jerk_at_stop_and_failure_logging(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'eval'
            cases=[dict(seed=8_000_000,family='constant',condition='id')]
            rows=evaluate_policy(GreedyAgent(),cases,root,env_factory=StopBoundaryEnv)
            r=rows[0]
            self.assertEqual(r['steps'],2)
            self.assertEqual(r['collision'],1)
            self.assertEqual(r['completed'],0)
            self.assertEqual(r['return'],2.)
            self.assertAlmostEqual(r['jerk_rms_mps3'],10.)
            self.assertAlmostEqual(r['progress_m'],.01)
            self.assertEqual(r['fallback_count'],2)
            with (root/'trace_id_constant_8000000.csv').open() as f:trace=list(csv.DictReader(f))
            self.assertAlmostEqual(float(trace[0]['measured_accel_mps2']),-1.)
            self.assertAlmostEqual(float(trace[1]['measured_accel_mps2']),0.)
            with self.assertRaises(ValueError):evaluate_policy(GreedyAgent(),cases,root,env_factory=StopBoundaryEnv)

    def test_paired_episode_bootstrap_and_missing_pair_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            cases=[dict(seed=8_000_000,family='constant',condition='id')]
            rows=evaluate_policy(GreedyAgent(),cases,Path(d)/'a',method='ppo',env_factory=StopBoundaryEnv)
            changed=dict(rows[0],method='wm_ppo')
            changed['return']+=3.
            summary,effects=summarize(rows+[changed],Path(d)/'summary',bootstrap_samples=50)
            effect=next(e for e in effects if e['family']=='all' and e['metric']=='return')
            self.assertEqual((effect['difference'],effect['ci_low'],effect['ci_high']),(3.,3.,3.))
            self.assertEqual(effect['pairs'],1)
            self.assertEqual(effect['training_seeds'],1)
            with self.assertRaises(ValueError):summarize(rows+[dict(changed,scenario_seed=99)],Path(d)/'missing')
            with self.assertRaises(ValueError):summarize(rows+rows,Path(d)/'duplicate')


if __name__=='__main__':unittest.main()
