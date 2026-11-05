import unittest
import numpy as np
from falconworld.planning import CEMOptimizer, CEMConfig, SafetySupervisor,LongitudinalMPC
from falconworld.models import PhysicsModel


class PlanningTests(unittest.TestCase):
    def test_optimizer_constraints_and_repeatability(self):
        opt = CEMOptimizer(CEMConfig(horizon=10, candidates=32, elites=4))
        seen = []
        def objective(seq):
            seen.append(seq.copy())
            return ((seq-1)**2).sum(axis=(1, 2))
        first, _ = opt.optimize(objective, [-6], [2], [0], slew=.3, seed=5)
        second, _ = opt.optimize(objective, [-6], [2], [0], slew=.3, seed=5)
        np.testing.assert_array_equal(first, second)
        for seq in seen:
            self.assertTrue(np.all(seq <= 2) and np.all(seq >= -6))
            self.assertLessEqual(np.max(np.abs(np.diff(np.concatenate((np.zeros((32,1,1)), seq), axis=1), axis=1))), .30000001)

    def test_solver_failure_and_emergency(self):
        safety = SafetySupervisor()
        self.assertEqual(safety.filter([20,10,10,0], np.nan)[1], 'solver_failure')
        self.assertEqual(safety.filter([1,10,0,0], 2)[0], -6)
        best, info = CEMOptimizer().optimize(lambda seq: np.full(len(seq), np.nan), [-6], [2])
        self.assertIsNone(best)
        self.assertEqual(info['finite_evaluations'], 0)

    def test_invalid_observation_and_model_failure_return_recorded_brake(self):
        planner=LongitudinalMPC(PhysicsModel(),CEMConfig(horizon=1,candidates=2,elites=1))
        command,info=planner.command([float('nan'),1,1,0],np.zeros((6,4)))
        self.assertEqual(command,-6)
        self.assertEqual(info['fallback'],'invalid_observation')
        class BrokenModel:
            def step(self,*args,**kwargs):raise FloatingPointError('Numerical failure')
        planner=LongitudinalMPC(BrokenModel(),CEMConfig(horizon=1,candidates=2,elites=1))
        command,info=planner.command([20,1,1,0],np.zeros((6,4)))
        self.assertEqual(command,-6)
        self.assertIn('Numerical failure',info['failure_detail'])
