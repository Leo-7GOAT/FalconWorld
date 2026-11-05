import tempfile
from pathlib import Path
import unittest
import numpy as np
from falconworld.r550_worldmodel import LowSpeedModel,LowSpeedMPC


class LowSpeedDeploymentTests(unittest.TestCase):
    def test_observable_history_and_delay_branch(self):
        model=LowSpeedModel([.5,.25,2])
        state=np.array([.05,.05,.05])
        stop=model.step(state,0)
        go=model.step(state,.3)
        self.assertEqual(stop[0],go[0]) # Previous SENT command controls delay sample.
        self.assertGreater(model.step(go,.3)[0],model.step(stop,0)[0])
        np.testing.assert_array_equal(state,[.05,.05,.05])

    def test_saved_model_preserves_domain_and_origin(self):
        model=LowSpeedModel([.5,.25,2])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'speed.npz';model.save(path)
            loaded=LowSpeedModel.load(path)
            self.assertEqual(loaded.origin,'SYNTHETIC_UNCALIBRATED')
            np.testing.assert_array_equal(model.step([.1,.09,.2],.1),loaded.step([.1,.09,.2],.1))
            with self.assertRaises(ValueError):loaded.step([8,7,6],.2)
            with self.assertRaises(ValueError):loaded.step([.1,.1,.1],-1)

    def test_shared_cem_uses_velocity_command_limits(self):
        planner=LowSpeedMPC(LowSpeedModel([.5,.25,2]),candidates=16,horizon=10)
        command,info=planner.plan([0,0,0],.2)
        self.assertGreater(command,0)
        self.assertLessEqual(command,.015+1e-12)
        self.assertFalse(info['fallback'])
        self.assertEqual(info['finite_evaluations'],48)


if __name__=='__main__':unittest.main()
