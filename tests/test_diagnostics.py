"""Key recursive uncertainty and observation-boundary invariants."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from falconworld.diagnostics import recursive_trajectory, recursive_members, calibrate_intervals, interval_evaluation, sensor_history
from falconworld.models import PhysicsModel, EnsembleModel


class HistorySensitive:
    def __init__(self, factor):
        self.factor = factor

    def step(self, state, action, history):
        out = state.copy()
        out[0] = history[-1, 0]*self.factor
        return out


def fixed_episodes():
    state = np.tile([50., 2., 2., 0.], (120, 1))
    target = state.copy()
    target[:60, 0] += 1
    target[60:, 0] += 3
    return {"state": state, "next_state": target, "action": np.zeros(120),
            "history": np.repeat(state[:, None], 6, axis=1), "episode": np.repeat([10, 20], 60),
            "family": np.repeat("constant", 120)}


class DiagnosticsTests(unittest.TestCase):
    def test_each_ensemble_member_rolls_own_history(self):
        ens = EnsembleModel([HistorySensitive(2), HistorySensitive(3)])
        state = np.array([1., 2., 2., 0.])
        mean, std, members = recursive_members(ens, state, np.zeros(3), np.tile(state, (6, 1)))
        np.testing.assert_array_equal(members[0, :, 0], [2, 4, 8])
        np.testing.assert_array_equal(members[1, :, 0], [3, 9, 27])
        np.testing.assert_array_equal(mean[:, 0], [2.5, 6.5, 17.5])
        np.testing.assert_array_equal(std[:, 0], [.5, 2.5, 9.5])
        np.testing.assert_array_equal(
            recursive_trajectory(ens, state, np.zeros(3), np.tile(state, (6, 1))), mean)

    def test_validation_episode_calibration_and_test_coverage(self):
        ens = EnsembleModel([PhysicsModel(), PhysicsModel()])
        data = fixed_episodes()
        radius, meta = calibrate_intervals(ens, data)
        self.assertEqual(meta["calibration_episodes"], 2)
        np.testing.assert_allclose(radius[:, 0], 2.8)
        np.testing.assert_array_equal(radius[:, 1:], 0)
        frozen = radius.copy()
        with tempfile.TemporaryDirectory() as tmp:
            rows = interval_evaluation(ens, {"test": data, "ood": data}, radius, Path(tmp))
        gap = [row for row in rows if row["component"] == "gap"]
        self.assertTrue(all(row["episodes"] == 2 for row in gap))
        self.assertTrue(all(row["episode_mean_coverage"] == .5 for row in gap))
        np.testing.assert_array_equal(radius, frozen)

    def test_sensor_history_removes_latent_actuator(self):
        h = np.tile([20., 4., 3., -6.], (6, 1))
        h[:, 1] = [5, 4.8, 4.6, 4.4, 4.2, 4]
        expected = sensor_history(h)
        h[:, 3] = np.arange(6)*1000
        np.testing.assert_array_equal(sensor_history(h), expected)
        np.testing.assert_allclose(expected[1:, 3], -2.)
        self.assertEqual(expected[0, 3], 0.)
        np.testing.assert_array_equal(h[:, :3], expected[:, :3])


if __name__ == "__main__":
    unittest.main()
