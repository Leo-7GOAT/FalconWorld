"""Scientific/data invariants, component causality and artifact portability."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from falconworld.data import DT, generate_dataset, load_dataset, step_truth
from falconworld.models import PhysicsModel, _fit_physics, train_models, load_models, estimate_sensor_state


class ModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = generate_dataset(cls.root / "data")
        cls.models = train_models(cls.data, cls.root / "models", seed=7)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_episode_split_and_causal_history(self):
        ids = [set(self.data[s]["episode"]) for s in ("train", "val", "test", "ood")]
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                self.assertFalse(a & b)
        for split in ("train", "val", "test", "ood"):
            d = self.data[split]
            np.testing.assert_array_equal(d["history"][:, -1], d["state"])
            # Earliest histories are repeat padded; later histories contain only prior states.
            for ep in np.unique(d["episode"]):
                ix = np.where(d["episode"] == ep)[0]
                if len(ix) > 6:
                    np.testing.assert_array_equal(d["history"][ix[6]], d["state"][ix[1:7]])

    def test_artifacts_identical_and_no_pickle(self):
        loaded = load_models(self.root / "models")
        data = load_dataset(self.root / "data")
        self.assertEqual(data["metadata"]["sha256"], self.data["metadata"]["sha256"])
        v = data["val"]
        for name, model in self.models.items():
            np.testing.assert_array_equal(model.step(v["state"][:30], v["action"][:30], v["history"][:30]),
                                          loaded[name].step(v["state"][:30], v["action"][:30], v["history"][:30]))

    def test_identification_is_sufficient_for_fixed_first_order(self):
        r = np.random.default_rng(34)
        states = np.column_stack([np.full(200, 60.), r.uniform(0, 18, 200), r.uniform(0, 18, 200), r.uniform(-5, 1, 200)])
        actions = r.uniform(-6, 2, 200)
        next_states = step_truth(states, actions, np.zeros(200), "linear")
        tau, _ = _fit_physics({"state": states, "action": actions, "next_state": next_states})
        self.assertAlmostEqual(tau, .8, places=12)
        np.testing.assert_allclose(PhysicsModel(tau=tau).step(states, actions), next_states, atol=1e-12)

    def test_enriched_calibration_and_train_only(self):
        strong = self.models["physical_enriched"]
        np.testing.assert_allclose(strong.enriched, [.45, .75, .045], atol=1e-8)
        # Changing every held-out label cannot change the calibration function.
        tau, params = _fit_physics(self.data["train"])
        self.assertAlmostEqual(tau, self.models["physical_cv"].tau)
        np.testing.assert_allclose(params, strong.enriched)

    def test_candidate_actions_affect_ego_not_exogenous_leader(self):
        state = np.array([25., 8., 7., -.3])
        h = np.tile(state, (6, 1)); h[:, 2] += np.linspace(.5, 0, 6)
        for model in self.models.values():
            brake = model.step(state, -6., h)
            accel = model.step(state, 2., h)
            self.assertGreater(accel[3], brake[3])
            self.assertEqual(accel[2], brake[2])
            self.assertEqual(brake.shape, (4,))
            np.testing.assert_array_equal(state, [25, 8, 7, -.3])

    def test_recursive_and_stopping_boundary(self):
        state = np.array([30., .01, 0., -6.])
        for model in self.models.values():
            s = state.copy()
            h = np.tile(s, (6, 1))
            for _ in range(30):
                s = model.step(s, -6, h)
                h = np.concatenate([h[1:], s[None]])
                self.assertGreaterEqual(s[1], 0)
                self.assertTrue(np.isfinite(s).all())
                self.assertTrue(-6 <= s[3] <= 2)
        with self.assertRaises(ValueError):
            step_truth(np.array([0, 0, 0, 0]), 0, 0)

    def test_ensemble_interval_is_finite_nonnegative(self):
        v = self.data["val"]
        m = self.models["ensemble"]
        widths = m.interval_radius(v["state"][:30], v["action"][:30], v["history"][:30])
        self.assertEqual(widths.shape, (30, 4))
        self.assertTrue(np.isfinite(widths).all())
        self.assertTrue((widths >= 0).all())

    def test_sensor_estimator_cannot_read_privileged_actuator(self):
        history = np.tile([20., 5., 6., -6.], (6, 1))
        history[-2, 1], history[-1, 1] = 5.1, 5.
        a = estimate_sensor_state([20, 5, 6, -6], history)
        history[:, 3] = 1.7
        b = estimate_sensor_state([20, 5, 6, 2], history)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a[3], -1.)
        stopped = np.tile([20., 0., 0., -6.], (6, 1))
        self.assertEqual(estimate_sensor_state(stopped[-1], stopped)[3], 0.)


if __name__ == "__main__":
    unittest.main()
