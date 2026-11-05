"""Hand-calculated cases and physical invariants for the learner's functions."""

from dataclasses import FrozenInstanceError
import unittest

from falconworld.nominal import NominalConfig, predict_one_step
from falconworld.state import LongitudinalState


class Lesson01Tests(unittest.TestCase):
    def assert_state(self, actual, expected):
        self.assertIsInstance(actual, LongitudinalState)
        values = actual.as_tuple()
        self.assertIsInstance(values, tuple)
        self.assertEqual(len(values), 4)
        for index, (value, target) in enumerate(zip(values, expected)):
            self.assertAlmostEqual(value, target, places=10, msg=f"component {index}")

    def test_01_tuple_order(self):
        state = LongitudinalState(20.0, 10.0, 8.0, -0.5)
        self.assertEqual(state.as_tuple(), (20.0, 10.0, 8.0, -0.5))

    def test_02_first_step_from_hand_calculation(self):
        state = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        predicted = predict_one_step(state, -2.0)
        self.assert_state(predicted, (19.8, 10.0, 8.0, -0.4))
        self.assertIsNot(predicted, state)
        self.assertEqual(state.as_tuple(), (20.0, 10.0, 8.0, 0.0))

    def test_03_second_step_uses_old_actuator_and_old_speeds(self):
        state = LongitudinalState(19.8, 10.0, 8.0, -0.4)
        self.assert_state(predict_one_step(state, -2.0), (19.6, 9.96, 8.0, -0.72))

    def test_04_equal_speed_equilibrium(self):
        state = LongitudinalState(30.0, 12.0, 12.0, 0.0)
        self.assert_state(predict_one_step(state, 0.0), (30.0, 12.0, 12.0, 0.0))

    def test_05_command_limits(self):
        state = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        self.assert_state(predict_one_step(state, 100.0), (19.8, 10.0, 8.0, 0.4))
        self.assert_state(predict_one_step(state, -100.0), (19.8, 10.0, 8.0, -1.2))

    def test_06_stopping_prevents_reverse_but_preserves_actuator_state(self):
        state = LongitudinalState(10.0, 0.1, 0.0, -6.0)
        self.assert_state(predict_one_step(state, -6.0), (9.99, 0.0, 0.0, -6.0))

    def test_07_negative_gap_is_not_hidden(self):
        state = LongitudinalState(0.05, 10.0, 0.0, 0.0)
        self.assert_state(predict_one_step(state, 0.0), (-0.95, 10.0, 0.0, 0.0))

    def test_08_custom_config_is_used(self):
        config = NominalConfig(dt=0.2, tau=0.4, accel_min=-4.0, accel_max=1.0)
        state = LongitudinalState(10.0, 5.0, 4.0, 0.0)
        self.assert_state(predict_one_step(state, 10.0, config), (9.8, 5.0, 4.0, 0.5))

    def test_09_state_is_immutable(self):
        state = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        with self.assertRaises(FrozenInstanceError):
            state.gap = 100.0

    def test_10_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            LongitudinalState(10.0, -1.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            LongitudinalState(float("nan"), 1.0, 1.0, 0.0)
        for kwargs in ({"dt": 0.0}, {"tau": 0.0}, {"dt": 0.6}, {"accel_max": -1.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NominalConfig(**kwargs)
        state = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        with self.assertRaises(ValueError):
            predict_one_step(state, float("inf"))


if __name__ == "__main__":
    unittest.main()
