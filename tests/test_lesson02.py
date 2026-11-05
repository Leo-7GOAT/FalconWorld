"""Lesson 02: hand calculations, stopping, collision, and model separation."""

import unittest

from falconworld.environment import EnvironmentConfig, StepResult, step_environment
from falconworld.nominal import NominalConfig, predict_one_step
from falconworld.state import LongitudinalState


class Lesson02Tests(unittest.TestCase):
    def assert_state(self, actual, expected):
        self.assertIsInstance(actual, LongitudinalState)
        for name, value, target in zip(
            ("gap", "v_ego", "v_lead", "a_actuator"), actual.as_tuple(), expected
        ):
            self.assertAlmostEqual(value, target, places=10, msg=name)

    def assert_step(self, actual, expected_state, expected_collision=False):
        self.assertIsInstance(actual, StepResult)
        self.assert_state(actual.state, expected_state)
        self.assertIs(actual.collision, expected_collision)

    def test_01_first_step_and_unchanged_input(self):
        initial = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        result = step_environment(initial, -2.0, -1.0)
        self.assert_step(result, (19.8, 10.0, 7.9, -0.25))
        self.assertIsNot(result.state, initial)
        self.assertEqual(initial.as_tuple(), (20.0, 10.0, 8.0, 0.0))

    def test_02_second_step_uses_old_speeds_and_actuator(self):
        initial = LongitudinalState(19.8, 10.0, 7.9, -0.25)
        result = step_environment(initial, -2.0, -1.0)
        self.assert_step(result, (19.59, 9.975, 7.8, -0.46875))

    def test_03_matched_parameters_and_constant_leader_match_nominal(self):
        env_config = EnvironmentConfig(tau_true=0.5)
        nominal_config = NominalConfig()
        actual = LongitudinalState(100.0, 10.0, 8.0, 0.0)
        predicted = actual
        commands = (-2.0, -2.0, 1.0, 1.0, -0.5, 0.5, 0.0, 2.0, -6.0, 0.0)
        for index, command in enumerate(commands):
            with self.subTest(step=index + 1):
                result = step_environment(actual, command, 0.0, env_config)
                predicted = predict_one_step(predicted, command, nominal_config)
                self.assert_step(result, predicted.as_tuple())
                actual = result.state

    def test_04_leader_acceleration_changes_next_speed_before_gap(self):
        initial = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        constant = step_environment(initial, 0.0, 0.0).state
        braking = step_environment(initial, 0.0, -2.0).state
        self.assert_state(constant, (19.8, 10.0, 8.0, 0.0))
        self.assert_state(braking, (19.8, 10.0, 7.8, 0.0))
        # The next interval starts with different leader speeds.
        constant_second = step_environment(constant, 0.0, 0.0)
        braking_second = step_environment(braking, 0.0, 0.0)
        self.assertAlmostEqual(constant_second.state.gap, 19.6, places=10)
        self.assertAlmostEqual(braking_second.state.gap, 19.58, places=10)

    def test_05_commands_are_limited_before_actuator_response(self):
        initial = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        for command, expected_actuator in ((100.0, 0.25), (-100.0, -0.75)):
            with self.subTest(command=command):
                result = step_environment(initial, command, 0.0)
                self.assert_step(result, (19.8, 10.0, 8.0, expected_actuator))

    def test_06_both_vehicles_stop_without_erasing_braking_state(self):
        initial = LongitudinalState(10.0, 0.1, 0.05, -6.0)
        stopped = step_environment(initial, -6.0, -1.0)
        self.assert_step(stopped, (9.995, 0.0, 0.0, -6.0))
        held = step_environment(stopped.state, -6.0, -1.0)
        self.assert_step(held, (9.995, 0.0, 0.0, -6.0))

    def test_07_negative_gap_is_preserved_and_ends_the_episode(self):
        initial = LongitudinalState(0.05, 10.0, 0.0, 0.0)
        result = step_environment(initial, 0.0, 0.0)
        self.assert_step(result, (-0.95, 10.0, 0.0, 0.0), True)
        with self.assertRaises(ValueError):
            step_environment(result.state, 0.0, 0.0)

    def test_08_exact_contact_is_also_a_collision(self):
        initial = LongitudinalState(1.0, 10.0, 0.0, 0.0)
        result = step_environment(initial, 0.0, 0.0)
        self.assert_step(result, (0.0, 10.0, 0.0, 0.0), True)
        with self.assertRaises(ValueError):
            step_environment(result.state, 0.0, 0.0)

    def test_09_nondefault_configuration_is_used(self):
        config = EnvironmentConfig(dt=0.2, tau_true=0.4, accel_min=-4.0, accel_max=1.0)
        initial = LongitudinalState(10.0, 5.0, 4.0, 0.0)
        result = step_environment(initial, 10.0, -2.0, config)
        self.assert_step(result, (9.8, 5.0, 3.6, 0.5))

    def test_10_invalid_step_inputs_fail_explicitly(self):
        initial = LongitudinalState(20.0, 10.0, 8.0, 0.0)
        for command, lead_accel in (
            (float("nan"), 0.0),
            (float("inf"), 0.0),
            (0.0, float("nan")),
            (0.0, -float("inf")),
        ):
            with self.subTest(command=command, lead_accel=lead_accel):
                with self.assertRaises(ValueError):
                    step_environment(initial, command, lead_accel)
        outside_bounds = LongitudinalState(20.0, 10.0, 8.0, 3.0)
        with self.assertRaises(ValueError):
            step_environment(outside_bounds, 0.0, 0.0)

    def test_11_invalid_configuration_fails_explicitly(self):
        invalid_configs = (
            {"dt": 0.0},
            {"dt": -0.1},
            {"tau_true": 0.0},
            {"tau_true": -0.5},
            {"dt": 0.9},
            {"dt": float("nan")},
            {"tau_true": float("inf")},
            {"accel_min": 1.0},
            {"accel_max": -1.0},
            {"accel_min": 0.0, "accel_max": 0.0},
            {"accel_min": -float("inf")},
        )
        for kwargs in invalid_configs:
            with self.subTest(config=kwargs):
                with self.assertRaises(ValueError):
                    EnvironmentConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
