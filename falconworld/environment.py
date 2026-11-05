"""Lesson 02: reference simulator dynamics, separate from the predictor.

This is a small teaching simulator, not measured vehicle ground truth.
The caller owns the episode, clock and lead-car scenario. The predictor must
not receive EnvironmentConfig or the future lead_accel scenario.
"""

from dataclasses import dataclass
from math import isfinite

from .state import LongitudinalState


@dataclass(frozen=True)
class EnvironmentConfig:
    """Simulator parameters; tau_true is a teaching choice, not a calibration."""

    dt: float = 0.1
    tau_true: float = 0.8
    accel_min: float = -6.0
    accel_max: float = 2.0

    def __post_init__(self) -> None:
        values = (self.dt, self.tau_true, self.accel_min, self.accel_max)
        if not all(isfinite(value) for value in values):
            raise ValueError("Every environment parameter must be finite.")
        if self.dt <= 0.0 or self.tau_true <= 0.0:
            raise ValueError("dt and tau_true must be positive.")
        if self.dt > self.tau_true:
            raise ValueError("Lesson 02 requires dt <= tau_true.")
        if not (self.accel_min <= 0.0 <= self.accel_max):
            raise ValueError("The command interval must include zero.")
        if self.accel_min >= self.accel_max:
            raise ValueError("accel_min must be less than accel_max.")


@dataclass(frozen=True)
class StepResult:
    state: LongitudinalState
    collision: bool


def step_environment(
    state: LongitudinalState,
    accel_cmd: float,
    lead_accel: float,
    config: EnvironmentConfig = EnvironmentConfig(),
) -> StepResult:
    """Advance the reference environment by one explicit-Euler step.

    accel_cmd: ego desired acceleration [m/s^2], clipped using config bounds.
    lead_accel: prescribed lead-car acceleration [m/s^2] for this interval.
                This is an environment input, not predictor information.

    Exercise:
      1. Bound accel_cmd and compute the four next-state values from OLD state.
      2. Update lead speed with lead_accel; both speeds stay nonnegative.
      3. Use config.tau_true for the ego actuator update.
      4. Return StepResult(new_state, collision=(gap_next <= 0.0)).

    Preserve negative gap. Do not mutate state or call predict_one_step.
    A collision result is terminal; no later environment step is permitted.
    See docs/lesson02.md for the equations and independent hand checks.
    """
    if not isfinite(accel_cmd) or not isfinite(lead_accel):
        raise ValueError("Both acceleration inputs must be finite.")
    if state.gap <= 0.0:
        raise ValueError("The episode has already reached contact/collision.")
    if not config.accel_min <= state.a_actuator <= config.accel_max:
        raise ValueError("The initial actuator state is outside configured bounds.")

    # Write the environment update below. Keep the validation above.
    bounded_cmd = max(
        config.accel_min,
        min(config.accel_max, accel_cmd),
    )

    gap_next = state.gap + (state.v_lead - state.v_ego) * config.dt
    v_ego_next = max(
        0.0,
        state.v_ego + state.a_actuator * config.dt,
    )
    v_lead_next = max(0, state.v_lead + lead_accel * config.dt)
    a_actuator_next = state.a_actuator + config.dt / config.tau_true * (bounded_cmd - state.a_actuator)
    return StepResult(
        state=LongitudinalState(
        gap=gap_next,
        v_ego=v_ego_next,
        v_lead=v_lead_next,
        a_actuator=a_actuator_next,
    ),
        collision=(gap_next <= 0.0),
    )

