"""Legacy nominal predictor; this module must not query an environment."""

from dataclasses import dataclass
from math import isfinite

from .state import LongitudinalState


@dataclass(frozen=True)
class NominalConfig:
    """Teaching settings, not vehicle specifications or safety guarantees."""

    dt: float = 0.1
    tau: float = 0.5
    accel_min: float = -6.0
    accel_max: float = 2.0

    def __post_init__(self) -> None:
        values = (self.dt, self.tau, self.accel_min, self.accel_max)
        if not all(isfinite(value) for value in values):
            raise ValueError("Every configuration value must be finite.")
        if self.dt <= 0.0 or self.tau <= 0.0:
            raise ValueError("dt and tau must be positive.")
        # A conservative teaching constraint: the actuator update is a convex
        # combination of its old value and the bounded command.
        if self.dt > self.tau:
            raise ValueError("Lesson 01 requires dt <= tau.")
        if not (self.accel_min <= 0.0 <= self.accel_max):
            raise ValueError("The acceleration interval must include zero.")
        if self.accel_min >= self.accel_max:
            raise ValueError("accel_min must be less than accel_max.")


def predict_one_step(
    state: LongitudinalState,
    accel_cmd: float,
    config: NominalConfig = NominalConfig(),
) -> LongitudinalState:
    """Predict one explicit-Euler step using only OLD state values.

    The lead vehicle keeps a constant speed in this nominal model.
    accel_cmd is desired acceleration in m/s^2, NOT throttle or brake percent.
    Read docs/lesson01.md for the equations and one hand-calculated example.

    Requirements:
      * Bound the command using config.accel_min and config.accel_max.
      * Keep ego speed nonnegative; retain a negative gap for collision checks.
      * Return a NEW LongitudinalState; do not alter the input or read env data.
      * Use the old actuator response to update ego speed.
    """

    if not isfinite(accel_cmd):
        raise ValueError("accel_cmd must be finite.")
    if not config.accel_min <= state.a_actuator <= config.accel_max:
        raise ValueError("The initial actuator state is outside configured bounds.")
    bounded_cmd = max(
        config.accel_min,
        min(config.accel_max, accel_cmd),
    )

    gap_next = state.gap + (state.v_lead - state.v_ego) * config.dt
    v_ego_next = max(
        0.0,
        state.v_ego + state.a_actuator * config.dt,
    )
    v_lead_next = state.v_lead
    a_actuator_next = state.a_actuator + config.dt / config.tau *(bounded_cmd - state.a_actuator)
    return LongitudinalState(
        gap=gap_next,
        v_ego=v_ego_next,
        v_lead=v_lead_next,
        a_actuator=a_actuator_next,
    )

