"""Shared state contract for lesson 01. All quantities use SI units."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LongitudinalState:
    """One-dimensional following state.

    gap: bumper-to-bumper clearance [m]; zero/negative means contact/overlap.
    v_ego, v_lead: forward speeds [m/s], both nonnegative.
    a_actuator: ego actuator response [m/s^2], BEFORE the no-reverse constraint.
        At standstill this can stay negative while the speed remains zero.
        It is not always equal to measured (v_next - v_current) / dt.
    """

    gap: float
    v_ego: float
    v_lead: float
    a_actuator: float

    def __post_init__(self) -> None:
        values = (self.gap, self.v_ego, self.v_lead, self.a_actuator)
        if not all(isfinite(value) for value in values):
            raise ValueError("Every state component must be finite.")
        if self.v_ego < 0.0 or self.v_lead < 0.0:
            raise ValueError("Forward speeds must be nonnegative.")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (gap, v_ego, v_lead, a_actuator), in this order.

        Use a tuple, preserve the values, and do not modify this object.
        A tuple keeps the first lesson independent of NumPy/PyTorch.
        """
        gap = self.gap
        v_ego = self.v_ego
        v_lead = self.v_lead
        a_actuator = self.a_actuator
        return (gap, v_ego, v_lead, a_actuator)
