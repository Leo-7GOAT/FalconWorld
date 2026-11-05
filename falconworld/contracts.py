"""Deployment-facing SI contracts; internal actuator state is never implied sensed."""
from dataclasses import dataclass
from math import isfinite
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class Observation:
    timestamp: float
    speed: float
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    gap: Optional[float] = None
    lead_speed: Optional[float] = None
    measured_acceleration: Optional[float] = None
    frame: Optional[int] = None

    def __post_init__(self):
        values = (self.timestamp, self.speed, self.x, self.y, self.yaw)
        optional = (self.gap, self.lead_speed, self.measured_acceleration)
        if not all(isfinite(v) for v in values + tuple(v for v in optional if v is not None)):
            raise ValueError("Observation must contain finite SI values")
        if self.timestamp < 0 or self.speed < 0 or (self.lead_speed is not None and self.lead_speed < 0):
            raise ValueError("Negative timestamp or forward speed")


@dataclass(frozen=True)
class Action:
    acceleration: float
    steering: float = 0.0

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.acceleration, self.steering)):
            raise ValueError("Action must be finite: acceleration m/s^2, left steering rad")


class History:
    def __init__(self, capacity=32):
        if capacity < 2:
            raise ValueError("History capacity must be >=2")
        self.capacity = capacity
        self.observations = []
        self.actions = []

    def append(self, observation, previous_action=None):
        if self.observations and observation.timestamp <= self.observations[-1].timestamp:
            raise ValueError("Observation timestamps must strictly increase")
        self.observations.append(observation)
        self.actions.append(previous_action)
        self.observations = self.observations[-self.capacity:]
        self.actions = self.actions[-self.capacity:]

    def acceleration(self):
        if len(self.observations) < 2:
            return 0.0
        a, b = self.observations[-2:]
        return (b.speed - a.speed) / (b.timestamp - a.timestamp)


class WorldModel(Protocol):
    def predict(self, history: History, actions: Sequence[Action]): ...


class Planner(Protocol):
    def plan(self, observation: Observation, history: History) -> Action: ...


class VehicleAdapter(Protocol):
    def observe(self) -> Observation: ...
    def apply(self, action: Action): ...


@dataclass(frozen=True)
class SafetyDecision:
    action: Action
    triggered: bool
    reason: str


class SafetySupervisor:
    """Explicit independent freshness, finite-cycle and stopping-distance checks."""
    def __init__(self, max_age=0.25, max_cycle=0.1, accel_min=-3.0, accel_max=1.0,
                 steering_max=0.4):
        if not all(isfinite(v) and v > 0 for v in
                   (max_age, max_cycle, -accel_min, accel_max, steering_max)):
            raise ValueError("Invalid safety limits")
        self.max_age, self.max_cycle = max_age, max_cycle
        self.accel_min, self.accel_max, self.steering_max = accel_min, accel_max, steering_max

    def check(self, observation, action, now, cycle_time=0.0):
        age = now - observation.timestamp
        reason = ""
        if not isfinite(age) or age < -1e-6 or age > self.max_age:
            reason = "stale_or_future_observation"
        elif not isfinite(cycle_time) or cycle_time < 0 or cycle_time > self.max_cycle:
            reason = "planner_timeout"
        elif observation.gap is not None and observation.gap <= 0.3 + observation.speed * self.max_age + observation.speed ** 2 / (-2 * self.accel_min):
            reason = "stopping_distance"
        if reason:
            return SafetyDecision(Action(self.accel_min, 0.0), True, reason)
        return SafetyDecision(Action(max(self.accel_min, min(self.accel_max, action.acceleration)),
                                     max(-self.steering_max, min(self.steering_max, action.steering))), False, "")
