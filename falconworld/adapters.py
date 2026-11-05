"""CARLA and R550 conversion boundaries. R550 is shadow-only in this package."""
from dataclasses import asdict, dataclass
from math import atan2, cos, hypot, isfinite, radians, sin
import json
from pathlib import Path
from .contracts import Action, Observation, SafetySupervisor


def carla_control(action, max_steer_rad, accel_scale=3.0, brake_scale=6.0):
    """Left-positive SI steering -> CARLA right-positive normalized steering.

    Acceleration conversion is a bounded feedforward approximation, calibrated
    again by the online model; it is not a universal vehicle inverse dynamics.
    """
    if not all(isfinite(v) and v > 0 for v in (max_steer_rad, accel_scale, brake_scale)):
        raise ValueError("CARLA conversion scales must be positive")
    return dict(throttle=max(0.0, min(1.0, action.acceleration / accel_scale)),
                brake=max(0.0, min(1.0, -action.acceleration / brake_scale)),
                steer=max(-1.0, min(1.0, -action.steering / max_steer_rad)))


class CarlaAdapter:
    def __init__(self, world, vehicle, dt=0.05):
        if not isfinite(dt) or dt <= 0 or dt > 0.1:
            raise ValueError("CARLA synchronous dt must be in (0,0.1]")
        self.world, self.vehicle, self.dt = world, vehicle, dt
        self.original_settings = world.get_settings()
        self.previous = None
        wheels = vehicle.get_physics_control().wheels
        self.max_steer_rad = radians(max(w.max_steer_angle for w in wheels))

    def __enter__(self):
        settings = self.world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, self.dt
        settings.substepping = True
        settings.max_substep_delta_time, settings.max_substeps = 0.01, 10
        self.world.apply_settings(settings)
        return self

    def __exit__(self, *args):
        self.world.apply_settings(self.original_settings)

    def observe(self):
        snapshot = self.world.get_snapshot()
        transform, velocity = self.vehicle.get_transform(), self.vehicle.get_velocity()
        speed = hypot(velocity.x, velocity.y)
        timestamp = snapshot.timestamp.elapsed_seconds
        acceleration = None
        if self.previous is not None and timestamp > self.previous.timestamp:
            acceleration = (speed - self.previous.speed) / (timestamp - self.previous.timestamp)
        # CARLA x-forward, y-right, z-up left-handed -> right-handed x,y-left.
        observation = Observation(timestamp, speed, transform.location.x, -transform.location.y,
                                  -radians(transform.rotation.yaw), measured_acceleration=acceleration,
                                  frame=snapshot.frame)
        self.previous = observation
        return observation

    def apply(self, action):
        import carla
        self.vehicle.apply_control(carla.VehicleControl(**carla_control(action, self.max_steer_rad)))

    def step(self, action):
        self.apply(action)
        self.world.tick(10.0)
        return self.observe()


@dataclass(frozen=True)
class R550Config:
    """Values must come from measured hardware/vendor evidence, never simulation."""
    wheelbase_m: float
    steering_limit_rad: float
    steering_rate_rad_s: float
    speed_limit_m_s: float
    acceleration_limit_m_s2: float
    deceleration_limit_m_s2: float
    command_timeout_s: float
    feedback_timeout_s: float
    speed_topic: str
    odometry_topic: str
    verified: bool = False
    evidence: str = ""

    def __post_init__(self):
        numbers = [v for v in asdict(self).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not all(isfinite(v) and v > 0 for v in numbers):
            raise ValueError("R550 dimensions and limits must be finite and positive")
        if not self.speed_topic or not self.odometry_topic:
            raise ValueError("R550 topics must be configured")
        if self.verified and not self.evidence.strip():
            raise ValueError("Verified configuration requires provenance")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class R550ShadowAdapter:
    """Produces reviewable bounded velocity/steering records; has no transport."""
    def __init__(self, config):
        self.config = config
        self.last_timestamp = None
        self.last_steering = 0.0
        self.supervisor = SafetySupervisor(max_age=config.feedback_timeout_s,
            max_cycle=config.command_timeout_s, accel_min=-config.deceleration_limit_m_s2,
            accel_max=config.acceleration_limit_m_s2, steering_max=config.steering_limit_rad)

    def shadow(self, observation, action, now, cycle_time=0.0, takeover=False):
        decision = self.supervisor.check(observation, action, now, cycle_time)
        dt = 0.0 if self.last_timestamp is None else observation.timestamp - self.last_timestamp
        if dt < 0:
            raise ValueError("Replay timestamp moved backwards")
        bounded_dt = min(dt, self.config.command_timeout_s)
        reason = "operator_takeover" if takeover else decision.reason
        if not reason and self.last_timestamp is not None and dt > self.config.feedback_timeout_s:
            reason = "feedback_gap"
        if reason:
            speed, steering = 0.0, 0.0
        else:
            speed = max(0.0, min(self.config.speed_limit_m_s,
                               observation.speed + decision.action.acceleration * bounded_dt))
            delta = self.config.steering_rate_rad_s * bounded_dt
            steering = max(self.last_steering - delta,
                           min(self.last_steering + delta, decision.action.steering))
        self.last_timestamp, self.last_steering = observation.timestamp, steering
        return dict(timestamp=observation.timestamp, target_speed_m_s=speed,
                    steering_rad=steering, fallback=bool(reason), reason=reason,
                    mode="shadow", sent=False, acknowledged=False,
                    configuration_verified=self.config.verified)
