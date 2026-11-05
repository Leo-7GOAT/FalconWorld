"""Produce a reference trajectory and a genuinely recursive nominal rollout.

No controller, model training, hardware commands or third-party dependencies.
Each nominal prediction becomes the next nominal input; it is NOT reset to
the reference state. The initial state and ego command sequence are shared.
"""

import argparse
import csv
from pathlib import Path

from falconworld.environment import EnvironmentConfig, step_environment
from falconworld.nominal import NominalConfig, predict_one_step
from falconworld.state import LongitudinalState


FIELDS = (
    "step", "time_s", "ego_cmd_for_interval", "lead_accel_env_for_interval",
    "env_gap", "env_v_ego", "env_v_lead", "env_a_actuator", "env_collision",
    "pred_gap", "pred_v_ego", "pred_v_lead", "pred_a_actuator",
    "pred_contact", "gap_error_pred_minus_env", "v_error_pred_minus_env",
)


def reference_lead_acceleration(step: int, scene: str) -> float:
    """Scenario data belongs to the environment side only.

    step is the zero-based interval index; braking starts at t=0.5 s
    with the fixed default dt=0.1 s. The nominal model never reads this.
    """
    if scene == "brake" and step >= 5:
        return -3.0
    return 0.0


def make_row(
    step: int,
    dt: float,
    ego_cmd: float | str,
    lead_accel: float | str,
    truth: LongitudinalState,
    predicted: LongitudinalState,
    collision: bool,
) -> dict:
    return dict(zip(FIELDS, (
        step, step * dt, ego_cmd, lead_accel,
        *truth.as_tuple(), collision,
        *predicted.as_tuple(), predicted.gap <= 0.0,
        predicted.gap - truth.gap,
        predicted.v_ego - truth.v_ego,
    )))


def run_episode(scene: str, steps: int, tau_true: float) -> tuple[list[dict], str]:
    if scene not in ("constant", "brake"):
        raise ValueError("scene must be constant or brake")
    if steps <= 0:
        raise ValueError("steps must be positive")

    nominal_config = NominalConfig()
    environment_config = EnvironmentConfig(
        dt=nominal_config.dt,
        tau_true=tau_true,
        accel_min=nominal_config.accel_min,
        accel_max=nominal_config.accel_max,
    )
    truth = LongitudinalState(20.0, 10.0, 8.0, 0.0)
    predicted = LongitudinalState(20.0, 10.0, 8.0, 0.0)
    rows = [make_row(0, nominal_config.dt, "", "", truth, predicted, False)]
    reason = "horizon_end"

    for interval in range(steps):
        ego_cmd = -2.0
        lead_accel = reference_lead_acceleration(interval, scene)

        # Predictor branch: only its own previous state, ego command and
        # nominal config. Do not pass the environment config or lead script.
        predicted = predict_one_step(predicted, ego_cmd, nominal_config)

        # Environment branch: true simulator parameters and current scenario
        # input. Its result becomes the next reference environment state.
        result = step_environment(truth, ego_cmd, lead_accel, environment_config)
        truth = result.state
        rows.append(make_row(
            interval + 1, nominal_config.dt, ego_cmd, lead_accel,
            truth, predicted, result.collision,
        ))
        if result.collision:
            reason = "collision"
            break

    return rows, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("constant", "brake"), default="constant")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--tau-true", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("outputs") / f"lesson02_{args.scene}.csv"

    try:
        rows, reason = run_episode(args.scene, args.steps, args.tau_true)
    except NotImplementedError as exc:
        print(f"WAITING: {exc}")
        print("Complete the environment exercise before generating a trajectory.")
        return 2
    except (ValueError, TypeError, AttributeError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("step   time    env_gap  pred_gap  env_v   pred_v   env_a   pred_a")
    for row in rows:
        if row["step"] <= 2 or row["step"] % 5 == 0 or row is rows[-1]:
            print(
                f'{row["step"]:4d} {row["time_s"]:6.2f} '
                f'{row["env_gap"]:9.4f} {row["pred_gap"]:9.4f} '
                f'{row["env_v_ego"]:7.4f} {row["pred_v_ego"]:7.4f} '
                f'{row["env_a_actuator"]:7.4f} {row["pred_a_actuator"]:7.4f}'
            )
    last = rows[-1]
    print(f'End: {reason}; simulated_steps={last["step"]}; rows={len(rows)}')
    print(f'Final gap error (pred-env): {last["gap_error_pred_minus_env"]:.6f} m')
    print(f'Final ego speed error (pred-env): {last["v_error_pred_minus_env"]:.6f} m/s')
    print(f"CSV: {output.resolve()}")
    print("Open-loop simulation only; this is not a closed-loop driving result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
