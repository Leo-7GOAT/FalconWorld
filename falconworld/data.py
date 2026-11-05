"""Episode-disjoint synthetic longitudinal data. All numbers are simulation assumptions.

The scenario script and condition belong to the environment, never the predictor.
Public prediction state is [clearance m, ego m/s, leader m/s, actuator m/s²].
Actuator is privileged simulation state; sensor experiments must estimate it.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

DT = 0.1
HISTORY = 6
FAMILIES = ("constant", "mild", "brake", "stopgo")
DATA_VERSION = "longitudinal-v1"


def make_episode(family, seed, condition="id"):
    if family not in FAMILIES or condition not in ("id", "ood", "linear"):
        raise ValueError("Unknown scenario family or condition")
    r = np.random.default_rng(seed)
    speed = r.uniform(6, 13) if condition != "ood" else r.uniform(14, 19)
    return {"state": np.array([r.uniform(22, 48), max(0., speed + r.uniform(-2, 2)), speed, 0.]),
            "params": {"onset": float(r.uniform(1.5, 3.5)), "braking": float(r.uniform(3.5, 5.5)),
                       "mild": float(r.uniform(.5, 1.4)), "restart": float(r.uniform(6.5, 8.5)),
                       "phase": float(r.uniform(0, 2 * np.pi))},
            "family": family, "condition": condition, "seed": int(seed)}


def scenario_lead_accel(family, t, params):
    """Environment-only exogenous acceleration; no response to ego action."""
    if family == "constant":
        return 0.
    onset = params["onset"]
    if family == "mild":
        # Smooth, observable acceleration trend with unknown phase.
        return float(-params["mild"] * np.sin(.65 * t + params["phase"]))
    if family == "brake":
        return -params["braking"] if onset <= t < onset + 3. else 0.
    if family == "stopgo":
        if onset <= t < params["restart"]:
            return -2.5
        return 1.5 if params["restart"] <= t < params["restart"] + 4 else 0.
    raise ValueError("Unknown family")


def step_truth(state, action, lead_accel, condition="id"):
    """Independent explicit Euler dynamics. Negative gap retained; contact terminal.

    ID: asymmetric lag tau=(0.45 acceleration / 0.75 braking)*(1+0.045*v).
    OOD: both lags multiplied by 1.45; higher initial speeds. No measured-car claim.
    Linear: fixed tau=.8 to verify parameter-identification sufficiency.
    """
    s = np.asarray(state, dtype=float)
    if s.shape[-1] != 4 or not np.isfinite(s).all() or not np.isfinite(action).all() or not np.isfinite(lead_accel).all():
        raise ValueError("Invalid state/action")
    if np.any(s[..., 0] <= 0) or np.any(s[..., 1:3] < 0):
        raise ValueError("Terminal/invalid initial state")
    u = np.clip(action, -6, 2)
    if condition == "linear":
        tau = .8
    elif condition in ("id", "ood"):
        tau = np.where(u >= s[..., 3], .45, .75) * (1 + .045 * s[..., 1])
        tau = tau * (1.45 if condition == "ood" else 1.)
    else:
        raise ValueError("Unknown condition")
    n = s.copy()
    n[..., 0] = s[..., 0] + DT * (s[..., 2] - s[..., 1])
    n[..., 1] = np.maximum(0, s[..., 1] + DT * s[..., 3])
    n[..., 2] = np.maximum(0, s[..., 2] + DT * lead_accel)
    n[..., 3] = s[..., 3] + DT * (u - s[..., 3]) / tau
    return n


def _roll_episode(ep, episode_id, steps):
    r = np.random.default_rng(ep["seed"] + 23123)
    state = ep["state"].copy()
    hist = np.repeat(state[None, :], HISTORY, axis=0)
    rows = []
    command = 0.
    for k in range(steps):
        if k % 8 == 0:
            command = float(r.uniform(-5, 2))
        # Broad actuator excitation plus feedback prevents collecting only crashes.
        feedback = .5 * (state[2] - state[1]) + .08 * (state[0] - 15 - 1.2 * state[1])
        u = np.clip(.55 * command + .45 * feedback, -6, 2)
        nxt = step_truth(state, u, scenario_lead_accel(ep["family"], k * DT, ep["params"]), ep["condition"])
        rows.append((state.copy(), u, nxt.copy(), hist.copy(), episode_id, ep["family"], k))
        if nxt[0] <= 0:
            break
        state = nxt
        hist = np.concatenate([hist[1:], state[None]])
    return rows


def generate_dataset(output_dir, preset="smoke"):
    """Generate once per preset, with fixed episode IDs and a content fingerprint."""
    if preset not in ("smoke", "standard", "full"):
        raise ValueError("Unknown preset")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = {"smoke": (12, 4, 4, 4), "standard": (48, 12, 12, 12), "full": (120, 30, 30, 30)}[preset]
    data, manifest = {}, []
    fields = ("state", "action", "next_state", "history", "episode", "family", "step")
    for split_index, (split, per_family) in enumerate(zip(("train", "val", "test", "ood"), count)):
        rows = []
        for fi, family in enumerate(FAMILIES):
            for i in range(per_family):
                episode_id = split_index * 100000 + fi * 1000 + i
                ep = make_episode(family, 1000000 + episode_id, "ood" if split == "ood" else "id")
                episode_rows = _roll_episode(ep, episode_id, 120)
                rows.extend(episode_rows)
                manifest.append({"split": split, "episode_id": episode_id, "seed": ep["seed"],
                                 "family": family, "condition": ep["condition"], "steps": len(episode_rows),
                                 "initial_state": ep["state"].tolist(), "environment_params": ep["params"]})
        data[split] = {field: np.asarray([row[j] for row in rows]) for j, field in enumerate(fields)}
        np.savez_compressed(out / f"{split}.npz", **data[split])
    digest = hashlib.sha256()
    for split in data:
        for field in ("state", "action", "next_state", "episode"):
            digest.update(data[split][field].tobytes())
    metadata = {"version": DATA_VERSION, "preset": preset, "dt": DT, "history": HISTORY,
                "sha256": digest.hexdigest(), "episodes": manifest,
                "state_access": "privileged actuator; sensor variant requires causal estimator",
                "test_policy": "episode split frozen before training; normalization/calibration train only"}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    data["metadata"] = metadata
    return data


def load_dataset(output_dir):
    out = Path(output_dir)
    data = {s: dict(np.load(out / f"{s}.npz", allow_pickle=False)) for s in ("train", "val", "test", "ood")}
    data["metadata"] = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return data
