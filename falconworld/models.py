"""Compact trainable world models and train-only calibrated physical baselines.

No model receives scenario identity, future leader inputs or hidden condition.
The leader head deliberately excludes ego state/actions: leader behavior is exogenous.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import least_squares
from .data import DT, HISTORY


def estimate_sensor_state(observation, history=None, previous_action=None, physics=None):
    """Causal common estimator, ignoring any privileged incoming actuator column.

    observation/history may have 3 or 4 columns; only gap and two speeds are read.
    Speed differences identify previous interval acceleration when moving. Optional
    known last command and calibrated dynamics propagate that estimate to now.
    At standstill latent braking is unobservable: returned value is an estimate.
    No environment condition, future frame or true actuator is used.
    """
    obs = np.asarray(observation, dtype=float)
    single = obs.ndim == 1
    s = np.atleast_2d(obs)
    if s.shape[1] not in (3, 4) or not np.isfinite(s[:, :3]).all():
        raise ValueError("Expected finite gap/ego/leader observation")
    out = np.column_stack([s[:, :3], np.zeros(len(s))])
    if history is not None:
        h = np.asarray(history, dtype=float)
        if h.ndim == 2:
            h = h[None]
        if h.shape[-1] not in (3, 4) or h.shape[-2] < 2 or not np.isfinite(h[..., :3]).all():
            raise ValueError("Expected causal speed history")
        h = np.broadcast_to(h, (len(s), h.shape[1], h.shape[2]))
        out[:, 3] = np.clip((h[:, -1, 1]-h[:, -2, 1])/DT, -6, 2)
    if previous_action is not None and physics is not None:
        _, u, _ = _batch(out, previous_action)
        out[:, 3] = np.clip(physics.actuator(out, u), -6, 2)
    return out[0] if single else out


def _batch(states, actions):
    s = np.asarray(states, dtype=float)
    single = s.ndim == 1
    s = np.atleast_2d(s)
    u = np.broadcast_to(np.asarray(actions, dtype=float).reshape(-1), (len(s),))
    if s.shape[1] != 4 or not np.isfinite(s).all() or not np.isfinite(u).all():
        raise ValueError("States/actions must be finite with state dimension four")
    return s, np.clip(u, -6, 2), single


def _history(s, history):
    if history is None:
        return np.repeat(s[:, None], HISTORY, axis=1)
    h = np.asarray(history, dtype=float)
    if h.ndim == 2:
        h = h[None]
    if h.shape[-1] != 4 or h.shape[-2] < 2 or not np.isfinite(h).all():
        raise ValueError("Invalid history")
    h = np.broadcast_to(h, (len(s), h.shape[1], 4))
    # Fixed feature dimension, with causal repeat padding for startup.
    if h.shape[1] < HISTORY:
        h = np.concatenate([np.repeat(h[:, :1], HISTORY-h.shape[1], axis=1), h], axis=1)
    return h[:, -HISTORY:]


def _kinematic(s, next_actuator, leader_delta):
    out = s.copy()
    out[:, 0] = s[:, 0] + DT * (s[:, 2] - s[:, 1])
    out[:, 1] = np.maximum(0, s[:, 1] + DT * s[:, 3])
    out[:, 2] = np.maximum(0, s[:, 2] + leader_delta)
    out[:, 3] = np.clip(next_actuator, -6, 2)
    return out


class PhysicsModel:
    def __init__(self, name="physical_cv", tau=.8, enriched=None, lead="cv"):
        self.name, self.tau, self.enriched, self.lead = name, float(tau), enriched, lead

    def actuator(self, s, u):
        if self.enriched is None:
            tau = self.tau
        else:
            accel, brake, speed = self.enriched
            tau = np.where(u >= s[:, 3], accel, brake) * (1 + speed * s[:, 1])
        return s[:, 3] + DT * (u - s[:, 3]) / np.maximum(DT, tau)

    def step(self, states, actions, history=None):
        s, u, single = _batch(states, actions)
        delta = np.zeros(len(s))
        if self.lead == "ca":
            h = _history(s, history)
            delta = np.clip(h[:, -1, 2] - h[:, -2, 2], -.6, .2)
        out = _kinematic(s, self.actuator(s, u), delta)
        return out[0] if single else out

    def specification(self):
        return {"kind": "physics", "name": self.name, "tau": self.tau,
                "enriched": self.enriched, "lead": self.lead}


class MLP:
    """One hidden tanh layer, analytic backpropagation, Adam and validation selection."""
    def __init__(self):
        self.weights = None

    def fit(self, x, y, xv, yv, seed=0, epochs=100):
        r = np.random.default_rng(seed)
        self.xmean, self.xscale = x.mean(0), np.maximum(x.std(0), .05)
        self.ymean, self.yscale = y.mean(0), np.maximum(y.std(0), .001)
        x, y = (x-self.xmean)/self.xscale, (y-self.ymean)/self.yscale
        xv, yv = (xv-self.xmean)/self.xscale, (yv-self.ymean)/self.yscale
        w = [r.normal(0, 1/np.sqrt(x.shape[1]), (x.shape[1], 32)), np.zeros(32),
             r.normal(0, .1, (32, y.shape[1])), np.zeros(y.shape[1])]
        m, v = [np.zeros_like(a) for a in w], [np.zeros_like(a) for a in w]
        best, bestw, tick, stale = np.inf, None, 0, 0
        self.log = []
        # Exact-zero residuals need no arbitrary learned perturbation.
        if np.max(np.abs(y)) < 1e-8:
            w[2][:] = 0; w[3][:] = 0
            self.weights = w
            self.log = [{"epoch": 0, "validation_mse_normalized": 0.}]
            return self
        for epoch in range(epochs):
            for index in np.array_split(r.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
                xb, yb = x[index], y[index]
                h = np.tanh(xb @ w[0] + w[1])
                e = 2 * (h @ w[2] + w[3] - yb) / len(index)
                dh = (e @ w[2].T) * (1-h*h)
                g = [xb.T @ dh + 1e-5*w[0], dh.sum(0), h.T @ e + 1e-5*w[2], e.sum(0)]
                tick += 1
                for j in range(4):
                    m[j] = .9*m[j] + .1*g[j]
                    v[j] = .999*v[j] + .001*g[j]**2
                    w[j] -= .002 * (m[j]/(1-.9**tick)) / (np.sqrt(v[j]/(1-.999**tick)) + 1e-8)
            if epoch % 5 == 0 or epoch == epochs-1:
                loss = float(np.mean((np.tanh(xv @ w[0]+w[1]) @ w[2]+w[3]-yv)**2))
                self.log.append({"epoch": epoch, "validation_mse_normalized": loss})
                if loss < best - 1e-7:
                    best, bestw, stale = loss, [a.copy() for a in w], 0
                else:
                    stale += 1
                if stale >= 5:
                    break
        self.weights = bestw
        return self

    def __call__(self, x):
        x = (x-self.xmean)/self.xscale
        w = self.weights
        return (np.tanh(x @ w[0]+w[1]) @ w[2]+w[3])*self.yscale+self.ymean

    def arrays(self, prefix):
        result = {prefix+k: getattr(self, k) for k in ("xmean", "xscale", "ymean", "yscale")}
        result.update({prefix+f"w{i}": a for i, a in enumerate(self.weights)})
        return result

    @classmethod
    def restore(cls, arrays, prefix):
        net = cls()
        for key in ("xmean", "xscale", "ymean", "yscale"):
            setattr(net, key, arrays[prefix+key])
        net.weights = [arrays[prefix+f"w{i}"] for i in range(4)]
        net.log = []
        return net


def _actuator_features(s, u):
    return np.column_stack([s[:, 1], s[:, 3], u, u-s[:, 3]])


def _leader_features(s, history, with_history):
    if not with_history:
        return s[:, 2:3]
    h = _history(s, history)
    # No ego quantities or commands may enter this exogenous leader head.
    return np.column_stack([s[:, 2], np.diff(h[:, :, 2], axis=1)])


class LearnedModel:
    def __init__(self, name, physics, actuator_net, leader_net, residual=True, with_history=False, lead_only=False):
        self.name, self.physics = name, physics
        self.actuator_net, self.leader_net = actuator_net, leader_net
        self.residual, self.with_history, self.lead_only = residual, with_history, lead_only

    def step(self, states, actions, history=None):
        s, u, single = _batch(states, actions)
        a = self.physics.actuator(s, u) if (self.residual or self.lead_only) else np.zeros(len(s))
        if not self.lead_only:
            a = a + self.actuator_net(_actuator_features(s, u))[:, 0]
        dv = self.leader_net(_leader_features(s, history, self.with_history))[:, 0]
        out = _kinematic(s, a, np.clip(dv, -.6, .2))
        return out[0] if single else out

    def specification(self):
        return {"kind": "learned", "name": self.name, "physics": self.physics.specification(),
                "residual": self.residual, "with_history": self.with_history, "lead_only": self.lead_only}


class EnsembleModel:
    """Empirical disagreement; validation radius is one-step, not a multi-step guarantee."""
    def __init__(self, members, radius=None):
        self.name, self.members = "ensemble", members
        self.radius = np.zeros(4) if radius is None else np.asarray(radius)

    def step(self, states, actions, history=None):
        return np.mean([m.step(states, actions, history) for m in self.members], axis=0)

    def uncertainty(self, states, actions, history=None):
        predictions = np.asarray([m.step(states, actions, history) for m in self.members])
        return np.std(predictions, axis=0)

    def interval_radius(self, states, actions, history=None):
        return self.radius + 1.645*self.uncertainty(states, actions, history)


def _fit_physics(train):
    s, u, _ = _batch(train["state"], train["action"])
    da = train["next_state"][:, 3]-s[:, 3]
    x = DT*(u-s[:, 3])
    tau = np.clip(float(x@x / max(x@da, 1e-12)), DT, 5.)
    def error(p):
        t = np.where(u >= s[:, 3], p[0], p[1]) * (1+p[2]*s[:, 1])
        return da - x/t
    params = least_squares(error, [.5, .8, .03], bounds=([.15,.15,0], [2,2,.2])).x.tolist()
    return tau, params


def _fit_learned(train, val, name, physics, seed, epochs, residual=True, with_history=False, lead_only=False):
    s, u, _ = _batch(train["state"], train["action"])
    sv, uv, _ = _batch(val["state"], val["action"])
    target, targetv = train["next_state"][:, 3:4].copy(), val["next_state"][:, 3:4].copy()
    if residual:
        target -= physics.actuator(s, u)[:, None]
        targetv -= physics.actuator(sv, uv)[:, None]
    an = MLP().fit(_actuator_features(s, u), target, _actuator_features(sv, uv), targetv, seed, epochs)
    ln = MLP().fit(_leader_features(s, train["history"], with_history),
                   train["next_state"][:, 2:3]-s[:, 2:3],
                   _leader_features(sv, val["history"], with_history),
                   val["next_state"][:, 2:3]-sv[:, 2:3], seed+102, epochs)
    return LearnedModel(name, physics, an, ln, residual, with_history, lead_only)


def save_model(model, path):
    path = Path(path)
    if isinstance(model, EnsembleModel):
        spec = {"kind": "ensemble", "members": [m.specification() for m in model.members]}
        arrays = {"radius": model.radius}
        for i, member in enumerate(model.members):
            arrays.update(member.actuator_net.arrays(f"m{i}_a_"))
            arrays.update(member.leader_net.arrays(f"m{i}_l_"))
    else:
        spec, arrays = model.specification(), {}
        if isinstance(model, LearnedModel):
            arrays.update(model.actuator_net.arrays("a_"))
            arrays.update(model.leader_net.arrays("l_"))
    arrays["specification"] = np.array(json.dumps(spec))
    np.savez_compressed(path, **arrays)


def _restore(spec, arrays, prefix=""):
    if spec["kind"] == "physics":
        return PhysicsModel(spec["name"], spec["tau"], spec["enriched"], spec["lead"])
    return LearnedModel(spec["name"], _restore(spec["physics"], arrays),
                        MLP.restore(arrays, prefix+"a_"), MLP.restore(arrays, prefix+"l_"),
                        spec["residual"], spec["with_history"], spec["lead_only"])


def load_model(path):
    with np.load(path, allow_pickle=False) as a:
        spec = json.loads(str(a["specification"]))
        if spec["kind"] == "ensemble":
            return EnsembleModel([_restore(s, a, f"m{i}_") for i, s in enumerate(spec["members"])], a["radius"])
        return _restore(spec, a)


def load_models(path):
    return {p.stem: load_model(p) for p in sorted(Path(path).glob("*.npz"))}


def train_models(data, output_dir, seed=0, preset="smoke"):
    start = time.perf_counter()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val = data["train"], data["val"]
    tau, params = _fit_physics(train)
    enriched = PhysicsModel("physical_enriched", tau, params, "ca")
    models = {"physical_cv": PhysicsModel("physical_cv", tau),
              "physical_ca": PhysicsModel("physical_ca", tau, lead="ca"),
              "physical_enriched": enriched}
    epochs = {"smoke": 80, "standard": 140, "full": 220}[preset]
    modes = [("neural", False, False, False), ("residual", True, False, False),
             ("history", True, True, False), ("learned_lead", True, True, True)]
    for name, residual, history, lead_only in modes:
        models[name] = _fit_learned(train, val, name, enriched, seed, epochs, residual, history, lead_only)
    members = [models["history"]]
    for i in range(2):
        members.append(_fit_learned(train, val, f"history_{i+1}", enriched, seed+31*(i+1), epochs, True, True))
    ensemble = EnsembleModel(members)
    val_residual = np.abs(ensemble.step(val["state"], val["action"], val["history"]) - val["next_state"])
    ensemble.radius = np.quantile(val_residual, .9, axis=0)
    models["ensemble"] = ensemble
    for name, model in models.items():
        save_model(model, out / f"{name}.npz")
    log = {"seed": seed, "preset": preset, "data_sha256": data["metadata"]["sha256"],
           "tau_calibrated": tau, "enriched_parameters": params,
           "training_seconds": time.perf_counter()-start, "normalization_fit_split": "train",
           "selection_split": "val", "history_length": HISTORY,
           "architecture": "separate 4x32x1 actuator and 1/6x32x1 exogenous leader tanh MLPs",
           "models": {name: {"actuator": m.actuator_net.log, "leader": m.leader_net.log}
                      for name, m in models.items() if isinstance(m, LearnedModel)},
           "ensemble_one_step_validation_radius": ensemble.radius.tolist()}
    (out / "training.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return models
