"""Joint acceleration/steering PPO and vectorized one-step model branches.

Simulation only. Route geometry and actuator states are observable; hidden
response condition never enters observations or model prediction.
"""
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from .ackermann import AckermannConfig, bounded_action, step_ackermann, wrap_angle, clearance

CFG = AckermannConfig()
SCALE = np.array([30., 4., np.pi, 6., 3., .5, 3., .5, 1.3, np.pi, 6.])
OBS_SIZE = 14
MAX_STEPS = 160
REWARD_VERSION = 'tracking-v2-bounded-survival'
LEARNING_REWARD_SCALE = .01


@dataclass(frozen=True)
class TrackingRoute:
    amplitude: float = 1.3
    phase: float = 0.
    target_speed: float = 3.
    road_width: float = 7.
    goal: float = 30.
    obstacles: tuple = ()

    def road(self, x):
        return self.amplitude * (np.sin(np.asarray(x)/9+self.phase)-np.sin(self.phase))

    def heading(self, x):
        return np.arctan(self.amplitude/9*np.cos(np.asarray(x)/9+self.phase))


def encode_tracking(state, previous, route):
    s = np.asarray(state, dtype=float)
    raw = np.concatenate([s, np.broadcast_to(previous, (*s.shape[:-1], 2)),
        np.broadcast_to([route.amplitude, route.phase, route.target_speed], (*s.shape[:-1], 3))], axis=-1)
    extra = np.stack([(s[..., 1]-route.road(s[..., 0]))/2,
        wrap_angle(s[..., 2]-route.heading(s[..., 0]))/np.pi,
        (route.road(s[..., 0]+3)-s[..., 1])/3], axis=-1)
    return np.concatenate([raw/SCALE, extra], axis=-1).astype(np.float32)


def applied_tracking(action, previous):
    action = np.asarray(action, dtype=float)
    if action.shape[-1:] != (2,) or not np.isfinite(action).all():
        raise ValueError('Expected finite acceleration/steering action [...,2]')
    a = np.clip(action, -1, 1)
    requested = np.stack([2.5*a[..., 0]-.5, .5*a[..., 1]], axis=-1)
    return bounded_action(requested, np.clip(previous, [-3, -.5], [2, .5]), CFG)


def tracking_transition(state, next_state, previous, applied, route):
    """Same swept-footprint failure/reward for truth and synthetic transitions."""
    s, n = np.asarray(state), np.asarray(next_state)
    if s.shape != n.shape or s.shape[-1] != 6 or not np.isfinite(n).all():
        raise ValueError('Invalid predicted tracking state')
    # Adaptive common sweep, with endpoint and wrapped-yaw interpolation.
    distance = np.linalg.norm(n[..., :2]-s[..., :2], axis=-1)
    rotation = np.abs(wrap_angle(n[..., 2]-s[..., 2]))
    subdivisions = max(2, int(np.ceil(np.max(distance+rotation*CFG.length)/(.1*CFG.width))))
    alpha = np.linspace(0, 1, subdivisions+1)
    swept = s[..., None, :] + alpha[:, None]*(n-s)[..., None, :]
    swept[..., 2] = s[..., 2, None]+alpha*wrap_angle(n[..., 2]-s[..., 2])[..., None]
    road_margin, _ = clearance(swept, route, CFG)
    failure = np.any(road_margin <= 0, axis=-1) | (n[..., 0] < -2)
    success = (n[..., 0] >= route.goal) & ~failure
    ey = n[..., 1]-route.road(n[..., 0])
    ep = wrap_angle(n[..., 2]-route.heading(n[..., 0]))
    ev = n[..., 3]-route.target_speed
    cost = .5*ey**2+.15*ep**2+.2*ev**2
    cost += .005*np.sum(applied**2, axis=-1)+.01*np.sum((applied-previous)**2, axis=-1)
    reward = 1.-np.minimum(2., cost)+.1*np.clip(n[..., 0]-s[..., 0], -1., 1.)
    reward = np.where(failure, -20., reward+50*success)
    return reward, failure | success, dict(failure=failure, success=success,
        lateral_error_m=ey, heading_error_rad=ep, speed_error_mps=ev,
        road_margin_m=np.min(road_margin, axis=-1))


def model_tracking_batch(observations, actions, model):
    """Decode only public observations; vectorize roots sharing route geometry."""
    o = np.asarray(observations, dtype=float)
    raw = o[:, :11]*SCALE
    states, previous = raw[:, :6], raw[:, 6:8]
    # float32 round-trips may slightly exceed physical actuator limits.
    states[:, 4:6] = np.clip(states[:, 4:6], [-3, -.5], [2, .5])
    applied = applied_tracking(actions, previous)
    next_states = model.predict_batch(states, applied)
    future = np.empty_like(o, dtype=np.float32)
    rewards, terminal = np.empty(len(o)), np.empty(len(o), dtype=bool)
    # Each root carries its own reference, not an environment/scenario handle.
    for params in np.unique(raw[:, 8:11], axis=0):
        ids = np.all(raw[:, 8:11] == params, axis=1)
        route = TrackingRoute(*map(float, params))
        future[ids] = encode_tracking(next_states[ids], applied[ids], route)
        rewards[ids], terminal[ids], _ = tracking_transition(states[ids], next_states[ids], previous[ids], applied[ids], route)
    return future, rewards, terminal


class AckermannTrackingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, training_seed=0, namespace=10_000_000):
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Box(-1., 1., (2,), np.float32)
        self.training_seed, self.namespace = training_seed, namespace
        self.episode = -1
        self.transition_count = 0
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = 0 if seed is not None else self.episode+1
        options = options or {}
        episode_seed = options.get('episode_seed', self.namespace+self.training_seed*100_000+self.episode)
        rng = np.random.default_rng(episode_seed)
        self.family = options.get('family', ('straight', 'curve')[self.episode % 2])
        if self.family not in ('straight', 'curve'):
            raise ValueError('Unknown route family')
        self.condition = float(options.get('condition', 1.))
        ood = self.condition != 1.
        amplitude = (1.8 if ood else 1.3) if self.family == 'curve' else 0.
        self.route = TrackingRoute(amplitude, float(rng.uniform(-.8, .8)), float(rng.uniform(4, 5) if ood else rng.uniform(2, 4)))
        self.state = np.array([0., rng.uniform(-.3, .3), self.route.heading(0)+rng.uniform(-.08, .08),
                               self.route.target_speed+rng.uniform(-.2, .2), 0., 0.])
        self.previous = np.zeros(2)
        self.steps = 0
        self.done = False
        self.case = dict(episode_seed=int(episode_seed), family=self.family, condition=self.condition)
        return encode_tracking(self.state, self.previous, self.route), dict(self.case)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after termination/truncation')
        old, previous = self.state.copy(), self.previous.copy()
        u = applied_tracking(action, previous)
        n = step_ackermann(old, u, CFG, condition=self.condition)
        reward, terminal, metrics = tracking_transition(old, n, previous, u, self.route)
        self.steps += 1
        self.transition_count += 1
        truncated = bool(self.steps >= MAX_STEPS and not terminal)
        self.state, self.previous = n, u
        self.done = bool(terminal or truncated)
        info = {k: float(v) if k not in ('failure', 'success') else bool(v) for k, v in metrics.items()}
        info.update(state=old, next_state=n.copy(), previous=previous, applied=u.copy(),
                    raw_reward=float(reward), steps=self.steps, **self.case)
        return encode_tracking(n, u, self.route), float(reward)*LEARNING_REWARD_SCALE, bool(terminal), truncated, info


class AckermannWorldModelPPO(PPO):
    """Continuous-action one-step replacement with frozen-policy TD targets.

    All roots sampled before updates; raw Gaussian actions/logprobs retained,
    clipped actions used for transition exactly as SB3 does in the real env.
    """
    def __init__(self, *args, world_model=None, model_fraction=.5, **kwargs):
        if not 0 <= model_fraction <= 1:
            raise ValueError('Invalid model fraction')
        self.world_model, self.model_fraction = world_model, model_fraction
        self.model_generated = self.model_replaced = self.optimizer_steps = 0
        super().__init__(*args, **kwargs)
        if self.observation_space.shape != (OBS_SIZE,) or self.action_space.shape != (2,):
            raise ValueError('Expected Ackermann tracking observation and continuous two-action contract')
        if self.device.type != 'cpu' or self.target_kl is not None:
            raise ValueError('Verified CPU fixed-update path requires target_kl=None')
        self.branch_rng = np.random.default_rng((self.seed or 0)+72_001)
        self.branch_torch_state = torch.Generator().manual_seed((self.seed or 0)+72_002).get_state()
        self.last_branch = {}

    def train(self):
        buf = self.rollout_buffer
        count = int(round(buf.buffer_size*buf.n_envs*self.model_fraction)) if self.world_model is not None else 0
        if count:
            if not buf.full or buf.generator_ready:
                raise ValueError('Expected fresh unflattened rollout')
            size = buf.buffer_size*buf.n_envs
            pool = buf.observations.reshape(size, OBS_SIZE).copy()
            roots = pool[self.branch_rng.integers(size, size=count)].copy()
            self.policy.set_training_mode(False)
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.set_rng_state(self.branch_torch_state)
                tensor, _ = self.policy.obs_to_tensor(roots)
                actions, values, log_probs = self.policy(tensor)
                self.branch_torch_state = torch.get_rng_state()
                raw_actions = actions.cpu().numpy()
                future, rewards, terminal = model_tracking_batch(roots, raw_actions, self.world_model)
                next_tensor, _ = self.policy.obs_to_tensor(future)
                next_values = self.policy.predict_values(next_tensor).cpu().numpy().reshape(-1)
                values_np = values.cpu().numpy().reshape(-1)
                returns = rewards*LEARNING_REWARD_SCALE+self.gamma*(~terminal)*next_values
                indices = self.branch_rng.choice(size, count, replace=False)
                changes = dict(observations=roots, actions=raw_actions, values=values_np,
                    log_probs=log_probs.cpu().numpy(), advantages=returns-values_np, returns=returns)
                for field, value in changes.items():
                    arr = getattr(buf, field)
                    arr.reshape(size, *arr.shape[2:])[indices] = value
                check_v, check_log, _ = self.policy.evaluate_actions(tensor, actions)
                error = float(torch.max(torch.abs(check_log-log_probs)))
                if error > 1e-5:
                    raise AssertionError('Behavior probability mismatch')
                self.last_branch = dict(roots=count, fraction=self.model_fraction, horizon=1,
                    max_logprob_error=error, terminal_count=int(terminal.sum()),
                    max_value_error=float(torch.max(torch.abs(check_v-values))))
            self.model_generated += count
            self.model_replaced += count
        before = max((int(v['step'].item()) for v in self.policy.optimizer.state.values() if 'step' in v), default=0)
        super().train()
        after = max((int(v['step'].item()) for v in self.policy.optimizer.state.values() if 'step' in v), default=0)
        self.optimizer_steps += after-before

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['world_model', 'branch_rng', 'branch_torch_state']
