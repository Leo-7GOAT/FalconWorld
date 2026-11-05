"""NumPy-only longitudinal RL contract; scripts are private to real stepping.

Privileged simulator observation: six [gap, ego speed, lead speed, actuator]
frames plus the last applied acceleration command. SI units throughout.
"""
from __future__ import annotations

import numpy as np
from .data import DT, HISTORY, FAMILIES, make_episode, scenario_lead_accel, step_truth
from .planning import SafetySupervisor

ACTION_DELTAS = np.array([-.8, -.4, 0., .4, .8])
ACTION_DELTAS.setflags(write=False)
STATE_SCALE = np.array([50., 20., 20., 6.])
STATE_SCALE.setflags(write=False)
OBSERVATION_SIZE = HISTORY * 4 + 1
ACTION_COUNT = len(ACTION_DELTAS)
MAX_STEPS = 120
REWARD_VERSION = 'longitudinal-rl-reward-v2-bounded-survival'


def _history_array(history):
    h = np.asarray(history, dtype=float)
    if h.shape[-2:] != (HISTORY, 4) or not np.isfinite(h).all():
        raise ValueError('Expected finite [...,6,4] raw history')
    if np.any(h[..., 1:3] < 0):
        raise ValueError('Speeds must be nonnegative')
    return h


def _previous(previous_u):
    p = np.asarray(previous_u, dtype=float)
    if not np.isfinite(p).all() or np.any(p < -6.) or np.any(p > 2.):
        raise ValueError('Previous applied command must be finite in [-6,2] m/s^2')
    return p


def observation(history, previous_u):
    """Raw 25-vector, with arbitrary leading batch dimensions supported."""
    h = _history_array(history)
    p = np.broadcast_to(_previous(previous_u), h.shape[:-2])
    return np.concatenate((h.reshape(*h.shape[:-2], HISTORY * 4), p[..., None]), axis=-1)


def decode_observation(raw_observation):
    o = np.asarray(raw_observation, dtype=float)
    if o.shape[-1:] != (OBSERVATION_SIZE,) or not np.isfinite(o).all():
        raise ValueError('Expected finite raw [...,25] observation')
    h = o[..., :-1].reshape(*o.shape[:-1], HISTORY, 4).copy()
    _history_array(h)
    p = _previous(o[..., -1]).copy()
    return h, float(p) if p.ndim == 0 else p


def encode(history, previous_u):
    """Fixed physical scales, no fitted statistics and no clipping (incl. OOD)."""
    h = _history_array(history)
    p = np.broadcast_to(_previous(previous_u), h.shape[:-2])
    return np.concatenate(((h / STATE_SCALE).reshape(*h.shape[:-2], HISTORY * 4),
                           p[..., None] / 6.), axis=-1).astype(np.float32)


def reward_components(current, next_state, previous_u, applied_u):
    """Bound only state errors; contact costs 100 even for shallow penetration.

    This shared function accepts single states or broadcastable state batches.
    Costs are dimensionless task-design assumptions, not vehicle calibration.
    """
    s, n = np.asarray(current, dtype=float), np.asarray(next_state, dtype=float)
    if s.shape[-1:] != (4,) or n.shape[-1:] != (4,) or not np.isfinite(s).all() or not np.isfinite(n).all():
        raise ValueError('Reward requires finite [...,4] current and next states')
    p, u = _previous(previous_u), _previous(applied_u)
    gap_error = np.clip(n[..., 0] - (4. + 1.2 * n[..., 1]), -50., 50.)
    speed_error = np.clip(n[..., 1] - 12., -20., 20.)
    reserve_error = np.clip(np.maximum(0., 2. + .5 * n[..., 1] - n[..., 0]), 0., 20.)
    return {'gap_cost': .01 * gap_error**2, 'speed_cost': .1 * speed_error**2,
            'action_cost': .01 * u**2, 'slew_cost': .1 * (u - p)**2,
            'collision_cost': 100. * (n[..., 0] <= 0), 'reserve_cost': 10. * reserve_error**2}


def reward_transition(current, next_state, previous_u, applied_u):
    dense_cost = sum(reward_components(current, next_state, previous_u, applied_u).values())
    value = np.where(np.asarray(next_state)[..., 0] <= 0, -100., 1. - np.minimum(2., dense_cost / 10.))
    return float(value) if np.ndim(value) == 0 else value


def apply_action(current, previous_u, action_index):
    """Discrete delta then box limit then common independent safety override."""
    if isinstance(action_index, (bool, np.bool_)) or not isinstance(action_index, (int, np.integer)) or not 0 <= action_index < ACTION_COUNT:
        raise ValueError('Action index must be integer in [0,4]')
    p = _previous(previous_u)
    if p.ndim != 0:
        raise ValueError('Single transition requires scalar previous command')
    s = np.asarray(current, dtype=float)
    if s.shape != (4,) or not np.isfinite(s).all() or s[0] <= 0 or np.any(s[1:3] < 0):
        raise ValueError('Cannot act on invalid or terminal current state')
    requested = float(np.clip(float(p) + ACTION_DELTAS[action_index], -6., 2.))
    applied, reason = SafetySupervisor().filter(s, requested)
    return applied, {'action_index': int(action_index), 'requested_action_mps2': requested,
                     'applied_action_mps2': float(applied), 'applied_u': float(applied), 'command': float(applied),
                     'fallback': reason}


def _finish(history, previous_u, applied_u, next_state, info):
    n = np.asarray(next_state, dtype=float)
    if n.shape != (4,) or not np.isfinite(n).all() or np.any(n[1:3] < 0):
        raise ValueError('Dynamics returned invalid next state')
    s = history[-1]
    new_history = np.concatenate((history[1:], n[None]), axis=0)
    costs = reward_components(s, n, previous_u, applied_u)
    actual_accel = (n[1] - s[1]) / DT
    previous_accel = (s[1] - history[-2, 1]) / DT
    info.update(next_state=n.copy(), state=n.copy(), raw_history=new_history.copy(), reward_costs={k: float(v) for k, v in costs.items()},
                collision=bool(n[0] <= 0), gap_m=float(n[0]), ego_speed_mps=float(n[1]),
                lead_speed_mps=float(n[2]), progress_m=float(s[1] * DT),
                measured_accel_mps2=float(actual_accel), jerk_mps3=float((actual_accel - previous_accel) / DT),
                gap_error_m=float(n[0] - (4. + 1.2 * n[1])),
                dense_cost=float(sum(costs.values())), reward_version=REWARD_VERSION)
    return new_history, reward_transition(s, n, previous_u, applied_u), bool(n[0] <= 0), info


def synthetic_step(world_model, rawhistory, previous_u, action_index):
    """Return (next_raw_history, reward, terminated, info), never a time limit.

    Only calls the model with current history and candidate applied command.
    No family, environment condition, script, parameters or future labels are read.
    Model rollouts must handle their own truncation and model-uncertainty limits.
    """
    h = _history_array(rawhistory)
    if h.ndim != 2:
        raise ValueError('synthetic_step requires one [6,4] history')
    u, info = apply_action(h[-1], previous_u, action_index)
    n = world_model.step(h[-1].copy(), u, history=h.copy())
    nh, reward, terminated, info = _finish(h, previous_u, u, n, info)
    info.update(transition_source='model', terminated=terminated, truncated=False,
                bootstrap_mask=0. if terminated else 1.)
    return nh, reward, terminated, info


class LongitudinalRLEnv:
    """Gym-style reset/step without a Gym dependency; privileged states only."""
    observation_size = OBSERVATION_SIZE
    action_count = ACTION_COUNT
    dt = DT
    max_steps = MAX_STEPS

    def __init__(self):
        self._episode = None
        self._done = True
        self._steps = 0
        self._history = None
        self._previous_u = 0.
        self._totals = {}

    @property
    def raw_history(self):
        if self._history is None:
            raise RuntimeError('reset must precede observation')
        return self._history.copy()

    @property
    def previous_u(self):
        return self._previous_u

    def reset(self, *, seed, family, condition):
        if family not in FAMILIES or condition not in ('id', 'ood'):
            raise ValueError('Unknown scenario family or ID/OOD condition')
        self._episode = make_episode(family, seed, condition)
        self._history = np.repeat(self._episode['state'][None], HISTORY, axis=0)
        self._previous_u = 0.
        self._steps = 0
        self._done = False
        self._totals = {'return': 0., 'progress_m': 0., 'fallback_count': 0, 'gap_error_squared': 0.,
                        'jerk_squared': 0., 'reference_progress_m': 0.}
        return observation(self._history, self._previous_u), {
            'seed': int(seed), 'family': family, 'condition': condition, 'raw_history': self.raw_history,
            'state': self.state, 'steps': 0, 'time_s': 0., 'previous_u': 0., 'reward_version': REWARD_VERSION, 'state_access': 'privileged_simulator'}

    @property
    def history(self):
        return self.raw_history

    @property
    def state(self):
        return self.raw_history[-1]

    @property
    def previous_action(self):
        return self._previous_u

    @property
    def tick(self):
        return self._steps

    def step(self, action_index):
        if self._done or self._episode is None:
            raise RuntimeError('reset required before step or after episode end')
        h = self._history
        u, info = apply_action(h[-1], self._previous_u, action_index)
        lead_accel = scenario_lead_accel(self._episode['family'], self._steps * DT, self._episode['params'])
        n = step_truth(h[-1], u, lead_accel, self._episode['condition'])
        new_history, reward, terminated, info = _finish(h, self._previous_u, u, n, info)
        self._steps += 1
        truncated = bool(self._steps >= self.max_steps and not terminated)
        self._history, self._previous_u = new_history, float(u)
        self._done = terminated or truncated
        totals = self._totals
        totals['return'] += reward
        totals['progress_m'] += info['progress_m']
        totals['reference_progress_m'] += float(h[-1, 2] * DT)
        totals['fallback_count'] += bool(info['fallback'])
        totals['gap_error_squared'] += info['gap_error_m']**2
        totals['jerk_squared'] += info['jerk_mps3']**2
        info.update(transition_source='truth', steps=self._steps, time_s=self._steps * DT,
                    terminated=terminated, truncated=truncated, completed=False, bootstrap_mask=0. if terminated else 1.,
                    cumulative_progress_m=float(totals['progress_m']), reference_progress_m=float(totals['reference_progress_m']))
        if self._done:
            info['episode'] = {'return': float(totals['return']), 'length': self._steps,
                'collision': int(terminated), 'progress_m': float(totals['progress_m']),
                'reference_progress_m': float(totals['reference_progress_m']),
                'completed': int(truncated and totals['progress_m'] >= .5 * max(5., totals['reference_progress_m'])),
                'fallback_count': int(totals['fallback_count']),
                'gap_rmse_m': float(np.sqrt(totals['gap_error_squared'] / self._steps)),
                'jerk_rms_mps3': float(np.sqrt(totals['jerk_squared'] / self._steps))}
        if self._done:info['completed']=bool(info['episode']['completed'])
        return observation(new_history, u), reward, terminated, truncated, info
