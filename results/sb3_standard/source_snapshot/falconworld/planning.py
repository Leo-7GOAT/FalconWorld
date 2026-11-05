"""Shared vectorized CEM optimizer and receding-horizon longitudinal control.

All longitudinal state experiments here use simulator privileged actuator state.
The optimizer itself is vehicle-independent; the objective owns dynamics/units.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 30
    candidates: int = 64
    iterations: int = 3
    elites: int = 8
    dt: float = .1
    smoothing: float = .2

    def __post_init__(self):
        if self.horizon < 1 or self.iterations < 1 or not 1 <= self.elites <= self.candidates:
            raise ValueError('Invalid CEM dimensions')
        if not np.isfinite(self.dt) or self.dt <= 0 or not 0 <= self.smoothing < 1:
            raise ValueError('Invalid CEM timing/smoothing')


class CEMOptimizer:
    def __init__(self, config=CEMConfig()):
        self.config = config

    def optimize(self, objective, action_low, action_high, previous=None, seed=0,
                 initial=None, slew=None):
        """Evaluate full B,H,A sequences; return minimum-cost sampled sequence.

        `previous` is last executed A-vector; `initial` is a warm-start sequence.
        Slew is maximum change per interval, applied to every candidate.
        """
        start = perf_counter()
        cfg = self.config
        low, high = np.atleast_1d(action_low), np.atleast_1d(action_high)
        if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
            raise ValueError('Invalid action box')
        dim = len(low)
        prev = np.zeros(dim) if previous is None else np.asarray(previous).reshape(dim)
        mean = np.tile(prev, (cfg.horizon, 1)) if initial is None else np.array(initial, copy=True)
        std = np.broadcast_to((high-low)/2, mean.shape).copy()
        rng = np.random.default_rng(seed)
        best, best_cost = None, np.inf
        finite_evaluations = 0
        for _ in range(cfg.iterations):
            # Smooth sampled noise in time so candidate budget explores useful controls.
            noise = rng.standard_normal((cfg.candidates, cfg.horizon, dim))
            for t in range(1, cfg.horizon):
                noise[:, t] = .75*noise[:, t-1] + .6614378278*noise[:, t]
            sequences = np.clip(mean + noise*std, low, high)
            sequences[0] = np.clip(mean, low, high)
            if slew is not None:
                last = np.broadcast_to(prev, (cfg.candidates, dim))
                for t in range(cfg.horizon):
                    sequences[:, t] = np.clip(sequences[:, t], last-slew, last+slew)
                    last = sequences[:, t]
            costs = np.asarray(objective(sequences), dtype=float)
            if costs.shape != (cfg.candidates,):
                raise ValueError('Objective must return one cost per candidate')
            finite_evaluations += int(np.isfinite(costs).sum())
            costs = np.where(np.isfinite(costs), costs, np.inf)
            order = np.argsort(costs)
            if costs[order[0]] < best_cost:
                best_cost, best = float(costs[order[0]]), sequences[order[0]].copy()
            elite = sequences[order[:cfg.elites]]
            mean = cfg.smoothing*mean + (1-cfg.smoothing)*elite.mean(axis=0)
            std = np.maximum(.05*(high-low), cfg.smoothing*std + (1-cfg.smoothing)*elite.std(axis=0))
        return best, {'cost': best_cost, 'solve_s': perf_counter()-start,
                      'finite_evaluations': finite_evaluations, **asdict(cfg)}


class SafetySupervisor:
    """Independent emergency override; counted separately from planner success."""
    def filter(self, state, command, solver_ok=True):
        state = np.asarray(state)
        if not solver_ok or not np.isfinite(command):
            return -6., 'solver_failure'
        if not np.all(np.isfinite(state)):
            return -6., 'invalid_observation'
        gap, ego, lead, _ = state
        closing = max(0., ego-lead)
        # Reaction-distance reserve, plus differential stopping-distance estimate.
        reserve = 1.5 + .35*ego + max(0., ego*ego-lead*lead)/(2*6.)
        if gap < reserve or (closing > 0 and gap/closing < .7):
            return -6., 'emergency_gap'
        return float(np.clip(command, -6, 2)), ''


class LongitudinalMPC:
    def __init__(self, model, config=CEMConfig(), risk_weight=0., target_speed=12., seed=0):
        self.model, self.optimizer = model, CEMOptimizer(config)
        self.risk_weight, self.target_speed, self.seed = risk_weight, target_speed, seed
        self.previous, self.warm, self.tick = 0., None, 0
        self.supervisor = SafetySupervisor()

    def command(self, state, history):
        start = perf_counter()
        cfg = self.optimizer.config
        state,history=np.asarray(state,dtype=float),np.asarray(history,dtype=float)
        if (state.shape!=(4,) or history.ndim!=2 or history.shape[1]!=4 or len(history)<2
                or not np.isfinite(state).all() or not np.isfinite(history).all()
                or np.any(state[1:3]<0) or state[0]<=0):
            self.previous,self.warm=-6.,None
            return -6.,dict(cost=float('inf'),inference_s=0.,cycle_s=perf_counter()-start,
                            fallback='invalid_observation',planned_command=-6.)
        inference_s = 0.
        def objective(sequences):
            nonlocal inference_s
            count = len(sequences)
            current = np.broadcast_to(state, (count, 4)).copy()
            hist = np.broadcast_to(np.asarray(history), (count, len(history), 4)).copy()
            members = getattr(self.model, 'members', None)
            member_states = [current.copy() for _ in members] if members else None
            member_history = [hist.copy() for _ in members] if members else None
            costs = np.zeros(count)
            prev = np.full(count, self.previous)
            for t in range(cfg.horizon):
                before = perf_counter()
                if members:
                    for j, member in enumerate(members):
                        member_states[j] = member.step(member_states[j], sequences[:, t, 0], history=member_history[j])
                        member_history[j] = np.concatenate((member_history[j][:, 1:], member_states[j][:, None, :]), axis=1)
                    future = np.mean(member_states, axis=0)
                    gap_risk = np.std(member_states, axis=0)[:, 0]
                else:
                    future = self.model.step(current, sequences[:, t, 0], history=hist)
                    gap_risk = 0.
                inference_s += perf_counter()-before
                gap, speed = future[:, 0], future[:, 1]
                desired = 4. + 1.2*speed
                effective_gap = gap-self.risk_weight*gap_risk
                costs += .12*(gap-desired)**2 + (speed-self.target_speed)**2
                costs += 300.*np.maximum(0., 2.+.5*speed-effective_gap)**2
                costs += 1e6*(effective_gap <= 0.)
                costs += .08*sequences[:, t, 0]**2 + .5*(sequences[:, t, 0]-prev)**2
                prev = sequences[:, t, 0]
                current = future
                hist = np.concatenate((hist[:, 1:], future[:, None, :]), axis=1)
            return costs
        try:
            best, info = self.optimizer.optimize(objective, [-6.], [2.], [self.previous],
                                                 seed=self.seed+self.tick, initial=self.warm, slew=.8)
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as error:
            self.previous,self.warm=-6.,None
            self.tick+=1
            return -6.,dict(cost=float('inf'),inference_s=inference_s,cycle_s=perf_counter()-start,
                            fallback='model_or_solver_exception',failure_detail=str(error),planned_command=-6.)
        ok = best is not None and np.isfinite(info['cost'])
        planned = best[0, 0] if ok else -6.
        command, reason = self.supervisor.filter(state, planned, ok)
        if perf_counter()-start>cfg.dt:
            command,reason=-6.,'deadline_exceeded'
        if ok:
            self.warm = np.concatenate((best[1:], best[-1:]))
        self.previous, self.tick = command, self.tick+1
        info.update(inference_s=inference_s, cycle_s=perf_counter()-start,
                    fallback=reason, planned_command=float(planned))
        return command, info


class ACC:
    """Conventional gap/speed feedback with the same emergency supervisor."""
    def __init__(self):
        self.previous = 0.
        self.supervisor = SafetySupervisor()

    def command(self, state, history):
        start = perf_counter()
        gap, ego, lead, _ = state
        raw = min(.45*(gap-4.-1.2*ego)+.9*(lead-ego), .8*(12.-ego))
        raw = np.clip(raw, self.previous-.8, self.previous+.8)
        command, reason = self.supervisor.filter(state, raw)
        self.previous = command
        return command, dict(cycle_s=perf_counter()-start, inference_s=0., fallback=reason)
