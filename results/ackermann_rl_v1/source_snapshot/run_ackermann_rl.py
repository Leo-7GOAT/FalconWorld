"""Reproducible joint lateral/longitudinal PPO research; CPU simulation only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import hashlib
import json
import shutil
from time import perf_counter
import numpy as np
import torch
from stable_baselines3 import PPO
from falconworld.ackermann import PhysicalModel, ResidualModel, bounded_action, step_ackermann, train_models
from falconworld.ackermann_rl import AckermannTrackingEnv, AckermannWorldModelPPO, CFG, MAX_STEPS

METHODS = ('ppo', 'wm_ppo', 'physics_ppo')
PRESETS = dict(smoke=dict(seeds=[0], steps=4096, pairs=2),
               quick=dict(seeds=[0, 1, 2], steps=65536, pairs=8))
CONFIG = dict(n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
              gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=.5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def offline(root, seed):
    folder = root/f'seed{seed}'/'models'
    folder.mkdir(parents=True)
    records = dict(train=[], validation=[])
    episodes = []
    for eid in range(30):
        split = 'train' if eid < 24 else 'validation'
        episode_seed = 80_000_000+seed*100_000+eid
        rng = np.random.default_rng(episode_seed)
        s = np.array([0., 0., rng.uniform(-.2, .2), rng.uniform(.5, 6), 0., 0.])
        previous = np.zeros(2)
        for step in range(100):
            if step % 8 == 0:
                requested = rng.uniform([-2.5, -.45], [1.8, .45])
            u = bounded_action(requested, previous, CFG)
            n = step_ackermann(s, u, CFG)
            records[split].append((s.copy(), u.copy(), n.copy()))
            s, previous = n, u
        episodes.append(dict(seed=episode_seed, split=split, steps=100))
    data = {}
    for split, rows in records.items():
        for i, name in enumerate(('state', 'action', 'next')):
            data[split+'_'+name] = np.array([r[i] for r in rows])
    np.savez_compressed(folder/'data.npz', **data)
    start = perf_counter()
    train_models(data, folder, CFG)
    write(folder/'manifest.json', dict(episodes=episodes, interactions=3000, train=2400, validation=600,
        test=0, dataset_sha256=sha(folder/'data.npz'), training_seconds=perf_counter()-start,
        model_sha256={p.name:sha(p) for p in folder.glob('*.npz') if p.name != 'data.npz'}))


def evaluate(policy, folder, seed, pairs, split='test'):
    folder.mkdir(parents=True, exist_ok=False)
    env = AckermannTrackingEnv()
    episodes = []
    traces = []
    namespace = 90_000_000 if split == 'test' else 85_000_000
    for ci, condition in enumerate((1., 1.3)):
        for fi, family in enumerate(('straight', 'curve')):
            for pair in range(pairs):
                case_seed = namespace+seed*100_000+ci*10_000+fi*1000+pair
                obs, _ = env.reset(options=dict(episode_seed=case_seed, family=family, condition=condition))
                ret = 0.; errors = []; speeds = []; headings = []; steer_slew = []; jerk = []; times = []
                history = []
                for step in range(MAX_STEPS):
                    start = perf_counter()
                    action, _ = policy.predict(obs, deterministic=True)
                    times.append((perf_counter()-start)*1000)
                    obs, reward, terminal, truncated, info = env.step(action)
                    ret += reward
                    errors.append(info['lateral_error_m']); speeds.append(info['speed_error_mps'])
                    headings.append(info['heading_error_rad'])
                    steer_slew.append((info['applied'][1]-info['previous'][1])/CFG.dt)
                    jerk.append((info['next_state'][4]-info['state'][4])/CFG.dt)
                    history.append(np.r_[info['state'], info['next_state'], info['previous'], action, info['applied'], reward,
                        info['failure'], info['success'], terminal, truncated])
                    if terminal or truncated:
                        break
                traces.append((case_seed, np.array(history)))
                episodes.append(dict(training_seed=seed, episode_seed=case_seed, family=family,
                    distribution='id' if ci == 0 else 'ood', condition=condition,
                    steps=env.steps, completed=bool(info['success']), road_departure=bool(info['failure']),
                    timeout=bool(truncated), return_raw=ret, progress_m=float(env.state[0]),
                    lateral_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                    heading_rmse_rad=float(np.sqrt(np.mean(np.square(headings)))),
                    speed_rmse_mps=float(np.sqrt(np.mean(np.square(speeds)))),
                    steering_rate_rms_radps=float(np.sqrt(np.mean(np.square(steer_slew)))),
                    jerk_rms_mps3=float(np.sqrt(np.mean(np.square(jerk)))),
                    inference_p95_ms=float(np.percentile(times, 95))))
    with (folder/'episodes.jsonl').open('w', encoding='utf8') as f:
        for record in episodes:
            f.write(json.dumps(record, allow_nan=False)+'\n')
    np.savez_compressed(folder/'trajectories.npz', **{str(k):v for k,v in traces})
    write(folder/'trace_schema.json', dict(columns=[*[f'state_{i}' for i in range(6)],
        *[f'next_state_{i}' for i in range(6)], 'previous_accel', 'previous_steer',
        'raw_action_accel', 'raw_action_steer', 'applied_accel_mps2', 'applied_steer_rad',
        'reward', 'road_departure', 'success', 'terminated', 'truncated']))
    return episodes


def job(args):
    root, seed, method, spec = Path(args['root']), args['seed'], args['method'], args['spec']
    torch.set_num_threads(1)
    out = root/f'seed{seed}'/method
    out.mkdir()
    predictor = ResidualModel.load(root/f'seed{seed}/models/residual.npz') if method == 'wm_ppo' else (
        PhysicalModel.load(root/f'seed{seed}/models/rich_physical.npz') if method == 'physics_ppo' else None)
    env = AckermannTrackingEnv(training_seed=seed)
    model = AckermannWorldModelPPO('MlpPolicy', env, seed=seed, device='cpu', world_model=predictor, **CONFIG)
    curves = []
    start = perf_counter(); training_seconds = 0.
    # Fixed final budget; validation does not select checkpoints or parameters.
    checkpoints = min(4, spec['steps']//2048)
    interval = spec['steps']//checkpoints
    for _ in range(checkpoints):
        before = perf_counter()
        model.learn(total_timesteps=interval, reset_num_timesteps=False)
        training_seconds += perf_counter()-before
        # Independent environment; deterministic action calls do not consume RNG.
        rows = evaluate(model, out/'validation'/str(env.transition_count), seed, 1, 'validation')
        curves.append(dict(online_steps=env.transition_count, mean_return=float(np.mean([r['return_raw'] for r in rows])),
            completion_rate=float(np.mean([r['completed'] for r in rows])),
            departure_rate=float(np.mean([r['road_departure'] for r in rows]))))
        write(out/'learning_curve.json', curves)
        print(f'{method} seed{seed} {env.transition_count}/{spec["steps"]} val={curves[-1]["mean_return"]:.2f} completion={curves[-1]["completion_rate"]:.2f}', flush=True)
    assert env.transition_count == model.num_timesteps == spec['steps']
    checkpoint = out/'final_policy.zip'
    model.save(checkpoint)
    frozen = PPO.load(checkpoint, device='cpu')
    probe, _ = AckermannTrackingEnv().reset(seed=123)
    np.testing.assert_array_equal(model.predict(probe, deterministic=True)[0], frozen.predict(probe, deterministic=True)[0])
    rows = evaluate(frozen, out/'test', seed, spec['pairs'])
    ledger = dict(method=method, training_seed=seed, online_steps=env.transition_count,
        optimizer_steps=model.optimizer_steps, synthetic_generated=model.model_generated,
        synthetic_replaced=model.model_replaced, branch_metadata=model.last_branch,
        offline_interactions=3000 if predictor is not None else 0,
        training_seconds=training_seconds, elapsed_seconds=perf_counter()-start,
        checkpoint_sha256=sha(checkpoint), test_sha256=sha(out/'test/episodes.jsonl'),
        trajectory_sha256=sha(out/'test/trajectories.npz'), test_episodes=len(rows))
    write(out/'complete.json', ledger)
    return ledger


def summarize(root, spec, ledgers):
    rows = []
    for seed in spec['seeds']:
        for method in METHODS:
            rows += [dict(json.loads(line), method=method) for line in
                     (root/f'seed{seed}'/method/'test/episodes.jsonl').read_text().splitlines()]
    summary = []
    for method in METHODS:
        for dist in ('id', 'ood'):
            selected = [r for r in rows if r['method']==method and r['distribution']==dist]
            summary.append(dict(method=method, distribution=dist, episodes=len(selected),
                completions=sum(r['completed'] for r in selected), departures=sum(r['road_departure'] for r in selected),
                **{name:float(np.mean([r[name] for r in selected])) for name in ('return_raw','progress_m','lateral_rmse_m','speed_rmse_mps','steering_rate_rms_radps')}))
    with (root/'summary.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write(root/'summary.json', summary)
    write(root/'cost_ledger.json', ledgers)
    expected_jobs = len(spec['seeds'])*len(METHODS)
    expected_episodes = expected_jobs*4*spec['pairs']
    assert len(rows)==expected_episodes
    keys = [(r['method'],r['training_seed'],r['episode_seed']) for r in rows]
    assert len(set(keys)) == len(keys)
    for row in ledgers:
        assert row['optimizer_steps']==spec['steps']//2048*320
        assert row['synthetic_replaced']==(0 if row['method']=='ppo' else spec['steps']//2)
    write(root/'completeness.json', dict(expected_jobs=expected_jobs, completed_jobs=len(ledgers),
        expected_episodes=expected_episodes, actual_episodes=len(rows), verified_complete=True))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='smoke')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); spec = PRESETS[args.preset]
    root = args.output or Path('outputs/ackermann_rl')/args.preset
    root.mkdir(parents=True, exist_ok=False)
    sources = ['run_ackermann_rl.py','falconworld/ackermann_rl.py','falconworld/ackermann.py']
    write(root/'registration.json', dict(preset=args.preset, spec=spec, methods=METHODS, policy_config=CONFIG,
        sources={name:sha(name) for name in sources}, policy_selection='fixed final budget; validation monitoring only',
        observation='14D, privileged actuator states and known route geometry; no hidden condition',
        action='continuous normalized acceleration and front-wheel angle; limits and slew applied equally',
        model='train/validation ridge-selected residual; no test fitting; frozen before policy training',
        model_branch='one step; half-buffer replacement; raw Gaussian actions/logprobs; terminal mask; no extra updates',
        limits='simulation only; no visual input, obstacle routing, vehicle deployment or general superiority claim',
        versions=dict(torch=torch.__version__,numpy=np.__version__)))
    for name in sources:
        target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    for seed in spec['seeds']:
        offline(root, seed)
    jobs = [dict(root=str(root), seed=seed, method=method, spec=spec) for seed in spec['seeds'] for method in METHODS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        ledgers = list(executor.map(job, jobs))
    summarize(root, spec, ledgers)


if __name__ == '__main__':
    main()
