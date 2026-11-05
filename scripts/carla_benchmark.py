"""Real CARLA 0.9.15 data/fit/recursive forecast/CEM speed tracking benchmark.

Run using the CARLA installation's Python 3.9 environment. Own actors only are
destroyed; existing server/map and original world settings are preserved.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from falconworld.adapters import CarlaAdapter, carla_control
from falconworld.contracts import Action
from falconworld.planning import CEMConfig,CEMOptimizer


def features(v, a, u, rich=False):
    t, b = np.maximum(u, 0) / 3, np.maximum(-u, 0) / 6
    columns = [np.ones_like(v), a, t, b, v]
    if rich:
        columns += [v*v/25, v*t/5, v*b/5, t*t, b*b, np.tanh(v/0.5)]
    return np.stack(columns, axis=-1)


def fit(rows, rich, ridge):
    values = np.asarray(rows)
    x = features(values[:, 0], values[:, 1], values[:, 2], rich)
    y = values[:, 3]
    return np.linalg.solve(x.T @ x + ridge*np.eye(x.shape[1]), x.T @ y)


def predict(v, a, u, weight, rich, dt):
    next_a = np.clip(features(v, a, u, rich) @ weight, -9, 5)
    next_v = np.maximum(0, v + dt*next_a)
    return next_v, (next_v-v)/dt


def steering(world, actor):
    import carla
    transform = actor.get_transform()
    waypoint = world.get_map().get_waypoint(transform.location)
    ahead = waypoint.next(7.0)
    target = ahead[0].transform.location if ahead else waypoint.transform.location
    dx, dy = target.x-transform.location.x, target.y-transform.location.y
    alpha = math.atan2(dy, dx)-math.radians(transform.rotation.yaw)
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    # right-positive CARLA desired yaw is inverted at SI action boundary.
    return -math.atan2(2*2.8*math.sin(alpha), max(math.hypot(dx,dy),1))


def spawn(world, index):
    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    bp.set_attribute('role_name', 'falconworld_benchmark')
    positions = world.get_map().get_spawn_points()
    for offset in range(len(positions)):
        actor = world.try_spawn_actor(bp, positions[(index+offset)%len(positions)])
        if actor is not None:
            return actor
    raise RuntimeError('No free CARLA vehicle spawn; existing actors were preserved')


def collect(world, seed, steps, dt):
    rng = np.random.default_rng(seed)
    actor = spawn(world, 18)
    rows, trace = [], []
    physics=actor.get_physics_control()
    spawn_transform=actor.get_transform()
    actor_metadata=dict(blueprint=actor.type_id,mass_kg=physics.mass,drag_coefficient=physics.drag_coefficient,
        bounding_box_half_extents_m=dict(x=actor.bounding_box.extent.x,y=actor.bounding_box.extent.y,z=actor.bounding_box.extent.z),
        wheel_max_steer_deg=[w.max_steer_angle for w in physics.wheels],
        wheel_positions_cm=[dict(x=w.position.x,y=w.position.y,z=w.position.z) for w in physics.wheels],
        spawn=dict(x=spawn_transform.location.x,y=spawn_transform.location.y,z=spawn_transform.location.z,
                   yaw_deg=spawn_transform.rotation.yaw,pitch_deg=spawn_transform.rotation.pitch,roll_deg=spawn_transform.rotation.roll))
    try:
        with CarlaAdapter(world, actor, dt) as adapter:
            for _ in range(12):
                adapter.step(Action(0))
            obs = adapter.observe()
            a, u = 0., 0.
            for i in range(steps):
                if i % 15 == 0:
                    u = float(rng.choice([-4.,-2.,-0.5,0.,1.,2.,3.]))
                old_v = obs.speed
                obs = adapter.step(Action(u, steering(world, actor)))
                next_a = (obs.speed-old_v)/dt
                rows.append([old_v,a,u,next_a,obs.speed])
                trace.append(dict(frame=obs.frame,timestamp=obs.timestamp,speed=obs.speed,
                                  measured_acceleration=next_a,command=u,x=obs.x,y=obs.y))
                a = next_a
    finally:
        actor.destroy()
    return np.asarray(rows), trace,actor_metadata


def forecast_metrics(rows, weights, dt):
    result = {}
    for name, (weight, rich) in weights.items():
        result[name] = {}
        for seconds in (0.5,1.,3.):
            horizon = round(seconds/dt)
            errors = []
            # Each window recursively consumes only its own predicted state.
            for start in range(0,len(rows)-horizon+1, max(horizon,1)):
                v,a = rows[start,:2]
                for j in range(horizon):
                    v,a = predict(v,a,rows[start+j,2],weight,rich,dt)
                errors.append(float(v-rows[start+horizon-1,4]))
            result[name][str(seconds)] = dict(speed_rmse_m_s=float(np.sqrt(np.mean(np.square(errors)))),windows=len(errors))
    return result


def cem(v,a,last,weight,rich,target,dt,rng):
    n,h,iters = 96,20,3
    inference_s = 0.
    def objective(sequences):
        nonlocal inference_s
        u=sequences[:,:,0]
        pv, pa, previous = np.full(n,v),np.full(n,a),np.full(n,last)
        cost = np.zeros(n)
        begin = time.perf_counter()
        for k in range(h):
            pv,pa = predict(pv,pa,u[:,k],weight,rich,dt)
            cost += (pv-target)**2 + 0.035*u[:,k]**2 + 0.15*(u[:,k]-previous)**2
            previous = u[:,k]
        inference_s += time.perf_counter()-begin
        return cost
    optimizer=CEMOptimizer(CEMConfig(candidates=n,horizon=h,iterations=iters,elites=12,dt=dt))
    sequence,_=optimizer.optimize(objective,[-4],[3],previous=[last],slew=.8,
        seed=int(rng.integers(0,2**31-1)))
    return (float(sequence[0,0]) if sequence is not None else float('nan')),inference_s


def closed_loop(world,name,weight,rich,seed,steps,dt):
    import carla
    actor=spawn(world,18)
    sensor=None
    collision_events=[]
    records=[]
    rng=np.random.default_rng(seed)
    try:
        sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=actor)
        sensor.listen(lambda event: collision_events.append(dict(frame=event.frame,other=event.other_actor.type_id)))
        with CarlaAdapter(world,actor,dt) as adapter:
            for _ in range(12): adapter.step(Action(0))
            obs=adapter.observe()
            a,last=0.,0.
            for i in range(steps):
                target=5. if i<80 else (2. if i<130 else (0. if i<160 else 5.))
                start=time.perf_counter()
                command,inference=cem(obs.speed,a,last,weight,rich,target,dt,rng)
                steer=steering(world,actor)
                planning=time.perf_counter()-start
                fallback=planning>dt or not math.isfinite(command)
                if fallback: command=-4.
                old=obs
                obs=adapter.step(Action(command,steer))
                new_a=(obs.speed-old.speed)/dt
                waypoint=world.get_map().get_waypoint(actor.get_location())
                lane_error=actor.get_location().distance(waypoint.transform.location)
                cycle=time.perf_counter()-start
                records.append(dict(step=i,frame=obs.frame,timestamp=obs.timestamp,speed=obs.speed,target=target,
                    acceleration=new_a,jerk=(new_a-a)/dt,command=command,x=obs.x,y=obs.y,
                    lane_error_m=lane_error,cycle_ms=cycle*1000,planning_ms=planning*1000,inference_ms=inference*1000,
                    fallback=int(fallback),collision_events=len(collision_events)))
                a,last=new_a,command
                if collision_events: break
    finally:
        if sensor is not None: sensor.stop(); sensor.destroy()
        actor.destroy()
    elapsed=len(records)*dt
    cycles=np.array([r['cycle_ms'] for r in records])
    return records,dict(model=name,seed=seed,steps=len(records),collision=bool(collision_events),
        collision_events=collision_events,completion=len(records)==steps and not collision_events,
        progress_m=float(sum(r['speed']*dt for r in records)),
        speed_rmse_m_s=float(np.sqrt(np.mean([(r['speed']-r['target'])**2 for r in records]))),
        jerk_rms_m_s3=float(np.sqrt(np.mean([r['jerk']**2 for r in records]))),
        lane_rmse_m=float(np.sqrt(np.mean([r['lane_error_m']**2 for r in records]))),
        cycle_p50_ms=float(np.percentile(cycles,50)),cycle_p95_ms=float(np.percentile(cycles,95)),
        timeout_rate=float(np.mean(cycles>dt*1000)),fallback_rate=float(np.mean([r['fallback'] for r in records])),
        model_rollout_mean_ms=float(np.mean([r['inference_ms'] for r in records])),simulated_seconds=elapsed)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=2100)
    p.add_argument('--output',default='outputs/carla');p.add_argument('--steps',type=int,default=240)
    args=p.parse_args()
    if args.steps < 1:p.error('--steps must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        p.error('Output directory must be empty: preserve existing experiments and choose a new --output')
    c=carla.Client(args.host,args.port);c.set_timeout(20)
    if c.get_client_version()!=c.get_server_version(): raise RuntimeError('CARLA client/server mismatch')
    world=c.get_world();dt=.1
    original=world.get_settings()
    metadata=dict(client=c.get_client_version(),server=c.get_server_version(),map=world.get_map().name,host=args.host,port=args.port,
        dt=dt,train_seeds=[101,102],validation_seed=201,test_seed=301,
        closed_seeds=[401,402],state='speed and backward difference acceleration; no hidden physics parameters',
        optimizer='falconworld.planning.CEMOptimizer',
        original_world_settings={key:getattr(original,key) for key in ['synchronous_mode','fixed_delta_seconds','no_rendering_mode',
            'substepping','max_substep_delta_time','max_substeps','deterministic_ragdolls']},
        weather=str(world.get_weather()),
        note='CARLA integration experiment, no lead car, learned response MPC + shared waypoint steering, episode split',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('falconworld/adapters.py'),Path('falconworld/contracts.py'),Path('falconworld/planning.py')]})
    (out/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    datasets=[]
    for seed in [101,102,201,301]:
        rows,trace,actor_metadata=collect(world,seed,max(args.steps,240),dt);datasets.append(rows)
        metadata['actor_configuration']=actor_metadata
        (out/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        write_csv(out/('data_%s.csv'%seed),trace)
        print('collected',seed,len(rows),flush=True)
    train=np.concatenate(datasets[:2]);valid=datasets[2];test=datasets[3]
    weights={};selection={}
    for name,rich in [('calibrated_physics',False),('learned_response',True)]:
        options=[]
        for ridge in [.001,.1,1.,10.]:
            weight=fit(train,rich,ridge)
            score=forecast_metrics(valid,{name:(weight,rich)},dt)[name]['1.0']['speed_rmse_m_s']
            options.append((score,ridge,weight))
        score,ridge,weight=min(options,key=lambda z:z[0]);weights[name]=(weight,rich)
        selection[name]=dict(ridge=ridge,validation_1s_rmse=score)
    np.savez(out/'models.npz',**{name:value[0] for name,value in weights.items()})
    with np.load(out/'models.npz',allow_pickle=False) as saved:
        weights={name:(saved[name].copy(),rich) for name,(_,rich) in weights.items()}
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    metrics=dict(prediction_test=forecast_metrics(test,weights,dt),closed_loop=[])
    for seed in [401,402]:
        for name,(weight,rich) in weights.items():
            trace,result=closed_loop(world,name,weight,rich,seed,args.steps,dt)
            write_csv(out/('closed_%s_%s.csv'%(name,seed)),trace)
            metrics['closed_loop'].append(result)
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            print(result,flush=True)
    print('CARLA benchmark complete:',out,flush=True)


if __name__=='__main__':main()
