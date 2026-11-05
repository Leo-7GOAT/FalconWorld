"""Export complete state/history/action transitions, not just neural heads, to ONNX."""
from pathlib import Path
import hashlib
import json
import numpy as np


def export_onnx(model, path):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper
    from .models import PhysicsModel, LearnedModel
    if not isinstance(model,(PhysicsModel,LearnedModel)):
        raise TypeError('Export members separately for an ensemble')
    nodes, initializers = [], []
    tick = 0
    def const(value):
        nonlocal tick
        name=f'c{tick}';tick+=1
        initializers.append(numpy_helper.from_array(np.asarray(value,dtype=np.float32),name))
        return name
    def ints(value):
        nonlocal tick
        name=f'i{tick}';tick+=1
        initializers.append(numpy_helper.from_array(np.asarray(value,dtype=np.int64),name))
        return name
    def op(kind,*inputs,**attrs):
        nonlocal tick
        name=f'n{tick}';tick+=1
        nodes.append(h.make_node(kind,list(inputs),[name],**attrs))
        return name
    def gather(value,indices,axis=1):return op('Gather',value,ints(indices),axis=axis)
    def clip(value,low,high):return op('Clip',value,const(low),const(high))
    def add(a,b):return op('Add',a,b)
    def sub(a,b):return op('Sub',a,b)
    def mul(a,b):return op('Mul',a,b)
    def net(network,features):
        x=op('Div',sub(features,const(network.xmean)),const(network.xscale))
        w=network.weights
        hidden=op('Tanh',add(op('MatMul',x,const(w[0])),const(w[1])))
        y=add(op('MatMul',hidden,const(w[2])),const(w[3]))
        return add(mul(y,const(network.yscale)),const(network.ymean))
    gap,speed,lead,actuator=[gather('state',[j]) for j in range(4)]
    action=clip('action',-6,2)
    physics=model if isinstance(model,PhysicsModel) else model.physics
    if physics.enriched is None:
        tau=const(physics.tau)
    else:
        acceleration,braking,k=physics.enriched
        tau=mul(op('Where',op('GreaterOrEqual',action,actuator),const(acceleration),const(braking)),
                add(const(1),mul(const(k),speed)))
    physical_a=add(actuator,op('Div',mul(const(.1),sub(action,actuator)),op('Max',const(.1),tau)))
    history_lead=op('Squeeze',gather('history',[2],axis=2),ints([2]))
    history_delta=sub(gather(history_lead,[1,2,3,4,5]),gather(history_lead,[0,1,2,3,4]))
    if isinstance(model,PhysicsModel):
        new_actuator=physical_a
        delta=clip(gather(history_delta,[4]),-.6,.2) if model.lead=='ca' else mul(lead,const(0))
    else:
        if model.lead_only:
            new_actuator=physical_a
        else:
            features=op('Concat',speed,actuator,action,sub(action,actuator),axis=1)
            learned_a=net(model.actuator_net,features)
            new_actuator=add(physical_a,learned_a) if model.residual else learned_a
        features=op('Concat',lead,history_delta,axis=1) if model.with_history else lead
        delta=clip(net(model.leader_net,features),-.6,.2)
    new_gap=add(gap,mul(const(.1),sub(lead,speed)))
    new_speed=op('Max',const(0),add(speed,mul(const(.1),actuator)))
    new_lead=op('Max',const(0),add(lead,delta))
    result=op('Concat',new_gap,new_speed,new_lead,clip(new_actuator,-6,2),axis=1)
    nodes.append(h.make_node('Identity',[result],['next_state']))
    graph=h.make_graph(nodes,'FalconWorld transition',[
        h.make_tensor_value_info('state',T.FLOAT,['batch',4]),
        h.make_tensor_value_info('action',T.FLOAT,['batch',1]),
        h.make_tensor_value_info('history',T.FLOAT,['batch',6,4])],
        [h.make_tensor_value_info('next_state',T.FLOAT,['batch',4])],initializer=initializers)
    artifact=h.make_model(graph,opset_imports=[h.make_opsetid('',17)])
    artifact.ir_version=9
    onnx.checker.check_model(artifact)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(artifact,str(path))
    return path


def export_ensemble(model, directory):
    """Export each complete member transition plus an auditable aggregation contract."""
    from .models import EnsembleModel
    if not isinstance(model, EnsembleModel) or not model.members:
        raise TypeError('A nonempty EnsembleModel is required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    members = []
    for index, member in enumerate(model.members):
        path = export_onnx(member, directory / f'member_{index}.onnx')
        members.append({'index': index, 'name': member.name, 'file': path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        'format': 'falconworld-onnx-ensemble-v1', 'name': model.name,
        'members': members, 'dt_s': .1,
        'inputs': {'state': ['batch', 4], 'action': ['batch', 1], 'history': ['batch', 6, 4]},
        'state_order': ['gap_m', 'ego_speed_mps', 'leader_speed_mps', 'actuator_mps2'],
        'action_semantics': 'desired acceleration m/s^2, clipped [-6,2]; no throttle or PWM conversion',
        'history_order': 'oldest to newest; includes current state',
        'recursive_state': 'each member owns its predicted state and history; append its own output',
        'aggregation': {'mean': 'arithmetic mean over member dimension',
                        'std': 'population standard deviation over member dimension, ddof=0'},
        'arithmetic': 'ONNX FP32 state transitions; NumPy aggregation',
        'one_step_validation_radius': model.radius.tolist(),
        'risk_scope': 'member disagreement is a heuristic, not calibrated confidence or a safety guarantee',
        'deployment_scope': 'host numerical verification only; Raspberry Pi/ARM64 performance unmeasured',
    }
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return path


class OnnxTransition:
    """One complete exported member with the same batched step interface as NumPy."""
    def __init__(self, path, name=None):
        import onnxruntime as ort
        path = Path(path)
        self.name = name or path.stem
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    def step(self, states, actions, history=None):
        s = np.asarray(states, dtype=np.float32)
        single = s.ndim == 1
        s = np.atleast_2d(s)
        a = np.broadcast_to(np.asarray(actions, dtype=np.float32).reshape(-1), (len(s),))[:, None]
        if history is None:
            h = np.repeat(s[:, None], 6, axis=1)
        else:
            h = np.asarray(history, dtype=np.float32)
            if h.ndim == 2:
                h = h[None]
            if h.shape[-1] != 4 or h.shape[-2] < 2:
                raise ValueError('History must contain at least two four-state observations')
            h = np.broadcast_to(h, (len(s), h.shape[1], 4))
            if h.shape[1] < 6:
                h = np.concatenate((np.repeat(h[:, :1], 6-h.shape[1], axis=1), h), axis=1)
            h = h[:, -6:]
        if s.shape[1] != 4 or not all(np.isfinite(x).all() for x in (s, a, h)):
            raise ValueError('Exported transition requires finite states/actions/history')
        out = self.session.run(None, {'state': s, 'action': a, 'history': h})[0]
        return out[0] if single else out


class OnnxEnsemble:
    """Load all hashed members; the planner may maintain their histories separately."""
    def __init__(self, manifest_path):
        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('format') != 'falconworld-onnx-ensemble-v1':
            raise ValueError('Unsupported ensemble manifest')
        self.name = self.manifest['name']
        self.radius = np.asarray(self.manifest['one_step_validation_radius'])
        self.members = []
        root = manifest_path.parent.resolve()
        for member in self.manifest['members']:
            path = (root / member['file']).resolve()
            if path.parent != root:
                raise ValueError('Member path must remain inside export directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != member['sha256']:
                raise ValueError(f'Ensemble member checksum mismatch: {path.name}')
            self.members.append(OnnxTransition(path, member['name']))
        if not self.members:
            raise ValueError('Ensemble must have members')

    def step(self, states, actions, history=None):
        return np.mean([member.step(states, actions, history) for member in self.members], axis=0)

    def uncertainty(self, states, actions, history=None):
        return np.std([member.step(states, actions, history) for member in self.members], axis=0, ddof=0)

    def interval_radius(self, states, actions, history=None):
        return self.radius + 1.645*self.uncertainty(states, actions, history)

    def rollout(self, states, action_sequences, history):
        """Return mean/std [H,B,4], members [M,H,B,4] with independent histories.

        action_sequences is [B,H] or common [H]. This differs from repeatedly
        calling step() on a mean state: that would collapse distinct trajectories.
        """
        s = np.atleast_2d(np.asarray(states, dtype=np.float32))
        actions = np.asarray(action_sequences, dtype=np.float32)
        if actions.ndim == 1:
            actions = np.broadcast_to(actions, (len(s), len(actions)))
        if actions.ndim != 2 or actions.shape[0] != len(s):
            raise ValueError('Actions must be [B,H] or common [H]')
        h = np.asarray(history, dtype=np.float32)
        if h.ndim == 2:
            h = h[None]
        h = np.broadcast_to(h, (len(s), 6, 4))
        trajectories = []
        for member in self.members:
            current, hist, trajectory = s.copy(), h.copy(), []
            for t in range(actions.shape[1]):
                current = member.step(current, actions[:, t], hist)
                trajectory.append(current.copy())
                hist = np.concatenate((hist[:, 1:], current[:, None]), axis=1)
            trajectories.append(np.asarray(trajectory))
        members = np.asarray(trajectories)
        return members.mean(axis=0), members.std(axis=0, ddof=0), members


def validate_exports(models,data,output):
    import onnxruntime as ort
    from .models import EnsembleModel
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    # Held-out numerical consistency is a deployment check, no parameter selection.
    arrays=data['test'];indices=np.linspace(0,len(arrays['state'])-1,128,dtype=int)
    for name,model in models.items():
        if isinstance(model,EnsembleModel):
            manifest = export_ensemble(model, output / name)
            runtime = OnnxEnsemble(manifest)
            initial = arrays['state'][indices].copy()
            initial_history = arrays['history'][indices].copy()
            # Vary candidate actions across the entire horizon; no true future enters prediction.
            action_sequences = np.clip(arrays['action'][indices, None] +
                                       .8*np.sin(np.arange(30)[None, :]*.7), -6, 2)
            actual_mean, actual_std, actual_members = runtime.rollout(initial, action_sequences, initial_history)
            expected_members = []
            for member in model.members:
                state, hist, trajectory = initial.copy(), initial_history.copy(), []
                for t in range(30):
                    state = member.step(state, action_sequences[:, t], hist)
                    trajectory.append(state.copy())
                    hist = np.concatenate((hist[:, 1:], state[:, None]), axis=1)
                expected_members.append(trajectory)
            expected_members = np.asarray(expected_members)
            expected_mean = expected_members.mean(axis=0)
            expected_std = expected_members.std(axis=0, ddof=0)
            member_errors = np.max(np.abs(actual_members-expected_members), axis=(1, 2, 3))
            mean_error = float(np.max(np.abs(actual_mean-expected_mean)))
            std_error = float(np.max(np.abs(actual_std-expected_std)))
            maximum = max(float(member_errors.max()), mean_error, std_error)
            if maximum > 1e-3:
                raise AssertionError(f'{name} independent 30-step ensemble export mismatch {maximum}')
            rows.append(dict(model=name, batch=128, recursive_steps=30, max_abs_error=maximum,
                             member_max_abs_errors=member_errors.tolist(), mean_max_abs_error=mean_error,
                             std_max_abs_error=std_error, members=len(runtime.members),
                             manifest=str(manifest.relative_to(output)),
                             sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                             provider='host CPUExecutionProvider',
                             input_history='each member appends its own prediction; no mean-history sharing',
                             aggregation='arithmetic mean; population std ddof=0',
                             risk_scope='numerical consistency only; no risk calibration or safety claim'))
            continue
        path=export_onnx(model,output/f'{name}.onnx')
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        state=arrays['state'][indices].copy();ref=state.copy();history=arrays['history'][indices].copy();rh=history.copy()
        actions=arrays['action'][indices].copy()
        errors=[]
        for _ in range(30):
            ref=model.step(ref,actions,history=rh)
            state=session.run(None,{'state':state.astype(np.float32),'action':actions[:,None].astype(np.float32),'history':history.astype(np.float32)})[0]
            errors.append(float(np.max(np.abs(state-ref))))
            history=np.concatenate((history[:,1:],state[:,None]),axis=1)
            rh=np.concatenate((rh[:,1:],ref[:,None]),axis=1)
        if max(errors)>1e-3:
            raise AssertionError(f'{name} 30-step export mismatch {max(errors)}')
        rows.append(dict(model=name,batch=128,recursive_steps=30,max_abs_error=max(errors),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),provider='host CPUExecutionProvider',
                         input_history='[batch,6,4], oldest to newest; append own prediction during rollout'))
    (output/'validation.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    return rows
