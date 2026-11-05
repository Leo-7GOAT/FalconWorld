"""Low-speed, observable-only deployment model. Training origin stays explicit.

State: [current speed, previous speed, previous SENT speed reference] [m/s].
Action: next speed reference [m/s], not acceleration or PWM. One simulated
sample of command delay is represented by the previous reference in the state.
No physical actuator state or leader measurement is needed for speed tracking.
"""
from pathlib import Path
import json
import numpy as np
from .planning import CEMConfig,CEMOptimizer

DT=.05


def physical(states,commands,parameters):
    s=np.atleast_2d(states)
    up,down,speed_factor=parameters
    tau=np.where(s[:,2]>=s[:,0],up,down)*(1+speed_factor*s[:,0])
    v=np.clip(s[:,0]+DT*(s[:,2]-s[:,0])/tau,0,.35)
    return np.column_stack([v,s[:,0],np.asarray(commands).reshape(-1)])


class LowSpeedModel:
    def __init__(self,parameters,arrays=None,origin='SYNTHETIC_UNCALIBRATED'):
        self.parameters=np.asarray(parameters)
        self.arrays=arrays
        self.origin=origin

    def step(self,states,commands):
        state=np.asarray(states,dtype=float)
        single=state.ndim==1
        s=np.atleast_2d(state)
        u=np.broadcast_to(np.asarray(commands).reshape(-1),(len(s),))
        if s.shape[1]!=3 or not np.isfinite(s).all() or not np.isfinite(u).all():
            raise ValueError('Need finite speed, previous speed, previous speed command')
        if np.any(s<0) or np.any(s>.35) or np.any(u<0) or np.any(u>.3):
            raise ValueError('Low-speed deployment model domain is [0,.35] state, [0,.3] command m/s')
        result=physical(s,u,self.parameters)
        if self.arrays is not None:
            a=self.arrays
            x=(np.column_stack([s,u])-a['xmean'])/a['xscale']
            residual=(np.tanh(x@a['w0']+a['w1'])@a['w2']+a['w3'])*a['yscale']+a['ymean']
            result[:,0]=np.clip(result[:,0]+residual[:,0],0,.35)
        return result[0] if single else result

    def save(self,path):
        values=dict(parameters=self.parameters,origin=np.asarray(self.origin))
        if self.arrays is not None:values.update(self.arrays)
        np.savez(path,**values)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in saved.files if k not in ('parameters','origin')}
            return cls(saved['parameters'].copy(),arrays or None,str(saved['origin']))


class LowSpeedMPC:
    def __init__(self,model,seed=0,candidates=48,horizon=20):
        self.model,self.seed,self.tick=model,seed,0
        self.optimizer=CEMOptimizer(CEMConfig(horizon=horizon,candidates=candidates,iterations=3,elites=8,dt=DT))

    def plan(self,state,target):
        state=np.asarray(state,dtype=float)
        if not 0<=target<=.3:raise ValueError('Target speed must be [0,.3]m/s')
        def objective(sequences):
            states=np.repeat(state[None],len(sequences),axis=0)
            cost=np.zeros(len(sequences));last=np.full(len(sequences),state[2])
            for t in range(sequences.shape[1]):
                u=sequences[:,t,0]
                states=self.model.step(states,u)
                cost+=(states[:,0]-target)**2+.06*(u-last)**2
                last=u
            return cost
        sequence,info=self.optimizer.optimize(objective,[0],[.3],previous=[state[2]],
            seed=self.seed+self.tick,slew=.015)
        self.tick+=1
        if sequence is None:return 0.,dict(info,fallback=True,reason='solver_failure')
        return float(sequence[0,0]),dict(info,fallback=False,reason='')


def export_onnx(model,path):
    import onnx
    from onnx import TensorProto as T,helper as h,numpy_helper
    nodes=[];initializers=[];counter=0
    def const(value):
        nonlocal counter
        name='c%d'%counter;counter+=1
        initializers.append(numpy_helper.from_array(np.asarray(value,dtype=np.float32),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal counter
        name='n%d'%counter;counter+=1;nodes.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    def col(i):return op('Gather','state',const_int([i]),axis=1)
    def const_int(value):
        nonlocal counter
        name='c%d'%counter;counter+=1
        initializers.append(numpy_helper.from_array(np.asarray(value,dtype=np.int64),name));return name
    v,previous_command=col(0),col(2)
    tau=op('Where',op('GreaterOrEqual',previous_command,v),const(model.parameters[0]),const(model.parameters[1]))
    tau=op('Mul',tau,op('Add',const(1),op('Mul',const(model.parameters[2]),v)))
    next_v=op('Clip',op('Add',v,op('Div',op('Mul',const(DT),op('Sub',previous_command,v)),tau)),const(0),const(.35))
    if model.arrays is not None:
        a=model.arrays
        x=op('Div',op('Sub',op('Concat','state','command',axis=1),const(a['xmean'])),const(a['xscale']))
        hidden=op('Tanh',op('Add',op('MatMul',x,const(a['w0'])),const(a['w1'])))
        residual=op('Add',op('Mul',op('Add',op('MatMul',hidden,const(a['w2'])),const(a['w3'])),const(a['yscale'])),const(a['ymean']))
        next_v=op('Clip',op('Add',next_v,residual),const(0),const(.35))
    nodes.append(h.make_node('Concat',[next_v,v,'command'],['next_state'],axis=1))
    graph=h.make_graph(nodes,'FalconWorldLowSpeed',[
        h.make_tensor_value_info('state',T.FLOAT,[None,3]),h.make_tensor_value_info('command',T.FLOAT,[None,1])],
        [h.make_tensor_value_info('next_state',T.FLOAT,[None,3])],initializer=initializers)
    artifact=h.make_model(graph,opset_imports=[h.make_opsetid('',13)],ir_version=8)
    artifact.metadata_props.add(key='training_origin',value=model.origin)
    artifact.metadata_props.add(key='units',value='state/command m/s; dt .05s; previous command is SENT reference not MCU ACK')
    onnx.checker.check_model(artifact);onnx.save(artifact,str(path))
    return path
