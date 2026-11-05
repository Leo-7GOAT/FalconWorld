"""Verify saved low-speed NPZ/ONNX over true recursive history updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.r550_worldmodel import DT,LowSpeedModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='outputs/r550/speed_model');args=p.parse_args();root=Path(args.input)
    model=LowSpeedModel.load(root/'residual_worldmodel.npz')
    session=ort.InferenceSession(str(root/'residual_worldmodel.onnx'),providers=['CPUExecutionProvider'])
    with np.load(root/'test_data.npz',allow_pickle=False) as data:
        states,commands=data['states'].copy(),data['commands'].copy()
    output=session.run(None,dict(state=states.astype(np.float32),command=commands[:,None].astype(np.float32)))[0]
    single=float(np.max(np.abs(output-model.step(states,commands))))
    starts=np.arange(0,len(states),240);reference=states[starts];deployed=reference.astype(np.float32)
    maximum=0.
    for step in range(round(3/DT)):
        command=commands[starts+step]
        reference=model.step(reference,command)
        deployed=session.run(None,dict(state=deployed,command=command[:,None].astype(np.float32)))[0]
        maximum=max(maximum,float(np.max(np.abs(reference-deployed))))
    if max(single,maximum)>1e-6:raise AssertionError('ONNX recurrent history mismatch')
    result=dict(one_step_max_abs_difference_m_s=single,
        recursive_3s_max_abs_difference_m_s=maximum,episodes=len(starts),threshold_m_s=1e-6,
        model_origin=model.origin,provider=session.get_providers(),host_only=True)
    (root/'onnx_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
