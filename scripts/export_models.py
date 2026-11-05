import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from falconworld.models import load_models
from falconworld.data import load_dataset
from falconworld.export import validate_exports

p=argparse.ArgumentParser()
p.add_argument('--models',type=Path,default=Path('outputs/smoke/seed0/models'))
p.add_argument('--data',type=Path,default=Path('outputs/smoke/data'))
p.add_argument('--output',type=Path,default=Path('outputs/deployment/onnx'))
a=p.parse_args()
print(validate_exports(load_models(a.models),load_dataset(a.data),a.output))
