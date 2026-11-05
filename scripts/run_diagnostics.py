"""Run frozen longitudinal diagnostics without changing training/model selection."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from falconworld.diagnostics import run_diagnostics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/smoke/data")
    parser.add_argument("--models", default="outputs/smoke/seed0/models")
    parser.add_argument("--output", default="outputs/diagnostics")
    args = parser.parse_args()
    result = run_diagnostics(args.data, args.models, args.output)
    print(f"Diagnostics saved to {Path(args.output).resolve()}")
    print(f"Dataset {result['data_sha256']}; statistical unit: episode")
