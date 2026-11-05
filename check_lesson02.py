"""Lesson 02 gate: unfinished work is WAITING, never PASS."""

from pathlib import Path
import unittest

from falconworld.environment import step_environment
from falconworld.nominal import predict_one_step
from falconworld.state import LongitudinalState


def main() -> int:
    initial = LongitudinalState(20.0, 10.0, 8.0, 0.0)
    try:
        initial.as_tuple()
        predict_one_step(initial, -2.0)
        step_environment(initial, -2.0, -1.0)
    except NotImplementedError as exc:
        print(f"WAITING: {exc}")
        print("Complete lesson 01, then implement falconworld/environment.py.")
        return 2
    except Exception as exc:
        print(f"CHECK FAILED: {type(exc).__name__}: {exc}")
        return 1

    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(root / "tests"), pattern="test_lesson02.py"
    )
    if suite.countTestCases() == 0:
        print("CHECK FAILED: lesson 02 tests were not found.")
        return 1
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        print("Review the failing case and the old-state update convention.")
        return 1
    print("PASS: lesson 02 environment numerical checks passed.")
    print("Next: run python run_lesson02.py --scene constant")
    print("This checks a teaching simulator, not a trained world model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
