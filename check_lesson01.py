"""Run the lesson gate; unfinished exercises are reported as WAITING, not PASS."""

from pathlib import Path
import unittest

from falconworld.nominal import predict_one_step
from falconworld.state import LongitudinalState


def main() -> int:
    initial = LongitudinalState(20.0, 10.0, 8.0, 0.0)
    try:
        initial.as_tuple()
        predict_one_step(initial, -2.0)
    except NotImplementedError as exc:
        print(f"WAITING: {exc}")
        print("Open docs/lesson01.md, complete the two exercises, then run again.")
        print("This is an unfinished exercise, not a completed world model.")
        return 2
    except Exception as exc:
        print(f"CHECK FAILED: {type(exc).__name__}: {exc}")
        return 1

    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(root / "tests"), pattern="test_lesson01.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        print("Review the failing case and the old-state update convention.")
        return 1

    print("PASS: lesson 01 numerical checks passed.")
    print("Legacy numerical compatibility verified; full research entry: reproduce.py.")
    print("This gate checks the nominal predictor, not a trained world model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
