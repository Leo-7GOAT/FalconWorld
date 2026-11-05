"""Read-only, standard-library-only environment report. No network or installs."""

import json
import platform
import sys
from importlib import metadata


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed (not needed for lesson 01)"


def main():
    ready = sys.version_info >= (3, 10)
    report = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "system": platform.system(),
        "architecture": platform.machine(),
        "wsl": "microsoft" in platform.release().lower(),
        "lesson01_ready": ready,
        "numpy": installed_version("numpy"),
        "torch": installed_version("torch"),
        "note": "Lesson 01 uses only Python standard library. No GPU or CARLA needed.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not ready:
        print("Please use Python 3.10 or newer for this teaching package.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
