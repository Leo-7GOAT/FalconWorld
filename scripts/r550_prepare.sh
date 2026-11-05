#!/usr/bin/env bash
set -euo pipefail
# Non-motion Raspberry Pi 4B / ARM64 / Ubuntu22.04 preparation and probe.
cd "$(dirname "$0")/.."
python3 -c 'import platform,sys; print(platform.platform(),platform.machine(),sys.version); assert platform.machine() in ("aarch64","arm64"), "Expected ARM64 target"'
test -f /opt/ros/humble/setup.bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv-r550
.venv-r550/bin/python -m pip install -r requirements-deploy.txt
.venv-r550/bin/python -c 'import numpy,onnxruntime,rclpy; print("ORT",onnxruntime.__version__,onnxruntime.get_available_providers())'
ros2 topic list
echo 'Prepared only. No drive publisher was launched. Fill measured R550 config, then run scripts/r550_ros2_shadow.py.'
