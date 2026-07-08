#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-exo_head_slam}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Using existing conda env: $ENV_NAME"
else
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

if [[ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]]; then
    # ROS Python packages are provided by the system ROS install, not pip.
    # Source before building so colcon sees ament/launch/rclpy packages.
    set +u
    source "/opt/ros/$ROS_DISTRO/setup.bash"
    set -u
else
    echo "Warning: /opt/ros/$ROS_DISTRO/setup.bash not found; install/source ROS 2 before running nodes." >&2
fi

python -m pip install --upgrade pip "setuptools==70.2.0" "wheel"

cd "$REPO_DIR"

python -m pip install \
    "numpy==1.26.1" \
    "opencv-python==4.9.0.80" \
    "opencv-contrib-python==4.9.0.80" \
    "scipy" \
    "pandas" \
    "matplotlib" \
    "pyyaml" \
    "ultralytics" \
    "open3d" \
    "pyquaternion" \
    "transforms3d" \
    "nvitop"

python -m pip install \
    "torch==2.3.1" \
    "torchvision==0.18.1"

python -m pip install \
    "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8"

python -m pip install -e .

make build

echo
echo "Setup complete."
echo "Run:"
echo "  conda activate $ENV_NAME"
echo "  source install/setup.bash"
