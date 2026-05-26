#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BAG="$HOME/master_thesis/SLAM3R/data/exo/rosbag2_2026_05_06-16_48_23/rosbag2_2026_05_06-16_48_23_0.mcap"
BAG_PATH="${1:-$DEFAULT_BAG}"

cleanup() {
	if [[ -n "${BAG_PID:-}" ]] && kill -0 "$BAG_PID" 2>/dev/null; then
		kill "$BAG_PID" || true
	fi
	if [[ -n "${PIPELINE_PID:-}" ]] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
		kill "$PIPELINE_PID" || true
	fi
	pkill -f '/home/baggio/master_thesis/exo_head_slam/install/exo_head_slam/lib/exo_head_slam/(semantic_masker|extrinsic_solver)' || true
}

wait_for_pipeline() {
	local attempts=30
	while [[ "$attempts" -gt 0 ]]; do
		if ros2 node list 2>/dev/null | grep -qE '(/head_semantic_masker|/exo_semantic_masker|/extrinsic_solver)'; then
			return 0
		fi
		attempts=$((attempts - 1))
		sleep 1
	done
	return 1
}

trap cleanup EXIT INT TERM

cleanup
colcon build --packages-select exo_head_slam --symlink-install
set +u
source install/setup.bash
set -u

# Ensure the generated shim scripts use the active python (conda env)
# rather than hardcoding /usr/bin/python3
if [ -d "install/exo_head_slam/lib/exo_head_slam" ]; then
    sed -i "1s|^#!.*python.*|#!$(which python3)|" install/exo_head_slam/lib/exo_head_slam/*
fi

ros2 launch exo_head_slam main_pipeline_launch.py &
PIPELINE_PID=$!

wait_for_pipeline

echo "Starting bag playback: $BAG_PATH"
ros2 bag play -i "$BAG_PATH" mcap --loop --rate 0.1 --disable-keyboard-controls > /tmp/exo_head_slam_bag.log 2>&1 &
BAG_PID=$!

wait "$PIPELINE_PID"