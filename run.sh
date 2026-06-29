#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BAG="data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap"
EVAL_IMU="false"
BAG_PATH="$DEFAULT_BAG"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-imu) EVAL_IMU="true"; shift ;;
        *) BAG_PATH="$1"; shift ;;
    esac
done

cleanup() {
	if [[ -n "${BAG_PID:-}" ]] && kill -0 "$BAG_PID" 2>/dev/null; then
		kill "$BAG_PID" || true
	fi
	if [[ -n "${PIPELINE_PID:-}" ]] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
		kill "$PIPELINE_PID" || true
	fi
	if [[ -n "${RVIZ_PID:-}" ]] && kill -0 "$RVIZ_PID" 2>/dev/null; then
		kill "$RVIZ_PID" || true
	fi
	# Kill all nodes by executable name to ensure no zombies remain
	pkill -f 'depth_preprocessor' || true
	pkill -f 'semantic_masker' || true
	pkill -f 'extrinsic_solver' || true
	pkill -f 'ros2 bag play' || true
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
make build
set +u
source install/setup.bash
set -u

# Ensure the generated shim scripts use the active python (conda env)
# rather than hardcoding /usr/bin/python3
if [ -d "install/exo_head_slam/lib/exo_head_slam" ]; then
    sed -i "1s|^#!.*python.*|#!$(which python3)|" install/exo_head_slam/lib/exo_head_slam/*
fi

ros2 launch exo_head_slam main_pipeline_launch.py use_sim_time:=true publish_debug_pcl:=true global_frame:=odom &
PIPELINE_PID=$!

wait_for_pipeline

echo "Starting bag playback: $BAG_PATH"
ros2 bag play -s mcap "$BAG_PATH" --loop --rate 0.3 --disable-keyboard-controls --clock &
     #--remap /tf:=/tf_old /tf_static:=/tf_static_old &
BAG_PID=$!

# Generate evaluation plots after pipeline exits
eval_imu() {
    echo "Generating evaluation plots..."
    python3 "$(dirname "$0")/plot_metrics.py"
}

# Launch RViz with pre-configured displays
rviz2 -d "$(ros2 pkg prefix exo_head_slam)/share/exo_head_slam/rviz/pipeline.rviz" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!

wait "$PIPELINE_PID"

# Evaluate IMU metrics if flag set
if [[ "${EVAL_IMU:-false}" == "true" ]]; then
    eval_imu
fi