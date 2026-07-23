#!/usr/bin/env bash
set -euo pipefail

rm -f ~/.ros/*.db

#DEFAULT_BAG="data/rosbag2_2026_06_11-15_25_14/rosbag2_2026_06_11-15_25_14_0.mcap"
#DEFAULT_BAG="data/rosbag2_2026_06_11-15_31_00/rosbag2_2026_06_11-15_31_00_0.mcap"
DEFAULT_BAG="data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap"

#DEFAULT_BAG="data/exoskeleton_dataset/exoskeleton_dataset_0.mcap"
BAG_PATH="${1:-$DEFAULT_BAG}"

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

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p metrics/pipeline
METRICS_CSV="metrics/pipeline/metrics_${TIMESTAMP}.csv"
BENCHMARK_PREFIX="metrics/eval/benchmark_${TIMESTAMP}"
echo "Logging metrics to: $METRICS_CSV"

# Preload fastcdr compat shim to provide missing serialize(unsigned int) symbol
# that rtabmap_msgs needs but fastcdr 2.2.5 lacks (needs 2.2.7+).
export LD_PRELOAD="$(realpath lib/libfastcdr_compat.so)${LD_PRELOAD:+:$LD_PRELOAD}"

BAG_INFO="$(ros2 bag info "$BAG_PATH" 2>/dev/null)"
USE_IMU=false
if grep -q 'Topic: /camera/exo/imu | Type: sensor_msgs/msg/Imu' <<<"$BAG_INFO"; then
	USE_IMU=true
fi
echo "RTAB-Map IMU leveling: $USE_IMU"

DATASET_MODE=false
DEPTH_UNIT_SCALE=0.001
EVALUATION_ENABLED=false
GT_PARENT_FRAME=""
GT_CHILD_FRAME=""
GT_TF_STATIC_TOPIC=""
if grep -q 'Topic: /ground_truth/global_map' <<<"$BAG_INFO"; then
	DATASET_MODE=true
	DEPTH_UNIT_SCALE=1.0
	EVALUATION_ENABLED=true
	GT_PARENT_FRAME="front_camera_link"
	GT_CHILD_FRAME="head_camera_link"
	GT_TF_STATIC_TOPIC="/ground_truth/tf_static"
	echo "Detected exoskeleton_dataset: enabling GT benchmark"
fi

LAUNCH_ARGS=(
	use_sim_time:=true
	publish_debug_pcl:=true
	metrics_csv_path:="$METRICS_CSV"
	benchmark_output_prefix:="$BENCHMARK_PREFIX"
	dataset_mode:="$DATASET_MODE"
	depth_unit_scale:="$DEPTH_UNIT_SCALE"
	evaluation_enabled:="$EVALUATION_ENABLED"
	use_imu:="$USE_IMU"
	imu_topic:=/camera/exo/imu
)
if [[ "$DATASET_MODE" == true ]]; then
	LAUNCH_ARGS+=(
		gt_parent_frame:="$GT_PARENT_FRAME"
		gt_child_frame:="$GT_CHILD_FRAME"
		gt_tf_static_topic:="$GT_TF_STATIC_TOPIC"
	)
fi
ros2 launch exo_head_slam main_pipeline_launch.py "${LAUNCH_ARGS[@]}" &
PIPELINE_PID=$!

wait_for_pipeline

echo "Starting bag playback: $BAG_PATH"
PLAY_ARGS=(-i "$BAG_PATH" mcap --rate 0.3 --disable-keyboard-controls --clock)
if [[ "$DATASET_MODE" == true ]]; then
	PLAY_ARGS+=(--remap /tf:=/ground_truth/tf /tf_static:=/ground_truth/tf_static)
else
	PLAY_ARGS+=(--loop)
fi
ros2 bag play "${PLAY_ARGS[@]}" &
BAG_PID=$!

# Launch RViz with pre-configured displays
rviz2 -d "$(ros2 pkg prefix exo_head_slam)/share/exo_head_slam/rviz/pipeline.rviz" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!

if [[ "$DATASET_MODE" == true ]]; then
	wait "$BAG_PID"
	sleep 5
	timeout 15 ros2 service call \
		/benchmark_evaluator/finalize std_srvs/srv/Trigger "{}"
	python3 plot_metrics.py "$METRICS_CSV"
	echo "Benchmark summary: ${BENCHMARK_PREFIX}.json"
	exit 0
fi

wait "$PIPELINE_PID"
