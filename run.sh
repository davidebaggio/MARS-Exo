#!/usr/bin/env bash
set -euo pipefail

#DEFAULT_BAG="data/rosbag2_2026_06_11-15_25_14/rosbag2_2026_06_11-15_25_14_0.mcap"
#DEFAULT_BAG="data/rosbag2_2026_06_11-15_31_00/rosbag2_2026_06_11-15_31_00_0.mcap"
#DEFAULT_BAG="data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap"
#DEFAULT_BAG="data/exoskeleton_dataset_0_0_0/exoskeleton_dataset_0_0_0.mcap"
#DEFAULT_BAG="data/exoskeleton_dataset_1_20_35/exoskeleton_dataset_1_20_35.mcap"
DEFAULT_BAG="data/exoskeleton_dataset_2_10_40/exoskeleton_dataset_2_10_40.mcap"

BAG_PATH="${1:-$DEFAULT_BAG}"
EXOSKELETON_DATASET=false
GT_LAUNCH_ARGS=()
EXO_PITCH_DEG=0
HEAD_PITCH_DEG=0

if [[ ! -e "$BAG_PATH" ]]; then
	echo "Bag path not found: $BAG_PATH" >&2
	exit 1
fi
if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /exoskeleton/odom | Type: nav_msgs/msg/Odometry'; then
	EXOSKELETON_DATASET=true
	GT_LAUNCH_ARGS=(gt_parent_frame:=gt_exo_link gt_child_frame:=gt_head_link)
	BAG_NAME="$(basename "$BAG_PATH" .mcap)"
	if [[ "$BAG_NAME" =~ ^exoskeleton_dataset_[^_]+_(-?[0-9]+([.][0-9]+)?)_(-?[0-9]+([.][0-9]+)?)$ ]]; then
		EXO_PITCH_DEG="${BASH_REMATCH[1]}"
		HEAD_PITCH_DEG="${BASH_REMATCH[3]}"
	else
		echo "Warning: cannot parse camera pitches from $BAG_NAME; using 0/0 degrees." >&2
	fi
fi
if [[ ! -f vggt-omega/vggt_omega/models/vggt_omega.py ]]; then
	echo "VGGT-Omega submodule is not initialized. Run: git submodule update --init --recursive" >&2
	exit 1
fi
if [[ ! -f vggt_omega_1b_512.pt ]]; then
	echo "VGGT-Omega checkpoint not found: vggt_omega_1b_512.pt" >&2
	exit 1
fi
if ! python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
	echo "CUDA-enabled PyTorch is required by VGGT-Omega." >&2
	exit 1
fi
if [[ ! -f yolov8n-seg.pt ]]; then
	echo "Warning: yolov8n-seg.pt missing; semantic masking will pass images through." >&2
fi

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
	pkill -f 'cloud_map_evaluator' || true
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

MISSING_RTABMAP_PACKAGES=()
for package in rtabmap_slam rtabmap_odom rtabmap_util; do
	if ! ros2 pkg prefix "$package" >/dev/null 2>&1; then
		MISSING_RTABMAP_PACKAGES+=("$package")
	fi
done
if (( ${#MISSING_RTABMAP_PACKAGES[@]} > 0 )); then
	echo "Warning: RTAB-Map disabled; missing packages: ${MISSING_RTABMAP_PACKAGES[*]}" >&2
fi

# Ensure the generated shim scripts use the active python (conda env)
# rather than hardcoding /usr/bin/python3
if [ -d "install/exo_head_slam/lib/exo_head_slam" ]; then
    sed -i "1s|^#!.*python.*|#!$(which python3)|" install/exo_head_slam/lib/exo_head_slam/*
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p metrics/pipeline
METRICS_CSV="metrics/pipeline/metrics_${TIMESTAMP}.csv"
<<<<<<< HEAD
BENCHMARK_PREFIX="metrics/eval/benchmark_${TIMESTAMP}"
=======
CLOUD_METRICS_CSV="${METRICS_CSV%.csv}_cloud.csv"
>>>>>>> 8a94f28
echo "Logging metrics to: $METRICS_CSV"

# fastcdr 2.2.5 needs the local compatibility shim. Current Jazzy releases do not.
FASTCDR_VERSION="$(dpkg-query -W -f='${Version}' "ros-${ROS_DISTRO:-jazzy}-fastcdr" 2>/dev/null || true)"
FASTCDR_VERSION="${FASTCDR_VERSION%%-*}"
if [[ -n "$FASTCDR_VERSION" ]] && dpkg --compare-versions "$FASTCDR_VERSION" lt 2.2.7; then
	if [[ ! -f lib/libfastcdr_compat.so ]]; then
		echo "Fast-CDR $FASTCDR_VERSION requires lib/libfastcdr_compat.so, but it is missing." >&2
		exit 1
	fi
	export LD_PRELOAD="$(realpath lib/libfastcdr_compat.so)${LD_PRELOAD:+:$LD_PRELOAD}"
fi

BAG_INFO="$(ros2 bag info "$BAG_PATH" 2>/dev/null)"
USE_IMU=false
<<<<<<< HEAD
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
=======
if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /camera/exo/imu | Type: sensor_msgs/msg/Imu'; then
	if ros2 pkg prefix imu_filter_madgwick >/dev/null 2>&1; then
		USE_IMU=true
	else
		echo "Warning: bag contains IMU data, but imu_filter_madgwick is missing; continuing without IMU." >&2
	fi
fi
echo "RTAB-Map IMU leveling: $USE_IMU"

EVALUATE_CLOUD_MAP=false
if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /ground_truth/visible_cloud | Type: sensor_msgs/msg/PointCloud2'; then
	EVALUATE_CLOUD_MAP=true
	echo "Logging visible-cloud evaluation to: $CLOUD_METRICS_CSV"
fi

ros2 launch exo_head_slam main_pipeline_launch.py use_sim_time:=true publish_debug_pcl:=true metrics_csv_path:="$METRICS_CSV" cloud_metrics_csv_path:="$CLOUD_METRICS_CSV" evaluate_cloud_map:="$EVALUATE_CLOUD_MAP" exo_pitch_deg:="$EXO_PITCH_DEG" head_pitch_deg:="$HEAD_PITCH_DEG" use_imu:="$USE_IMU" imu_topic:=/camera/exo/imu exoskeleton_dataset:="$EXOSKELETON_DATASET" "${GT_LAUNCH_ARGS[@]}" &
>>>>>>> 8a94f28
PIPELINE_PID=$!

wait_for_pipeline

echo "Starting bag playback: $BAG_PATH"
<<<<<<< HEAD
PLAY_ARGS=(-i "$BAG_PATH" mcap --rate 0.3 --disable-keyboard-controls --clock)
if [[ "$DATASET_MODE" == true ]]; then
	PLAY_ARGS+=(--remap /tf:=/ground_truth/tf /tf_static:=/ground_truth/tf_static)
else
	PLAY_ARGS+=(--loop)
fi
ros2 bag play "${PLAY_ARGS[@]}" &
=======
PLAY_REMAP=()
if [[ "$EXOSKELETON_DATASET" == true ]]; then
	# Avoid duplicate parents; the launch file republishes only the camera transforms it needs.
	PLAY_REMAP=(--remap /tf_static:=/recorded/tf_static)
fi
ros2 bag play -i "$BAG_PATH" mcap --loop --rate 0.3 --disable-keyboard-controls --clock "${PLAY_REMAP[@]}" &
     #--remap /tf:=/tf_old /tf_static:=/tf_static_old &
>>>>>>> 8a94f28
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
