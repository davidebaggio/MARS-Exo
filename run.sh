#!/usr/bin/env bash
set -euo pipefail

#DEFAULT_BAG="data/rosbag2_2026_06_11-15_25_14/rosbag2_2026_06_11-15_25_14_0.mcap"
#DEFAULT_BAG="data/rosbag2_2026_06_11-15_31_00/rosbag2_2026_06_11-15_31_00_0.mcap"
DEFAULT_BAG="data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap"
#DEFAULT_BAG="data/exoskeleton_dataset_0_0_0/exoskeleton_dataset_0_0_0.mcap"
#DEFAULT_BAG="data/exoskeleton_dataset_1_20_35/exoskeleton_dataset_1_20_35.mcap"
#DEFAULT_BAG="data/exoskeleton_dataset_2_10_40/exoskeleton_dataset_2_10_40.mcap"

PLAYBACK_RATE=0.6
USE_IMU=true
PUBLISH_DEBUG_PCL=false
LOOP_PLAYBACK=false
BAG_PATH=""

usage() {
	echo "Usage: $0 [--rate RATE] [--imu] [--debug-pcl] [--loop] [bag_path]"
}

while (( $# > 0 )); do
	case "$1" in
		--rate)
			if (( $# < 2 )); then
				echo "--rate requires a value" >&2
				exit 2
			fi
			PLAYBACK_RATE="$2"
			shift 2
			;;
		--imu)
			USE_IMU=true
			shift
			;;
		--debug-pcl)
			PUBLISH_DEBUG_PCL=true
			shift
			;;
		--loop)
			LOOP_PLAYBACK=true
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		--*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 2
			;;
		*)
			if [[ -n "$BAG_PATH" ]]; then
				echo "Only one bag path may be provided" >&2
				exit 2
			fi
			BAG_PATH="$1"
			shift
			;;
	esac
done

if [[ ! "$PLAYBACK_RATE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
	! awk 'BEGIN { exit !(ARGV[1] > 0) }' "$PLAYBACK_RATE"; then
	echo "Playback rate must be a positive number: $PLAYBACK_RATE" >&2
	exit 2
fi

BAG_PATH="${BAG_PATH:-$DEFAULT_BAG}"
EXOSKELETON_DATASET=false
GT_LAUNCH_ARGS=()
EXO_PITCH_DEG=0
HEAD_PITCH_DEG=0
CLOUD_RECORD_PID=""
CLOUD_EVAL_BAG=""
CLOUD_METRICS_CSV=""
OFFLINE_COMMAND_PRINTED=false

if [[ ! -e "$BAG_PATH" ]]; then
	echo "Bag path not found: $BAG_PATH" >&2
	exit 1
fi
BAG_NAME="$(basename "$BAG_PATH" .mcap)"
SLIDING_WINDOW_SIZE="$(sed -nE 's/^[[:space:]]*sliding_window_size:[[:space:]]*([0-9]+).*/\1/p' config/common.yaml | head -n 1)"
if [[ -z "$SLIDING_WINDOW_SIZE" ]]; then
	echo "sliding_window_size not found in config/common.yaml" >&2
	exit 1
fi
if [[ "$BAG_NAME" =~ _(-?[0-9]+([.][0-9]+)?)_(-?[0-9]+([.][0-9]+)?)_(-?[0-9]+([.][0-9]+)?)$ ]]; then
	DATASET_NUMBERS="${BASH_REMATCH[1]}_${BASH_REMATCH[3]}_${BASH_REMATCH[5]}"
	DATASET_EXO_PITCH="${BASH_REMATCH[3]}"
	DATASET_HEAD_PITCH="${BASH_REMATCH[5]}"
else
	DATASET_NUMBERS="$BAG_NAME"
	DATASET_EXO_PITCH=""
	DATASET_HEAD_PITCH=""
fi
RUN_ID="${DATASET_NUMBERS}_${SLIDING_WINDOW_SIZE}"

if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /exoskeleton/odom | Type: nav_msgs/msg/Odometry'; then
	EXOSKELETON_DATASET=true
	GT_LAUNCH_ARGS=(gt_parent_frame:=gt_exo_link gt_child_frame:=gt_head_link)
	if [[ -n "$DATASET_EXO_PITCH" ]]; then
		EXO_PITCH_DEG="$DATASET_EXO_PITCH"
		HEAD_PITCH_DEG="$DATASET_HEAD_PITCH"
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
	if [[ -n "${CLOUD_RECORD_PID:-}" ]] && kill -0 "$CLOUD_RECORD_PID" 2>/dev/null; then
		kill -INT "$CLOUD_RECORD_PID" || true
		wait "$CLOUD_RECORD_PID" || true
		CLOUD_RECORD_PID=""
	fi
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
	if [[ -n "${CLOUD_EVAL_BAG:-}" && "$OFFLINE_COMMAND_PRINTED" == false ]]; then
		echo "Offline cloud evaluation command:"
		echo "source install/setup.bash && ros2 launch exo_head_slam cloud_map_evaluation_launch.py bag_path:=$CLOUD_EVAL_BAG metrics_csv_path:=$CLOUD_METRICS_CSV"
		OFFLINE_COMMAND_PRINTED=true
	fi
}

wait_for_pipeline() {
	local attempts=180
	while [[ "$attempts" -gt 0 ]]; do
		# The node name appears while VGGT is still loading. Its masked-image
		# subscription is created only after the model and pipeline are ready.
		if ros2 node info /extrinsic_solver 2>/dev/null | grep '/head/masked/image_raw' >/dev/null; then
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

mkdir -p metrics/pipeline
METRICS_CSV="metrics/pipeline/metrics_${RUN_ID}.csv"
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

if [[ "$USE_IMU" == true ]]; then
	if ! ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /camera/exo/imu | Type: sensor_msgs/msg/Imu'; then
		echo "--imu requested, but the bag has no /camera/exo/imu topic" >&2
		exit 1
	fi
	if ! ros2 pkg prefix imu_filter_madgwick >/dev/null 2>&1; then
		echo "--imu requested, but imu_filter_madgwick is not installed" >&2
		exit 1
	fi
fi
echo "RTAB-Map IMU leveling: $USE_IMU"

RECORD_CLOUD_MAP=false
if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /ground_truth/visible_cloud | Type: sensor_msgs/msg/PointCloud2'; then
	RECORD_CLOUD_MAP=true
	CLOUD_METRICS_CSV="${METRICS_CSV%.csv}_cloud.csv"
	CLOUD_EVAL_BAG="${METRICS_CSV%.csv}_cloud_bag"
	case "$CLOUD_EVAL_BAG" in
		metrics/pipeline/metrics_*_cloud_bag) rm -rf -- "$CLOUD_EVAL_BAG" ;;
		*) echo "Refusing to replace unexpected bag path: $CLOUD_EVAL_BAG" >&2; exit 1 ;;
	esac
	rm -f -- "$CLOUD_METRICS_CSV" "${CLOUD_METRICS_CSV%.csv}_global.csv" \
		"${CLOUD_METRICS_CSV%.csv}_global_maps.npz"
	echo "Recording cloud evaluation inputs to: $CLOUD_EVAL_BAG"
fi

echo "Playback rate: ${PLAYBACK_RATE}x"
echo "Loop playback: $LOOP_PLAYBACK"
echo "Debug point clouds: $PUBLISH_DEBUG_PCL"
echo "RTAB-Map input: /camera/exo/color/image_raw + /exo/filtered/depth_raw"

ros2 launch exo_head_slam main_pipeline_launch.py use_sim_time:=true publish_debug_pcl:="$PUBLISH_DEBUG_PCL" metrics_csv_path:="$METRICS_CSV" exo_pitch_deg:="$EXO_PITCH_DEG" head_pitch_deg:="$HEAD_PITCH_DEG" use_imu:="$USE_IMU" imu_topic:=/camera/exo/imu exoskeleton_dataset:="$EXOSKELETON_DATASET" "${GT_LAUNCH_ARGS[@]}" &
PIPELINE_PID=$!

if ! wait_for_pipeline; then
	echo "Pipeline did not become ready within 180 seconds" >&2
	exit 1
fi

if [[ "$RECORD_CLOUD_MAP" == true ]]; then
	ros2 bag record -s mcap -o "$CLOUD_EVAL_BAG" --disable-keyboard-controls \
		--custom-data "exo_pitch_deg=$EXO_PITCH_DEG" \
			"dataset_numbers=$DATASET_NUMBERS" \
			"sliding_window_size=$SLIDING_WINDOW_SIZE" --topics \
		/vggt/combined_pointcloud /ground_truth/visible_cloud /exoskeleton/odom &
	CLOUD_RECORD_PID=$!
	sleep 1
fi

echo "Starting bag playback: $BAG_PATH"
PLAY_REMAP=()
if [[ "$EXOSKELETON_DATASET" == true ]]; then
	# Avoid duplicate parents; the launch file republishes only the camera transforms it needs.
	PLAY_REMAP=(--remap /tf_static:=/recorded/tf_static)
fi
PLAY_ARGS=(-i "$BAG_PATH" mcap --rate "$PLAYBACK_RATE" --disable-keyboard-controls --clock)
if [[ "$LOOP_PLAYBACK" == true ]]; then
	PLAY_ARGS+=(--loop)
fi
ros2 bag play "${PLAY_ARGS[@]}" "${PLAY_REMAP[@]}" &
BAG_PID=$!

# Launch RViz with pre-configured displays
rviz2 -d "$(ros2 pkg prefix exo_head_slam)/share/exo_head_slam/rviz/pipeline.rviz" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!

if [[ "$LOOP_PLAYBACK" == true ]]; then
	wait "$PIPELINE_PID"
else
	BAG_STATUS=0
	wait "$BAG_PID" || BAG_STATUS=$?
	BAG_PID=""
	echo "Bag playback finished; draining pipeline for 5 seconds..."
	sleep 5
	exit "$BAG_STATUS"
fi
