#!/usr/bin/env bash
set -euo pipefail

#DEFAULT_BAG="data/rosbag2_2026_06_11-15_25_14/rosbag2_2026_06_11-15_25_14_0.mcap"
#DEFAULT_BAG="data/rosbag2_2026_06_11-15_31_00/rosbag2_2026_06_11-15_31_00_0.mcap"
DEFAULT_BAG="data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap"
BAG_PATH="${1:-$DEFAULT_BAG}"

if [[ ! -e "$BAG_PATH" ]]; then
	echo "Bag path not found: $BAG_PATH" >&2
	exit 1
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

USE_IMU=false
if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q 'Topic: /camera/exo/imu | Type: sensor_msgs/msg/Imu'; then
	if ros2 pkg prefix imu_filter_madgwick >/dev/null 2>&1; then
		USE_IMU=true
	else
		echo "Warning: bag contains IMU data, but imu_filter_madgwick is missing; continuing without IMU." >&2
	fi
fi
echo "RTAB-Map IMU leveling: $USE_IMU"

ros2 launch exo_head_slam main_pipeline_launch.py use_sim_time:=true publish_debug_pcl:=true metrics_csv_path:="$METRICS_CSV" use_imu:="$USE_IMU" imu_topic:=/camera/exo/imu &
PIPELINE_PID=$!

wait_for_pipeline

echo "Starting bag playback: $BAG_PATH"
ros2 bag play -i "$BAG_PATH" mcap --loop --rate 0.3 --disable-keyboard-controls --clock &
     #--remap /tf:=/tf_old /tf_static:=/tf_static_old &
BAG_PID=$!

# Launch RViz with pre-configured displays
rviz2 -d "$(ros2 pkg prefix exo_head_slam)/share/exo_head_slam/rviz/pipeline.rviz" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!

wait "$PIPELINE_PID"
