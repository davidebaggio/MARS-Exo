#!/usr/bin/env bash
set -euo pipefail

rm -f ~/.ros/*.db

#DEFAULT_BAG="data/rosbag2_2026_06_11-15_25_14/rosbag2_2026_06_11-15_25_14_0.mcap"
#DEFAULT_BAG="data/rosbag2_2026_06_11-15_31_00/rosbag2_2026_06_11-15_31_00_0.mcap"
DEFAULT_BAG="data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap"
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
	pkill -f 'orbslam3_rgbd_imu' || true
	pkill -f 'dense_global_map' || true
	pkill -f 'ros2 bag play' || true
}

wait_for_pipeline() {
	local attempts=30
	while [[ "$attempts" -gt 0 ]]; do
		if ros2 node list 2>/dev/null | grep -qE '(/orbslam3_exo|/dense_global_map_accumulator|/extrinsic_solver)'; then
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

ros2 launch exo_head_slam main_pipeline_launch.py \
	use_sim_time:=true \
	publish_debug_pcl:=true \
	imu_topic:=/camera/exo/imu \
	orbslam3_vocabulary_path:=third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
	orbslam3_settings_path:=config/orbslam3_exo.yaml \
	map_start_z:=1 \
	dense_map_voxel_size:=0.03 \
	dense_map_max_points:=250000 \
	dense_map_downsample_factor:=2 \
	dense_map_min_depth:=0.1 \
	dense_map_max_depth:=10.0 &
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
