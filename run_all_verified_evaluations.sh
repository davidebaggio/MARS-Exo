#!/usr/bin/env bash
# Run every dataset/window configuration headlessly, then evaluate all saved VGGT clouds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="metrics/batch_windows/pipeline"
EVAL_DIR="metrics/batch_windows/eval"
DATASETS=(0_0_0 1_20_35 2_10_40)
WINDOWS=(1 2 4 6)

usage() {
	echo "Usage: $0 [--rate RATE]"
}

RATE=""
while (( $# > 0 )); do
	case "$1" in
		--rate)
			[[ $# -ge 2 ]] || { usage >&2; exit 2; }
			RATE="$2"
			shift 2
			;;
		-h|--help) usage; exit 0 ;;
		*) usage >&2; exit 2 ;;
	esac
done

cd "$ROOT"
set +u
source /opt/ros/jazzy/setup.bash
set -u
make build

for dataset in "${DATASETS[@]}"; do
	bag="$ROOT/data/exoskeleton_dataset_${dataset}/exoskeleton_dataset_${dataset}.mcap"
	[[ -f "$bag" ]] || { echo "Bag not found: $bag" >&2; exit 1; }
	for window in "${WINDOWS[@]}"; do
		echo "===== pipeline: dataset=${dataset}, window=${window} ====="
		args=(--headless --no-build --window "$window" --metrics-dir "$PIPELINE_DIR" --eval-dir "$EVAL_DIR")
		[[ -z "$RATE" ]] || args+=(--rate "$RATE")
		./run.sh "${args[@]}" "$bag"
	done
done

set +u
source install/setup.bash
set -u
for dataset in "${DATASETS[@]}"; do
	for window in "${WINDOWS[@]}"; do
		name="metrics_${dataset}_${window}"
		cloud_bag="$PIPELINE_DIR/${name}_cloud_bag"
		cloud_csv="$PIPELINE_DIR/${name}_cloud.csv"
		[[ -f "$cloud_bag/metadata.yaml" ]] || { echo "Cloud bag missing: $cloud_bag" >&2; exit 1; }
		echo "===== cloud evaluation: dataset=${dataset}, window=${window} ====="
		ros2 launch exo_head_slam cloud_map_evaluation_launch.py \
			bag_path:="$cloud_bag" metrics_csv_path:="$cloud_csv" \
			playback_rate:=3.0 global:=true
		EVAL_DIR="$EVAL_DIR" python3 plot_metrics.py "$PIPELINE_DIR/${name}.csv"
		EVAL_DIR="$EVAL_DIR" python3 plot_metrics.py "${cloud_csv%.csv}_global.csv"
	done
done

echo "All 12 pipeline and cloud evaluations completed."
echo "Pipeline CSVs: $ROOT/$PIPELINE_DIR"
echo "Reports and plots: $ROOT/$EVAL_DIR"
