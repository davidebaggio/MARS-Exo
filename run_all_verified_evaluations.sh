#!/usr/bin/env bash
# Run every compatible ROS 2 dataset once, headlessly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="metrics/lightglue/final_evals"

usage() {
	echo "Usage: $0 [--rate RATE]"
}

RATE="0.6"
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

shopt -s globstar nullglob
DATASETS=()
for metadata in data/**/metadata.yaml; do
	bag="${metadata%/metadata.yaml}"
	grep -Eq 'name: /camera/(exo/color/image_raw|rgb/image_color)' "$metadata" \
		&& DATASETS+=("$bag")
done
(( ${#DATASETS[@]} > 0 )) || { echo "No compatible datasets found." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
for bag in "${DATASETS[@]}"; do
	dataset="$(basename "$bag")"
	dataset="${dataset%.bag}"
	echo "===== pipeline: dataset=${dataset} ====="
	args=(--headless --no-build --metrics-dir "$OUTPUT_DIR" --eval-dir "$OUTPUT_DIR" --rate "$RATE")
	./run.sh "${args[@]}" "$bag"
done

echo "All ${#DATASETS[@]} evaluations completed."
echo "Results: $ROOT/$OUTPUT_DIR"
