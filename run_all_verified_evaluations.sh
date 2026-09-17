#!/usr/bin/env bash
# Run every ground-truth dataset/window configuration headlessly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="metrics/final_evals"
WINDOWS=(1 2 4 6)

usage() {
	echo "Usage: $0 [--rate RATE]"
}

RATE="0.08"
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
	#grep -Eq 'name: /(ground_truth/odom|exoskeleton/odom)' "$metadata" && DATASETS+=("$bag")
	grep -Eq 'name: /(ground_truth/odom)' "$metadata" && DATASETS+=("$bag")
done
(( ${#DATASETS[@]} > 0 )) || { echo "No ground-truth datasets found." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
for bag in "${DATASETS[@]}"; do
	dataset="$(basename "$bag")"
	dataset="${dataset%.bag}"
	for window in "${WINDOWS[@]}"; do
		echo "===== pipeline: dataset=${dataset}, window=${window} ====="
		args=(--headless --no-build --window "$window" --metrics-dir "$OUTPUT_DIR" --eval-dir "$OUTPUT_DIR" --rate "$RATE")
		./run.sh "${args[@]}" "$bag"
	done
done

echo "All $((${#DATASETS[@]} * ${#WINDOWS[@]})) evaluations completed."
echo "Results: $ROOT/$OUTPUT_DIR"
