#!/usr/bin/env python3
import os
import sys
import csv
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(CURRENT_DIR, '..', 'extrinsic_metrics.csv')
DEFAULT_OUTPUT = os.path.join(CURRENT_DIR, '..', 'imu_evaluation.csv')


def compute_imu_evaluation(input_csv: str, output_csv: str):
    if not os.path.exists(input_csv):
        print(f"Error: Input metrics file '{input_csv}' not found.")
        sys.exit(1)

    print(f"Loading metrics from {input_csv}...")
    df = pd.read_csv(input_csv)

    if df.empty:
        print("Error: Metrics file is empty.")
        sys.exit(0)

    has_imu = 'gravity_error_deg' in df.columns and df['gravity_error_deg'].notna().any()

    t_start = df['timestamp'].iloc[0]
    time_sec = df['timestamp'] - t_start

    rows = []
    for idx, row in df.iterrows():
        r = {
            'timestamp': row.get('timestamp', np.nan),
            'time_sec': time_sec.iloc[idx] if idx < len(time_sec) else np.nan,
            'num_2d_matches': row.get('num_2d_matches', np.nan),
            'num_3d_matches': row.get('num_3d_matches', np.nan),
            'inliers': row.get('inliers', np.nan),
            'inlier_ratio': row.get('inlier_ratio', np.nan),
            'rmse': row.get('rmse', np.nan),
            'status': row.get('status', ''),
        }
        if has_imu:
            r['gravity_error_deg'] = row.get('gravity_error_deg', np.nan)
            r['is_stationary'] = row.get('is_stationary', np.nan)
            r['imu_constraint_applied'] = row.get('imu_constraint_applied', np.nan)
        rows.append(r)

    summary = {}
    total = len(df)
    successes = int((df['status'] == 'SUCCESS').sum())
    summary['total_frames'] = total
    summary['successes'] = successes
    summary['success_rate_pct'] = round(successes / total * 100, 2) if total > 0 else 0.0

    if has_imu:
        summary['gravity_rejects'] = int((df['status'] == 'REJECTED_GRAVITY_MISMATCH').sum())
        gerr = df['gravity_error_deg'].dropna()
        summary['gravity_error_mean_deg'] = round(float(gerr.mean()), 4) if len(gerr) > 0 else np.nan
        summary['gravity_error_median_deg'] = round(float(gerr.median()), 4) if len(gerr) > 0 else np.nan
        summary['gravity_error_std_deg'] = round(float(gerr.std()), 4) if len(gerr) > 0 else np.nan
        summary['gravity_error_min_deg'] = round(float(gerr.min()), 4) if len(gerr) > 0 else np.nan
        summary['gravity_error_max_deg'] = round(float(gerr.max()), 4) if len(gerr) > 0 else np.nan
    else:
        summary['gravity_rejects'] = 0

    try:
        with open(output_csv, mode='w', newline='') as f:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({k: summary.get(k, '') for k in fieldnames})
        print(f"IMU evaluation CSV saved to: {os.path.abspath(output_csv)}")
    except Exception as e:
        print(f"Error writing IMU evaluation CSV: {e}")

    return output_csv, has_imu


def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    compute_imu_evaluation(input_csv, output_csv)


if __name__ == '__main__':
    main()
