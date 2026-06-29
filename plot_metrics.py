#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

if not os.environ.get('DISPLAY', '').strip():
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_extrinsic(df, time_sec, axs):
    ax = axs[0]
    ax.plot(time_sec, df['num_2d_matches'], label='2D Matches', color='#3498db', alpha=0.8)
    ax.plot(time_sec, df['num_3d_matches'], label='3D Deprojected Matches', color='#e67e22', alpha=0.8)
    ax.plot(time_sec, df['inliers'], label='RANSAC Inliers', color='#2ecc71', linewidth=2)
    ax.set_title('Match Statistics', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Count')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    ax1 = axs[1]
    ax1.plot(time_sec, df['rmse'], color='#e74c3c', label='RMSE (m)', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RMSE (m)', color='#e74c3c')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(time_sec, df['inlier_ratio'], color='#9b59b6', label='Inlier Ratio', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Inlier Ratio', color='#9b59b6')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    ax1.set_title('RANSAC Quality', fontweight='bold')

    ax = axs[2]
    has_gt = df['gt_t_x'].notna().any() and df['error_t'].notna().any()
    if has_gt:
        ax.plot(time_sec, df['error_t'], color='#e74c3c', label='Translation Error (m)', linewidth=2)
        ax.set_ylabel('Translation Error (m)', color='#e74c3c')
        ax.tick_params(axis='y', labelcolor='#e74c3c')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax_rot = ax.twinx()
        ax_rot.plot(time_sec, df['error_r_deg'], color='#2c3e50', label='Rotation Error (deg)', linestyle='-', alpha=0.8)
        ax_rot.set_ylabel('Rotation Error (deg)', color='#2c3e50')
        ax_rot.tick_params(axis='y', labelcolor='#2c3e50')
        ax.set_title('Errors vs GT', fontweight='bold')
    else:
        ax.plot(time_sec, df['t_x'], label='t_x', color='#2ecc71', alpha=0.8)
        ax.plot(time_sec, df['t_y'], label='t_y', color='#3498db', alpha=0.8)
        ax.plot(time_sec, df['t_z'], label='t_z', color='#e67e22', alpha=0.8)
        ax.set_title('Translation Estimates', fontweight='bold')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')

    ax = axs[3]
    status_counts = df['status'].value_counts()
    colors = ['#2ecc71' if s == 'SUCCESS' else '#e74c3c' for s in status_counts.index]
    status_counts.plot(kind='barh', color=colors, ax=ax, edgecolor='black', alpha=0.8)
    ax.set_title('Solver Status', fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)


def plot_imu(df, time_sec, axs):
    ax = axs[0]
    ax.plot(time_sec, df['gravity_error_deg'], color='#9b59b6', linewidth=1.5)
    ax.axhline(y=15.0, color='red', linestyle='--', alpha=0.5, label='Rejection threshold')
    ax.set_title('Gravity Alignment Error', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Error (deg)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    ax = axs[1]
    success_mask = df['status'] == 'SUCCESS'
    if 'imu_constraint_applied' in df.columns:
        imu_mask = df['imu_constraint_applied'] == True
        if imu_mask.any():
            ax.scatter(time_sec[success_mask & imu_mask],
                       df.loc[success_mask & imu_mask, 'gravity_error_deg'],
                       color='#2ecc71', label='IMU ON', alpha=0.7, s=20)
        ax.scatter(time_sec[success_mask & ~imu_mask],
                   df.loc[success_mask & ~imu_mask, 'gravity_error_deg'],
                   color='#3498db', label='IMU OFF', alpha=0.7, s=20)
        ax.set_title('Gravity Error by Constraint', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gravity Error (deg)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    ax = axs[2]
    status_colors = {'SUCCESS': '#2ecc71', 'REJECTED_GRAVITY_MISMATCH': '#e74c3c',
                     'REJECTED_CONFIDENCE_LIMITS': '#f39c12', 'REJECTED_TRANS_JUMP': '#e67e22',
                     'REJECTED_ROT_JUMP': '#d35400', 'NO_2D_MATCHES': '#95a5a6',
                     'INSUFFICIENT_3D_MATCHES': '#7f8c8d', 'RANSAC_FAILED': '#c0392b'}
    status_counts = df['status'].value_counts()
    colors_list = [status_colors.get(s, '#34495e') for s in status_counts.index]
    status_counts.plot(kind='barh', color=colors_list, ax=ax, edgecolor='black', alpha=0.8)
    ax.set_title('Status Breakdown', fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    ax = axs[3]
    if 'rmse' in df.columns and 'gravity_error_deg' in df.columns:
        sc = ax.scatter(df['rmse'], df['gravity_error_deg'],
                        c=time_sec, cmap='viridis', alpha=0.6, s=15)
        ax.set_title('RMSE vs Gravity Error', fontweight='bold')
        ax.set_xlabel('RANSAC RMSE (m)')
        ax.set_ylabel('Gravity Error (deg)')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Time (s)')


def main():
    csv_path = 'extrinsic_metrics.csv'
    eval_mode = 'auto'

    args = sys.argv[1:]
    if args and args[0] in ('--imu', '--eval', '--extrinsic'):
        eval_mode = args[0].lstrip('-')
        if eval_mode == 'imu':
            csv_path = 'imu_evaluation.csv'
        args = args[1:]

    if args:
        csv_path = args[0]

    if not os.path.exists(csv_path):
        print(f"Error: Metrics file '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading metrics from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("Error: Metrics file is empty.")
        sys.exit(0)

    has_imu = 'gravity_error_deg' in df.columns and df['gravity_error_deg'].notna().any()
    is_summary = 'total_frames' in df.columns

    if is_summary:
        print("Summary row detected. Plotting evaluation summary stats...")
        print(df[df['total_frames'].notna()].to_string(index=False))
        return

    t_start = df['timestamp'].iloc[0]
    time_sec = df['timestamp'] - t_start

    if eval_mode == 'imu' or (eval_mode == 'auto' and has_imu):
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('IMU-Enhanced Extrinsic Calibration Metrics', fontsize=16, fontweight='bold')
        plot_imu(df, time_sec, axs.flatten())
        output = 'imu_metrics_plot.png'
    else:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Extrinsic Calibration Evaluation Metrics', fontsize=16, fontweight='bold')
        plot_extrinsic(df, time_sec, axs.flatten())
        output = 'extrinsic_metrics_plot.png'

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Plot saved to: {os.path.abspath(output)}")

    if os.environ.get('DISPLAY', '').strip():
        plt.show()


if __name__ == '__main__':
    main()
