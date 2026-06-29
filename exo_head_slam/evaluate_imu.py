#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

if not os.environ.get('DISPLAY', '').strip():
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def main():
    csv_path = 'extrinsic_metrics.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: Metrics file '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading metrics from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("Error: Metrics file is empty.")
        sys.exit(0)

    t_start = df['timestamp'].iloc[0]
    time_sec = df['timestamp'] - t_start

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('IMU-Enhanced Extrinsic Calibration Metrics', fontsize=16, fontweight='bold')

    has_imu = 'gravity_error_deg' in df.columns and df['gravity_error_deg'].notna().any()

    if has_imu:
        ax = axs[0, 0]
        ax.plot(time_sec, df['gravity_error_deg'], color='#9b59b6', linewidth=1.5)
        ax.axhline(y=15.0, color='red', linestyle='--', alpha=0.5, label='Rejection threshold')
        ax.set_title('Gravity Alignment Error', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Error (deg)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        ax = axs[0, 1]
        success_mask = df['status'] == 'SUCCESS'
        if 'imu_constraint_applied' in df.columns:
            imu_mask = df['imu_constraint_applied'] == True
            if imu_mask.any():
                ax.scatter(time_sec[success_mask & imu_mask],
                           df.loc[success_mask & imu_mask, 'gravity_error_deg'],
                           color='#2ecc71', label='IMU constraint ON', alpha=0.7, s=20)
            ax.scatter(time_sec[success_mask & ~imu_mask],
                       df.loc[success_mask & ~imu_mask, 'gravity_error_deg'],
                       color='#3498db', label='IMU constraint OFF', alpha=0.7, s=20)
            ax.set_title('Gravity Error by IMU Constraint', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Gravity Error (deg)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No IMU constraint data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title('IMU Constraint Analysis', fontweight='bold')

        ax = axs[0, 2]
        if 'is_stationary' in df.columns:
            stationary = df['is_stationary'] == True
            moving = df['is_stationary'] == False
            unknown = df['is_stationary'].isna()
            labels = []
            sizes = []
            colors_ = []
            if stationary.any():
                labels.append(f'Stationary ({stationary.sum()})')
                sizes.append(stationary.sum())
                colors_.append('#2ecc71')
            if moving.any():
                labels.append(f'Moving ({moving.sum()})')
                sizes.append(moving.sum())
                colors_.append('#e74c3c')
            if unknown.any():
                labels.append(f'Unknown ({unknown.sum()})')
                sizes.append(unknown.sum())
                colors_.append('#95a5a6')
            if sizes:
                ax.pie(sizes, labels=labels, colors=colors_, autopct='%1.1f%%', startangle=90)
            ax.set_title('Motion State Distribution', fontweight='bold')

        ax = axs[1, 0]
        status_colors = {'SUCCESS': '#2ecc71', 'REJECTED_GRAVITY_MISMATCH': '#e74c3c',
                         'REJECTED_CONFIDENCE_LIMITS': '#f39c12', 'REJECTED_TRANS_JUMP': '#e67e22',
                         'REJECTED_ROT_JUMP': '#d35400', 'NO_2D_MATCHES': '#95a5a6',
                         'INSUFFICIENT_3D_MATCHES': '#7f8c8d', 'RANSAC_FAILED': '#c0392b'}
        status_counts = df['status'].value_counts()
        colors_list = [status_colors.get(s, '#34495e') for s in status_counts.index]
        status_counts.plot(kind='barh', color=colors_list, ax=ax, edgecolor='black', alpha=0.8)
        ax.set_title('Status Breakdown (with IMU)', fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        ax = axs[1, 1]
        if 'rmse' in df.columns and 'gravity_error_deg' in df.columns:
            sc = ax.scatter(df['rmse'], df['gravity_error_deg'],
                            c=time_sec, cmap='viridis', alpha=0.6, s=15)
            ax.set_title('RMSE vs Gravity Error', fontweight='bold')
            ax.set_xlabel('RANSAC RMSE (m)')
            ax.set_ylabel('Gravity Alignment Error (deg)')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.colorbar(sc, ax=ax, label='Time (s)')

        ax = axs[1, 2]
        ax.axis('off')
        total = len(df)
        successes = (df['status'] == 'SUCCESS').sum()
        gravity_rejects = (df['status'] == 'REJECTED_GRAVITY_MISMATCH').sum() if 'REJECTED_GRAVITY_MISMATCH' in df['status'].values else 0
        success_rate = successes / total * 100 if total > 0 else 0
        summary_text = (
            f"Total frames: {total}\n"
            f"Successes: {successes} ({success_rate:.1f}%)\n"
            f"Gravity rejects: {gravity_rejects}\n"
            f"Mean gravity error: {df['gravity_error_deg'].mean():.2f}°\n"
            f"Median gravity error: {df['gravity_error_deg'].median():.2f}°\n"
            f"Std gravity error: {df['gravity_error_deg'].std():.2f}°"
        )
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=13,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('IMU Metrics Summary', fontweight='bold')
    else:
        for ax in axs.flat:
            ax.text(0.5, 0.5, 'No IMU metrics data found.\nRun pipeline with IMU enabled.',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('No IMU Data', fontweight='bold')

    plt.tight_layout()

    output = 'imu_metrics_plot.png'
    plt.savefig(output, dpi=150)
    print(f"IMU metrics plot saved to: {os.path.abspath(output)}")

    if os.environ.get('DISPLAY', '').strip():
        plt.show()


if __name__ == '__main__':
    main()
