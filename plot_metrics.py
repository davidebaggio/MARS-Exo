#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

# ponytail: support headless systems (like docker or ssh) without crashing
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
        print("Please run the pipeline first to generate the metrics file.")
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

    # Normalize time to start at 0
    t_start = df['timestamp'].iloc[0]
    time_sec = df['timestamp'] - t_start

    # Create figure with a modern clean grid layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Extrinsic Calibration Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 1. Plot Match Counts
    ax = axs[0, 0]
    ax.plot(time_sec, df['num_2d_matches'], label='2D Matches', color='#3498db', alpha=0.8)
    ax.plot(time_sec, df['num_3d_matches'], label='3D Deprojected Matches', color='#e67e22', alpha=0.8)
    ax.plot(time_sec, df['inliers'], label='RANSAC Inliers', color='#2ecc71', linewidth=2)
    ax.set_title('Match Statistics over Time', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Count')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 2. Plot RANSAC RMSE & Inlier Ratio
    ax1 = axs[0, 1]
    ax1.plot(time_sec, df['rmse'], color='#e74c3c', label='RMSE (m)', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('RMSE (meters)', color='#e74c3c')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(time_sec, df['inlier_ratio'], color='#9b59b6', label='Inlier Ratio', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Inlier Ratio', color='#9b59b6')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    ax1.set_title('RANSAC Quality Metrics', fontweight='bold')

    # 3. Accuracy / Trajectory plot
    ax = axs[1, 0]
    has_gt = df['gt_t_x'].notna().any() and df['error_t'].notna().any()
    
    if has_gt:
        # Plot ground truth error
        ax.plot(time_sec, df['error_t'], color='#e74c3c', label='Translation Error (m)', linewidth=2)
        ax.set_ylabel('Translation Error (meters)', color='#e74c3c')
        ax.tick_params(axis='y', labelcolor='#e74c3c')
        ax.grid(True, linestyle='--', alpha=0.6)

        ax_rot = ax.twinx()
        ax_rot.plot(time_sec, df['error_r_deg'], color='#2c3e50', label='Rotation Error (deg)', linestyle='-', alpha=0.8)
        ax_rot.set_ylabel('Rotation Error (degrees)', color='#2c3e50')
        ax_rot.tick_params(axis='y', labelcolor='#2c3e50')
        ax.set_title('Estimation Errors vs Ground Truth', fontweight='bold')
    else:
        # Plot raw estimates (useful to check calibration convergence and stability)
        ax.plot(time_sec, df['t_x'], label='t_x', color='#2ecc71', alpha=0.8)
        ax.plot(time_sec, df['t_y'], label='t_y', color='#3498db', alpha=0.8)
        ax.plot(time_sec, df['t_z'], label='t_z', color='#e67e22', alpha=0.8)
        ax.set_title('Estimated Translation Components (No GT)', fontweight='bold')
        ax.set_ylabel('Distance (meters)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
    ax.set_xlabel('Time (seconds)')

    # 4. Solver Status Breakdown
    ax = axs[1, 1]
    status_counts = df['status'].value_counts()
    colors = ['#2ecc71' if s == 'SUCCESS' else '#e74c3c' for s in status_counts.index]
    status_counts.plot(kind='barh', color=colors, ax=ax, edgecolor='black', alpha=0.8)
    ax.set_title('Solver Status Breakdown', fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Status / Rejection Code')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save figure
    output_filename = 'extrinsic_metrics_plot.png'
    plt.savefig(output_filename, dpi=150)
    print(f"Plot successfully saved to: {os.path.abspath(output_filename)}")

    # Show if in interactive environment
    if os.environ.get('DISPLAY', '').strip():
        plt.show()

if __name__ == '__main__':
    main()
