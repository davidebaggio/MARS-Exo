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
    fig.suptitle('VGGT-1B Extrinsic Calibration Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 1. Plot VGGT Depth Scale Alignment
    ax = axs[0, 0]
    if 'scale' in df.columns:
        ax.plot(time_sec, df['scale'], label='Depth Scale Factor', color='#3498db', linewidth=2)
        ax.set_title('VGGT Depth Scale Alignment', fontweight='bold')
        ax.set_ylabel('Scale Factor (s = median(D_metric / D_pred))')
        # Add horizontal line at median scale for reference
        median_s = df['scale'].median()
        ax.axhline(median_s, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Median: {median_s:.3f}')
    else:
        # Fallback to matches if old file
        ax.plot(time_sec, df.get('num_2d_matches', np.zeros(len(df))), label='2D Matches', color='#3498db', alpha=0.8)
        ax.set_title('Matches (Legacy)', fontweight='bold')
        ax.set_ylabel('Count')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 2. Plot Estimated Translation Components
    ax = axs[0, 1]
    ax.plot(time_sec, df['t_x'], label='t_x (Left/Right)', color='#2ecc71', alpha=0.8, linewidth=2)
    ax.plot(time_sec, df['t_y'], label='t_y (Up/Down)', color='#3498db', alpha=0.8, linewidth=2)
    ax.plot(time_sec, df['t_z'], label='t_z (Forward/Backward)', color='#e67e22', alpha=0.8, linewidth=2)
    ax.set_title('Estimated Camera Extrinsics (Translation)', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Distance (meters)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 3. Accuracy / Trajectory Plot
    ax = axs[1, 0]
    has_gt = 'error_t' in df.columns and df['error_t'].notna().any()
    
    if has_gt:
        # Plot ground truth error
        ax.plot(time_sec, df['error_t'], color='#e74c3c', label='Translation Error (m)', linewidth=2)
        ax.set_ylabel('Translation Error (meters)', color='#e74c3c')
        ax.tick_params(axis='y', labelcolor='#e74c3c')
        ax.grid(True, linestyle='--', alpha=0.6)

        if 'error_r_deg' in df.columns:
            ax_rot = ax.twinx()
            ax_rot.plot(time_sec, df['error_r_deg'], color='#2c3e50', label='Rotation Error (deg)', linestyle='-', alpha=0.8)
            ax_rot.set_ylabel('Rotation Error (degrees)', color='#2c3e50')
            ax_rot.tick_params(axis='y', labelcolor='#2c3e50')
        ax.set_title('Estimation Errors vs Ground Truth', fontweight='bold')
    else:
        # Write text indicating no GT frames configured
        ax.text(0.5, 0.5, "No Ground Truth Available\n\nTo compute errors, configure:\n- gt_parent_frame\n- gt_child_frame", 
                ha='center', va='center', fontsize=12, color='#7f8c8d', weight='bold')
        ax.set_title('Estimation Errors vs Ground Truth', fontweight='bold')
        ax.set_axis_off()
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
    output_filename = 'extrinsic_metrics_plot.png' if csv_path == 'extrinsic_metrics.csv' else f"{os.path.splitext(os.path.basename(csv_path))[0]}_plot.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Plot successfully saved to: {os.path.abspath(output_filename)}")

    # Show if in interactive environment
    if os.environ.get('DISPLAY', '').strip():
        plt.show()

if __name__ == '__main__':
    main()
