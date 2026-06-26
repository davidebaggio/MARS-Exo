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

    # Calculate statistics
    total_frames = len(df)
    success_direct = len(df[df['status'] == 'SUCCESS'])
    success_fallback = len(df[df['status'] == 'SUCCESS_SLAM_FALLBACK'])
    failed = total_frames - success_direct - success_fallback
    
    print("\n" + "="*45)
    print("      EXTRINSIC CALIBRATION METRICS SUMMARY")
    print("="*45)
    print(f"Total synced frames:  {total_frames}")
    print(f"Direct Match success: {success_direct} ({success_direct/total_frames*100:.1f}%)")
    print(f"SLAM Fallback success: {success_fallback} ({success_fallback/total_frames*100:.1f}%)")
    print(f"Solver failures:      {failed} ({failed/total_frames*100:.1f}%)")
    if (success_direct + success_fallback) > 0:
        reliance_rate = success_fallback / (success_direct + success_fallback) * 100
        print(f"Fallback reliance:    {reliance_rate:.1f}% of successful frames")
    print("-"*45)

    # B. Direct vs. Fallback Drift Comparison (against Ground Truth)
    has_gt = df['gt_t_x'].notna().any() and df['error_t'].notna().any()
    if has_gt:
        direct_df = df[df['status'] == 'SUCCESS']
        fallback_df = df[df['status'] == 'SUCCESS_SLAM_FALLBACK']
        print("      DRIFT COMPARISON (vs. Ground Truth)")
        print("-"*45)
        if len(direct_df) > 0:
            print(f"Direct Match (SUCCESS):")
            print(f"  Mean Trans Error: {direct_df['error_t'].mean():.4f} m")
            print(f"  Mean Rot Error:   {direct_df['error_r_deg'].mean():.2f}°")
        else:
            print("Direct Match: No successful frames.")
            
        if len(fallback_df) > 0:
            print(f"SLAM Fallback (SUCCESS_SLAM_FALLBACK):")
            print(f"  Mean Trans Error: {fallback_df['error_t'].mean():.4f} m")
            print(f"  Mean Rot Error:   {fallback_df['error_r_deg'].mean():.2f}°")
        else:
            print("SLAM Fallback: No successful frames.")
        print("-"*45)

    # C. Inter-Agent Map Consistency / Relative Distance Jitter
    valid_mask = df['status'].isin(['SUCCESS', 'SUCCESS_SLAM_FALLBACK'])
    valid_df = df[valid_mask]
    if len(valid_df) > 0:
        dist_estim = np.sqrt(valid_df['t_x']**2 + valid_df['t_y']**2 + valid_df['t_z']**2)
        print("    INTER-AGENT DISTANCE & MAP CONSISTENCY")
        print("-"*45)
        print(f"Estimated Camera-to-Camera Distance:")
        print(f"  Mean Distance:    {dist_estim.mean():.4f} m")
        print(f"  Std Dev (Jitter): {dist_estim.std():.4f} m")
        if has_gt:
            gt_dist = np.sqrt(valid_df['gt_t_x']**2 + valid_df['gt_t_y']**2 + valid_df['gt_t_z']**2)
            print(f"  Ground Truth Mean: {gt_dist.mean():.4f} m")
    print("-"*45)

    # D. Variance of translation components by solver mode
    for mode_name, mode_status in [('Direct Match (SUCCESS)', 'SUCCESS'),
                                    ('SLAM Fallback (SUCCESS_SLAM_FALLBACK)', 'SUCCESS_SLAM_FALLBACK')]:
        mode_df = df[df['status'] == mode_status]
        n = len(mode_df)
        if n < 2:
            print(f"{mode_name}: insufficient frames ({n}) for variance computation")
            continue
        var_tx = mode_df['t_x'].var(ddof=1)
        var_ty = mode_df['t_y'].var(ddof=1)
        var_tz = mode_df['t_z'].var(ddof=1)
        print(f"    VARIANCE BY SOLVER MODE")
        print(f"{mode_name} ({n} frames):")
        print(f"  Var(t_x): {var_tx:.6f}  Var(t_y): {var_ty:.6f}  Var(t_z): {var_tz:.6f}")
        print(f"  Avg Variance: {(var_tx + var_ty + var_tz) / 3:.6f}")
    print("="*45 + "\n")

    # 3. Accuracy / Trajectory plot
    ax = axs[1, 0]
    has_gt = df['gt_t_x'].notna().any() and df['error_t'].notna().any()
    
    # Helper to shade fallback regions
    def shade_fallback_regions(axis):
        fallback_span_labeled = False
        for i in range(len(time_sec)):
            if df['status'].iloc[i] == 'SUCCESS_SLAM_FALLBACK':
                dt = 0.5
                if i < len(time_sec) - 1:
                    dt = time_sec.iloc[i+1] - time_sec.iloc[i]
                label = 'SLAM Fallback Active' if not fallback_span_labeled else ""
                axis.axvspan(time_sec.iloc[i], time_sec.iloc[i] + dt, color='#3498db', alpha=0.15, label=label)
                fallback_span_labeled = True

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
        
        # Shade fallback regions
        shade_fallback_regions(ax)
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_rot.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        # Plot raw estimates (useful to check calibration convergence and stability)
        ax.plot(time_sec, df['t_x'], label='t_x', color='#2ecc71', alpha=0.8)
        ax.plot(time_sec, df['t_y'], label='t_y', color='#3498db', alpha=0.8)
        ax.plot(time_sec, df['t_z'], label='t_z', color='#e67e22', alpha=0.8)
        ax.set_title('Estimated Translation Components (No GT)', fontweight='bold')
        ax.set_ylabel('Distance (meters)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Shade fallback regions
        shade_fallback_regions(ax)
        ax.legend(loc='upper right')
        
    ax.set_xlabel('Time (seconds)')

    # 4. Solver Status Breakdown
    ax = axs[1, 1]
    status_counts = df['status'].value_counts()
    
    # Color mapping for different statuses
    status_colors = {
        'SUCCESS': '#2ecc71',
        'SUCCESS_SLAM_FALLBACK': '#3498db'
    }
    colors = [status_colors.get(s, '#e74c3c') for s in status_counts.index]
    
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
