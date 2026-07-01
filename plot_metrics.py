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
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        import glob
        files = glob.glob('metrics/pipeline/metrics_*.csv')
        if files:
            files.sort()
            csv_path = files[-1]
            print(f"No CSV path provided. Automatically picked the latest: {csv_path}")
        else:
            csv_path = 'extrinsic_metrics.csv'

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

    # Determine filenames based on CSV input
    csv_basename = os.path.basename(csv_path)
    csv_name_no_ext = os.path.splitext(csv_basename)[0]
    
    # Ensure metrics/eval/ directory exists
    eval_dir = 'metrics/eval'
    os.makedirs(eval_dir, exist_ok=True)
    
    output_image = os.path.join(eval_dir, f"{csv_name_no_ext}_plot.png")
    output_text = os.path.join(eval_dir, f"{csv_name_no_ext}_summary.txt")

    # Normalize time to start at 0
    t_start = df['timestamp'].iloc[0]
    time_sec = df['timestamp'] - t_start

    # Create figure with a modern clean grid layout (3 rows, 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('VGGT-1B Calibration & Depth Estimation Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 1. Plot VGGT Depth Scale Alignment
    ax = axs[0, 0]
    if 'scale' in df.columns:
        ax.plot(time_sec, df['scale'], label='Depth Scale Factor', color='#3498db', linewidth=2)
        ax.set_title('VGGT Depth Scale Alignment', fontweight='bold')
        ax.set_ylabel('Scale Factor (s = median(D_metric / D_pred))')
        median_s = df['scale'].median()
        ax.axhline(median_s, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Median: {median_s:.3f}')
    else:
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

    # 3. Plot Head Camera Depth Error
    ax = axs[1, 0]
    if 'head_depth_rmse' in df.columns and 'head_depth_mae' in df.columns:
        ax.plot(time_sec, df['head_depth_rmse'], label='RMSE (m)', color='#e74c3c', linewidth=2)
        ax.plot(time_sec, df['head_depth_mae'], label='MAE (m)', color='#f39c12', linewidth=1.5, linestyle='--')
        ax.set_title('Head Camera Depth Estimation Error', fontweight='bold')
        ax.set_ylabel('Error (meters)')
        mean_rmse = df['head_depth_rmse'].mean()
        ax.axhline(mean_rmse, color='#c0392b', linestyle=':', alpha=0.8, label=f'Mean RMSE: {mean_rmse:.3f}')
    else:
        ax.text(0.5, 0.5, "No Head Depth Error data", ha='center', va='center')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 4. Plot Exo Camera Depth Error
    ax = axs[1, 1]
    if 'exo_depth_rmse' in df.columns and 'exo_depth_mae' in df.columns:
        ax.plot(time_sec, df['exo_depth_rmse'], label='RMSE (m)', color='#9b59b6', linewidth=2)
        ax.plot(time_sec, df['exo_depth_mae'], label='MAE (m)', color='#1abc9c', linewidth=1.5, linestyle='--')
        ax.set_title('Exo Camera Depth Estimation Error', fontweight='bold')
        ax.set_ylabel('Error (meters)')
        mean_rmse = df['exo_depth_rmse'].mean()
        ax.axhline(mean_rmse, color='#8e44ad', linestyle=':', alpha=0.8, label=f'Mean RMSE: {mean_rmse:.3f}')
    else:
        ax.text(0.5, 0.5, "No Exo Depth Error data", ha='center', va='center')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 5. Accuracy / Trajectory Plot vs Ground Truth
    ax = axs[2, 0]
    has_gt = 'error_t' in df.columns and df['error_t'].notna().any()
    
    if has_gt:
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
        ax.text(0.5, 0.5, "No Ground Truth Available\n\nTo compute errors, configure:\n- gt_parent_frame\n- gt_child_frame", 
                ha='center', va='center', fontsize=12, color='#7f8c8d', weight='bold')
        ax.set_title('Estimation Errors vs Ground Truth', fontweight='bold')
        ax.set_axis_off()
    ax.set_xlabel('Time (seconds)')

    # 6. Solver Status Breakdown
    ax = axs[2, 1]
    status_counts = df['status'].value_counts()
    colors = ['#2ecc71' if s == 'SUCCESS' else '#e74c3c' for s in status_counts.index]
    status_counts.plot(kind='barh', color=colors, ax=ax, edgecolor='black', alpha=0.8)
    ax.set_title('Solver Status Breakdown', fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Status / Rejection Code')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_image, dpi=150)
    print(f"Plot successfully saved to: {os.path.abspath(output_image)}")

    # 5. Compute Stats and Generate Text Summary
    total_frames = len(df)
    status_freq = df['status'].value_counts()
    status_pct = df['status'].value_counts(normalize=True) * 100.0

    # Filter for SUCCESS frames for extrinsic stats
    success_df = df[df['status'] == 'SUCCESS']
    num_success = len(success_df)

    summary_lines = []
    summary_lines.append("=========================================")
    summary_lines.append("EXTRINSIC CALIBRATION EVALUATION SUMMARY")
    summary_lines.append("=========================================")
    summary_lines.append(f"Metrics CSV File: {csv_path}")
    summary_lines.append(f"Total Frames Processed: {total_frames}")
    summary_lines.append("")
    summary_lines.append("-----------------------------------------")
    summary_lines.append("SOLVER STATUS BREAKDOWN")
    summary_lines.append("-----------------------------------------")
    for status, count in status_freq.items():
        pct = status_pct[status]
        summary_lines.append(f"{status:<20}: {count:>4} ({pct:>6.2f}%)")
    summary_lines.append("")

    if num_success > 0:
        summary_lines.append("-----------------------------------------")
        summary_lines.append("ESTIMATED EXTRINSICS STATS (SUCCESS ONLY)")
        summary_lines.append("-----------------------------------------")
        for col, label in [('t_x', 't_x (Left/Right)'), ('t_y', 't_y (Up/Down)'), ('t_z', 't_z (Forward/Backward)')]:
            vals = success_df[col]
            mean_val = vals.mean()
            var_val = vals.var() if num_success > 1 else 0.0
            std_val = vals.std() if num_success > 1 else 0.0
            min_val = vals.min()
            max_val = vals.max()
            summary_lines.append(f"{label}:")
            summary_lines.append(f"  Mean:     {mean_val:>8.4f} m")
            summary_lines.append(f"  Variance: {var_val:>8.6f} m^2 (StdDev: {std_val:.4f} m)")
            summary_lines.append(f"  Min/Max:  {min_val:>8.4f} / {max_val:>8.4f} m")
        summary_lines.append("")

        if 'scale' in success_df.columns:
            summary_lines.append("-----------------------------------------")
            summary_lines.append("VGGT DEPTH SCALE ALIGNMENT")
            summary_lines.append("-----------------------------------------")
            scales = success_df['scale']
            mean_s = scales.mean()
            median_s = scales.median()
            var_s = scales.var() if num_success > 1 else 0.0
            min_s = scales.min()
            max_s = scales.max()
            summary_lines.append("Scale Factor:")
            summary_lines.append(f"  Mean:     {mean_s:>8.4f}")
            summary_lines.append(f"  Median:   {median_s:>8.4f}")
            summary_lines.append(f"  Variance: {var_s:>8.6f}")
            summary_lines.append(f"  Min/Max:  {min_s:>8.4f} / {max_s:>8.4f}")
            summary_lines.append("")

        # Depth Estimation accuracy
        has_depth_metrics = 'head_depth_rmse' in success_df.columns
        if has_depth_metrics:
            summary_lines.append("-----------------------------------------")
            summary_lines.append("VGGT VS GT DEPTH ESTIMATION ERROR")
            summary_lines.append("-----------------------------------------")
            
            for cam, prefix in [('Head Camera', 'head_depth'), ('Exo Camera', 'exo_depth')]:
                rmse_col = f"{prefix}_rmse"
                mae_col = f"{prefix}_mae"
                
                rmses = success_df[rmse_col].dropna()
                maes = success_df[mae_col].dropna()
                
                if len(rmses) > 0:
                    summary_lines.append(f"{cam}:")
                    summary_lines.append(f"  RMSE Mean:     {rmses.mean():>8.4f} m")
                    summary_lines.append(f"  RMSE Variance: {rmses.var() if len(rmses) > 1 else 0.0:>8.6f} m^2")
                    summary_lines.append(f"  MAE Mean:      {maes.mean():>8.4f} m")
                    summary_lines.append(f"  MAE Variance:  {maes.var() if len(maes) > 1 else 0.0:>8.6f} m^2")
            summary_lines.append("")

        # Ground truth errors
        success_has_gt = 'error_t' in success_df.columns and success_df['error_t'].notna().any()
        if success_has_gt:
            summary_lines.append("-----------------------------------------")
            summary_lines.append("ACCURACY VS GROUND TRUTH (SUCCESS ONLY)")
            summary_lines.append("-----------------------------------------")
            
            err_t = success_df['error_t'].dropna()
            mean_err_t = err_t.mean()
            max_err_t = err_t.max()
            var_err_t = err_t.var() if len(err_t) > 1 else 0.0
            summary_lines.append("Translation Error:")
            summary_lines.append(f"  Mean:     {mean_err_t:>8.4f} m")
            summary_lines.append(f"  Max:      {max_err_t:>8.4f} m")
            summary_lines.append(f"  Variance: {var_err_t:>8.6f} m^2")

            if 'error_r_deg' in success_df.columns:
                err_r = success_df['error_r_deg'].dropna()
                mean_err_r = err_r.mean()
                max_err_r = err_r.max()
                var_err_r = err_r.var() if len(err_r) > 1 else 0.0
                summary_lines.append("Rotation Error:")
                summary_lines.append(f"  Mean:     {mean_err_r:>8.4f} deg")
                summary_lines.append(f"  Max:      {max_err_r:>8.4f} deg")
                summary_lines.append(f"  Variance: {var_err_r:>8.6f} deg^2")
            summary_lines.append("")
    else:
        summary_lines.append("No successful frames to compute extrinsics statistics.")

    # Write summary text to file
    try:
        with open(output_text, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"Summary report successfully saved to: {os.path.abspath(output_text)}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

    # Show if in interactive environment
    if os.environ.get('DISPLAY', '').strip():
        plt.show()

if __name__ == '__main__':
    main()
