#!/usr/bin/env python3
import glob
import os
import sys
import pandas as pd
import numpy as np

# Support headless systems (like docker or ssh) without crashing
if not os.environ.get('DISPLAY', '').strip():
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_cloud_metrics(csv_path, df):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    eval_dir = 'metrics/eval'
    os.makedirs(eval_dir, exist_ok=True)
    output_image = os.path.join(eval_dir, f'{name}_plot.png')
    output_text = os.path.join(eval_dir, f'{name}_summary.txt')
    cloud = df.mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle('VGGT Visible Cloud vs Ground Truth', fontsize=16, fontweight='bold')
    columns = ['accuracy_mean', 'accuracy_rmse', 'completeness_mean',
               'completeness_rmse', 'chamfer']
    labels = ['Accuracy\nmean', 'Accuracy\nRMSE', 'Completeness\nmean',
              'Completeness\nRMSE', 'Chamfer']
    bars = ax.bar(labels, [cloud[column] for column in columns], color='#3498db')
    ax.bar_label(bars, fmt='%.3f')
    ax.set_ylabel('Distance (m)')
    ax.set_title(f'Mean metrics over {len(df)} evaluation row(s)\nF-score: {cloud["fscore"]:.3f}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    maps_path = f'{os.path.splitext(csv_path)[0]}_maps.npz'
    if os.path.exists(maps_path):
        import open3d as o3d

        with np.load(maps_path) as maps:
            predicted = maps['predicted']
            ground_truth = maps['ground_truth']

        gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_truth))
        gt_cloud.paint_uniform_color([0.18, 0.80, 0.44])
        predicted_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(predicted))
        predicted_cloud.paint_uniform_color([0.90, 0.30, 0.24])
        combined = gt_cloud + predicted_cloud
        output_cloud = os.path.join(eval_dir, f'{name}_maps.ply')
        o3d.io.write_point_cloud(output_cloud, combined)
        print(f'Open3D cloud successfully saved to: {os.path.abspath(output_cloud)}')

        if os.environ.get('DISPLAY', '').strip():
            extent = np.ptp(np.vstack((predicted, ground_truth)), axis=0).max()
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.1, extent * 0.1))
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window(
                window_name='Global maps — green: ground truth, red: VGGT predicted')
            for geometry in (gt_cloud, predicted_cloud, axes):
                visualizer.add_geometry(geometry)
            visualizer.get_render_option().point_size *= 0.5
            visualizer.run()
            visualizer.destroy_window()
    else:
        print(f'Global map data not found: {maps_path}. Rerun offline evaluation to create it.')

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    summary = [
        'VGGT VISIBLE CLOUD VS GROUND TRUTH',
        f'Metrics CSV: {csv_path}',
        f'Evaluation rows: {len(df)}',
        f'Voxel size: {cloud["voxel_size"]:.4f} m',
        f'Accuracy mean: {cloud["accuracy_mean"]:.4f} m',
        f'Accuracy RMSE: {cloud["accuracy_rmse"]:.4f} m',
        f'Completeness mean: {cloud["completeness_mean"]:.4f} m',
        f'Completeness RMSE: {cloud["completeness_rmse"]:.4f} m',
        f'Chamfer distance: {cloud["chamfer"]:.4f} m',
        f'F-score: {cloud["fscore"]:.4f}',
    ]
    with open(output_text, 'w') as stream:
        stream.write('\n'.join(summary))
    print(f'Plot successfully saved to: {os.path.abspath(output_image)}')
    print(f'Summary report successfully saved to: {os.path.abspath(output_text)}')


def latest_metrics_files(directory='metrics/pipeline'):
    files = glob.glob(os.path.join(directory, 'metrics_*.csv'))
    regular = [path for path in files
               if not path.endswith(('_cloud.csv', '_cloud_global.csv'))]
    global_cloud = [path for path in files if path.endswith('_cloud_global.csv')]
    return [max(paths, key=os.path.getmtime) for paths in (regular, global_cloud) if paths]


def main(csv_path):

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

    if {'accuracy_mean', 'completeness_mean', 'chamfer', 'fscore'} <= set(df.columns):
        plot_cloud_metrics(csv_path, df)
        return

    csv_basename = os.path.basename(csv_path)
    csv_name_no_ext = os.path.splitext(csv_basename)[0]
    cloud_csv_path = f'{os.path.splitext(csv_path)[0]}_cloud.csv'
    cloud_df = pd.read_csv(cloud_csv_path) if os.path.exists(cloud_csv_path) else None

    eval_dir = 'metrics/eval'
    os.makedirs(eval_dir, exist_ok=True)

    output_image = os.path.join(eval_dir, f"{csv_name_no_ext}_plot.png")
    output_text = os.path.join(eval_dir, f"{csv_name_no_ext}_summary.txt")

    t_start = df['timestamp'].iloc[0]
    time_sec = df['timestamp'] - t_start

    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle('VGGT-Omega Calibration & Depth Estimation Evaluation Metrics', fontsize=16, fontweight='bold')

    # 1. VGGT Depth Scale Alignment
    ax = axs[0, 0]
    if 'scale' in df.columns:
        ax.plot(time_sec, df['scale'], label='Depth Scale Factor', color='#3498db', linewidth=2)
        ax.set_title('VGGT Depth Scale Alignment', fontweight='bold')
        ax.set_ylabel('Scale Factor (s = median(D_metric / D_pred))')
        median_s = df['scale'].median()
        ax.axhline(median_s, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Median: {median_s:.3f}')
    else:
        ax.plot(time_sec, np.zeros(len(df)), label='scale (missing)', color='#3498db', alpha=0.8)
        ax.set_title('Scale (missing)', fontweight='bold')
        ax.set_ylabel('N/A')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 2. Estimated Translation Components
    ax = axs[0, 1]
    for col, label, color in [('t_x', 't_x (Left/Right)', '#2ecc71'),
                              ('t_y', 't_y (Up/Down)', '#3498db'),
                              ('t_z', 't_z (Forward/Backward)', '#e67e22')]:
        if col in df.columns:
            ax.plot(time_sec, df[col], label=label, color=color, alpha=0.8, linewidth=2)
    ax.set_title('Estimated Camera Extrinsics (Translation)', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Distance (meters)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # 3. Head Camera Depth Error
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

    # 4. Exo Camera Depth Error
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
            ax_rot.plot(time_sec, df['error_r_deg'], color='#2c3e50', label='Rotation Error (deg)',
                        linestyle='-', alpha=0.8)
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

    # 7. Head VGGT Depth-Confidence Percentiles
    ax = axs[3, 0]
    has_h_conf = any(c in df.columns for c in ['head_conf_p50', 'head_conf_p95'])
    if has_h_conf:
        if 'head_conf_p50' in df.columns:
            ax.plot(time_sec, df['head_conf_p50'], label='Head conf p50', color='#16a085', linewidth=2)
        if 'head_conf_p95' in df.columns:
            ax.plot(time_sec, df['head_conf_p95'], label='Head conf p95', color='#c0392b', linewidth=2)
        if 'conf_thr' in df.columns:
            ax.plot(time_sec, df['conf_thr'], label='Conf threshold', color='#7f8c8d',
                    linestyle='--', alpha=0.8)
        ax.set_title('VGGT Depth-Conf (Head) Percentiles', fontweight='bold')
        ax.set_ylabel('Confidence (1, +inf)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, "No Head VGGT conf stats\n(version < refactor)", ha='center', va='center')
        ax.set_axis_off()
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 8. Exo VGGT Depth-Confidence Percentiles
    ax = axs[3, 1]
    has_e_conf = any(c in df.columns for c in ['exo_conf_p50', 'exo_conf_p95'])
    if has_e_conf:
        if 'exo_conf_p50' in df.columns:
            ax.plot(time_sec, df['exo_conf_p50'], label='Exo conf p50', color='#2980b9', linewidth=2)
        if 'exo_conf_p95' in df.columns:
            ax.plot(time_sec, df['exo_conf_p95'], label='Exo conf p95', color='#d35400', linewidth=2)
        if 'conf_thr' in df.columns:
            ax.plot(time_sec, df['conf_thr'], label='Conf threshold', color='#7f8c8d',
                    linestyle='--', alpha=0.8)
        ax.set_title('VGGT Depth-Conf (Exo) Percentiles', fontweight='bold')
        ax.set_ylabel('Confidence (1, +inf)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, "No Exo VGGT conf stats\n(version < refactor)", ha='center', va='center')
        ax.set_axis_off()
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    plt.savefig(output_image, dpi=150)
    print(f"Plot successfully saved to: {os.path.abspath(output_image)}")

    # Compute Stats and Generate Text Summary
    total_frames = len(df)
    status_freq = df['status'].value_counts()
    status_pct = df['status'].value_counts(normalize=True) * 100.0

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

    if 'cycle_time_sec' in df.columns and df['cycle_time_sec'].notna().any():
        cycle_times = df['cycle_time_sec'].dropna()
        summary_lines.extend([
            "-----------------------------------------",
            "PIPELINE CYCLE COMPUTE TIME",
            "-----------------------------------------",
            f"Average: {cycle_times.mean():>8.3f} s",
            f"Minimum: {cycle_times.min():>8.3f} s",
            f"Maximum: {cycle_times.max():>8.3f} s",
            "",
        ])

    if num_success > 0:
        summary_lines.append("-----------------------------------------")
        summary_lines.append("ESTIMATED EXTRINSICS STATS (SUCCESS ONLY)")
        summary_lines.append("-----------------------------------------")
        for col, label in [('t_x', 't_x (Left/Right)'), ('t_y', 't_y (Up/Down)'),
                           ('t_z', 't_z (Forward/Backward)')]:
            if col not in success_df.columns:
                continue
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
            summary_lines.append("Scale Factor:")
            summary_lines.append(f"  Mean:     {scales.mean():>8.4f}")
            summary_lines.append(f"  Median:   {scales.median():>8.4f}")
            summary_lines.append(f"  Variance: {scales.var() if num_success > 1 else 0.0:>8.6f}")
            summary_lines.append(f"  Min/Max:  {scales.min():>8.4f} / {scales.max():>8.4f}")
            summary_lines.append("")

        has_depth_metrics = 'head_depth_rmse' in success_df.columns
        if has_depth_metrics:
            summary_lines.append("-----------------------------------------")
            summary_lines.append("VGGT VS GT DEPTH ESTIMATION ERROR")
            summary_lines.append("-----------------------------------------")
            for cam, prefix in [('Head Camera', 'head_depth'), ('Exo Camera', 'exo_depth')]:
                rmse_col = f"{prefix}_rmse"
                mae_col = f"{prefix}_mae"
                if rmse_col not in success_df.columns:
                    continue
                rmses = success_df[rmse_col].dropna()
                maes = success_df[mae_col].dropna()
                if len(rmses) > 0:
                    summary_lines.append(f"{cam}:")
                    summary_lines.append(f"  RMSE Mean:     {rmses.mean():>8.4f} m")
                    summary_lines.append(f"  RMSE Variance: {rmses.var() if len(rmses) > 1 else 0.0:>8.6f} m^2")
                    summary_lines.append(f"  MAE Mean:      {maes.mean():>8.4f} m")
                    summary_lines.append(f"  MAE Variance:  {maes.var() if len(maes) > 1 else 0.0:>8.6f} m^2")
            summary_lines.append("")

        has_conf_stats = any(c in success_df.columns for c in
                             ['head_conf_p50', 'head_conf_p95', 'exo_conf_p50', 'exo_conf_p95'])
        if has_conf_stats:
            summary_lines.append("-----------------------------------------")
            summary_lines.append("VGGT DEPTH-HEAD CONFIDENCE STATS (SUCCESS ONLY)")
            summary_lines.append("-----------------------------------------")
            for cam, p50_col, p95_col in [('Head Camera', 'head_conf_p50', 'head_conf_p95'),
                                          ('Exo Camera', 'exo_conf_p50', 'exo_conf_p95')]:
                if p50_col not in success_df.columns and p95_col not in success_df.columns:
                    continue
                p50 = success_df[p50_col].dropna() if p50_col in success_df.columns else None
                p95 = success_df[p95_col].dropna() if p95_col in success_df.columns else None
                summary_lines.append(f"{cam}:")
                if p50 is not None and len(p50) > 0:
                    summary_lines.append(f"  p50 Mean: {p50.mean():>8.4f}  Median: {p50.median():>8.4f}")
                if p95 is not None and len(p95) > 0:
                    summary_lines.append(f"  p95 Mean: {p95.mean():>8.4f}  Median: {p95.median():>8.4f}")
            if 'conf_thr' in success_df.columns:
                thr = success_df['conf_thr'].dropna()
                if len(thr) > 0:
                    summary_lines.append(f"Conf threshold (Mean): {thr.mean():>8.4f}")
            summary_lines.append("")

        success_has_gt = 'error_t' in success_df.columns and success_df['error_t'].notna().any()
        if success_has_gt:
            summary_lines.append("-----------------------------------------")
            summary_lines.append("ACCURACY VS GROUND TRUTH (SUCCESS ONLY)")
            summary_lines.append("-----------------------------------------")
            err_t = success_df['error_t'].dropna()
            summary_lines.append("Translation Error:")
            summary_lines.append(f"  Mean:     {err_t.mean():>8.4f} m")
            summary_lines.append(f"  Max:      {err_t.max():>8.4f} m")
            summary_lines.append(f"  Variance: {err_t.var() if len(err_t) > 1 else 0.0:>8.6f} m^2")
            if 'error_r_deg' in success_df.columns:
                err_r = success_df['error_r_deg'].dropna()
                summary_lines.append("Rotation Error:")
                summary_lines.append(f"  Mean:     {err_r.mean():>8.4f} deg")
                summary_lines.append(f"  Max:      {err_r.max():>8.4f} deg")
                summary_lines.append(f"  Variance: {err_r.var() if len(err_r) > 1 else 0.0:>8.6f} deg^2")
            summary_lines.append("")
    else:
        summary_lines.append("No successful frames to compute extrinsics statistics.")

    if cloud_df is not None and not cloud_df.empty:
        cloud = cloud_df.mean(numeric_only=True)
        summary_lines.extend([
            "",
            "-----------------------------------------",
            "VGGT VISIBLE CLOUD VS GROUND TRUTH",
            "-----------------------------------------",
            f"Cloud Metrics CSV: {cloud_csv_path}",
            f"Synchronized Frames: {len(cloud_df)}",
            f"Voxel Size:        {cloud['voxel_size']:>8.4f} m",
            f"Accuracy Mean:     {cloud['accuracy_mean']:>8.4f} m",
            f"Accuracy RMSE:     {cloud['accuracy_rmse']:>8.4f} m",
            f"Completeness Mean: {cloud['completeness_mean']:>8.4f} m",
            f"Completeness RMSE: {cloud['completeness_rmse']:>8.4f} m",
            f"Chamfer Distance:  {cloud['chamfer']:>8.4f} m",
            f"F-score:           {cloud['fscore']:>8.4f}",
        ])

    try:
        with open(output_text, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"Summary report successfully saved to: {os.path.abspath(output_text)}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

if __name__ == '__main__':
    paths = [sys.argv[1]] if len(sys.argv) > 1 else latest_metrics_files()
    if not paths:
        paths = ['extrinsic_metrics.csv']
    else:
        print(f'Plotting: {", ".join(paths)}')
    for path in paths:
        main(path)
    if os.environ.get('DISPLAY', '').strip():
        plt.show()
