#!/usr/bin/env python3
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt


EVAL_DIR = os.environ.get('EVAL_DIR', 'metrics/lightglue/eval')


def output_paths(csv_path):
    os.makedirs(EVAL_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    return (
        os.path.join(EVAL_DIR, f'{name}_plot.png'),
        os.path.join(EVAL_DIR, f'{name}_summary.txt'),
    )


def plot_cloud_metrics(csv_path, frame):
    output_image, output_text = output_paths(csv_path)
    cloud = frame.mean(numeric_only=True)
    columns = [
        'accuracy_mean', 'accuracy_rmse', 'completeness_mean',
        'completeness_rmse', 'chamfer',
    ]
    labels = ['Accuracy mean', 'Accuracy RMSE', 'Completeness mean',
              'Completeness RMSE', 'Chamfer']
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, [cloud[column] for column in columns])
    axis.bar_label(bars, fmt='%.3f')
    axis.set_ylabel('Distance (m)')
    axis.set_title(
        f'LightGlue combined cloud vs ground truth; '
        f'F@10cm={cloud["fscore"]:.3f}'
    )
    axis.tick_params(axis='x', rotation=20)
    axis.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_image, dpi=150)
    plt.close(fig)

    summary = [
        'LIGHTGLUE COMBINED CLOUD VS GROUND TRUTH',
        f'Metrics CSV: {csv_path}',
        f'Evaluation rows: {len(frame)}',
        f'Voxel size: {cloud["voxel_size"]:.4f} m',
        f'Accuracy mean: {cloud["accuracy_mean"]:.4f} m',
        f'Accuracy RMSE: {cloud["accuracy_rmse"]:.4f} m',
        f'Completeness mean: {cloud["completeness_mean"]:.4f} m',
        f'Completeness RMSE: {cloud["completeness_rmse"]:.4f} m',
        f'Chamfer distance: {cloud["chamfer"]:.4f} m',
        f'F@10cm: {cloud["fscore"]:.4f}',
    ]
    for threshold in ('02', '05', '10'):
        column = f'fscore_{threshold}cm'
        if column in cloud:
            summary.append(f'F@{int(threshold)}cm: {cloud[column]:.4f}')
    with open(output_text, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(summary))


def plot_solver_metrics(csv_path, frame):
    output_image, output_text = output_paths(csv_path)
    seconds = frame['timestamp'] - frame['timestamp'].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LightGlue + RANSAC extrinsic evaluation', fontsize=16)

    axis = axes[0, 0]
    for column, label in (
        ('num_2d_matches', '2D matches'),
        ('num_3d_matches', '3D matches'),
        ('inliers', 'RANSAC inliers'),
    ):
        axis.plot(seconds, frame[column], label=label)
    axis.set_title('Matches')
    axis.legend()
    axis.grid(True, linestyle='--', alpha=0.5)

    axis = axes[0, 1]
    axis.plot(seconds, frame['rmse'], color='tab:red', label='RMSE (m)')
    ratio_axis = axis.twinx()
    ratio_axis.plot(
        seconds, frame['inlier_ratio'], color='tab:purple',
        linestyle='--', label='Inlier ratio',
    )
    axis.set_title('RANSAC quality')
    axis.set_ylabel('RMSE (m)')
    ratio_axis.set_ylabel('Inlier ratio')
    axis.grid(True, linestyle='--', alpha=0.5)

    axis = axes[1, 0]
    if frame['error_t'].notna().any():
        axis.plot(seconds, frame['error_t'], color='tab:red',
                  label='Translation error (m)')
        rotation_axis = axis.twinx()
        rotation_axis.plot(
            seconds, frame['error_r_deg'], color='tab:blue',
            label='Rotation error (deg)',
        )
        axis.set_title('Extrinsic error')
        rotation_axis.set_ylabel('Degrees')
    else:
        for column in ('t_x', 't_y', 't_z'):
            axis.plot(seconds, frame[column], label=column)
        axis.set_title('Estimated translation')
        axis.legend()
    axis.grid(True, linestyle='--', alpha=0.5)

    axis = axes[1, 1]
    counts = frame['status'].value_counts()
    counts.plot(kind='barh', ax=axis)
    axis.set_title('Solver status')
    axis.grid(True, axis='x', linestyle='--', alpha=0.5)

    for axis in axes.flat:
        axis.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output_image, dpi=150)
    plt.close(fig)

    success = frame[frame['status'] == 'SUCCESS']
    summary = [
        'LIGHTGLUE + RANSAC EXTRINSIC EVALUATION',
        f'Metrics CSV: {csv_path}',
        f'Attempts: {len(frame)}',
        f'Successes: {len(success)} ({100.0 * len(success) / len(frame):.2f}%)',
    ]
    for status, count in frame['status'].value_counts().items():
        summary.append(f'{status}: {count}')
    for column, label in (
        ('num_2d_matches', '2D matches'),
        ('num_3d_matches', '3D matches'),
        ('inliers', 'RANSAC inliers'),
        ('inlier_ratio', 'Inlier ratio'),
        ('rmse', 'RANSAC RMSE (m)'),
        ('cycle_time_sec', 'Cycle time (s)'),
        ('error_t', 'Translation error (m)'),
        ('error_r_deg', 'Rotation error (deg)'),
    ):
        if column in success and success[column].notna().any():
            values = success[column].dropna()
            summary.append(
                f'{label}: mean={values.mean():.6f}, '
                f'median={values.median():.6f}, max={values.max():.6f}'
            )
    with open(output_text, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(summary))


def main(csv_path):
    if not os.path.exists(csv_path):
        raise SystemExit(f'Metrics file not found: {csv_path}')
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise SystemExit(f'Metrics file is empty: {csv_path}')
    cloud_columns = {'accuracy_mean', 'completeness_mean', 'chamfer', 'fscore'}
    if cloud_columns <= set(frame.columns):
        plot_cloud_metrics(csv_path, frame)
    else:
        plot_solver_metrics(csv_path, frame)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'extrinsic_metrics.csv')
