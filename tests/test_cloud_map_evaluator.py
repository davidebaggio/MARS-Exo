import os

import numpy as np
import pandas as pd

from exo_head_slam.cloud_map_evaluator_node import cloud_metrics, merge_voxels
from plot_metrics import latest_metrics_files, main as plot_main, plot_cloud_metrics


def test_global_voxel_merge_and_metrics():
    voxels = {}
    merge_voxels(voxels, np.array([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]), 0.1)
    merge_voxels(voxels, np.array([[0.02, 0.0, 0.0], [2.0, 0.0, 0.0]]), 0.1)

    global_map = np.asarray(list(voxels.values()))
    assert len(global_map) == 3
    assert cloud_metrics(global_map, global_map.copy(), 0.1)['fscore'] == 1.0


def test_global_cloud_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('DISPLAY', raising=False)
    csv_path = tmp_path / 'cloud_global.csv'
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    np.savez_compressed(tmp_path / 'cloud_global_maps.npz', predicted=points,
                        ground_truth=points)
    metrics = cloud_metrics(points, points, 0.1)
    frame = pd.DataFrame([{'voxel_size': 0.1, **metrics}])

    plot_cloud_metrics(str(csv_path), frame)

    assert (tmp_path / 'metrics/eval/cloud_global_plot.png').exists()
    assert (tmp_path / 'metrics/eval/cloud_global_maps.ply').exists()


def test_latest_metrics_selects_regular_and_global(tmp_path):
    for name in ('metrics_1.csv', 'metrics_1_cloud_global.csv',
                 'metrics_2.csv', 'metrics_2_cloud.csv', 'metrics_2_cloud_global.csv'):
        (tmp_path / name).touch()
        os.utime(tmp_path / name, (1, 1))
    os.utime(tmp_path / 'metrics_2.csv', (2, 2))
    os.utime(tmp_path / 'metrics_2_cloud_global.csv', (2, 2))

    assert latest_metrics_files(tmp_path) == [
        str(tmp_path / 'metrics_2.csv'),
        str(tmp_path / 'metrics_2_cloud_global.csv'),
    ]


def test_cycle_time_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('DISPLAY', raising=False)
    csv_path = tmp_path / 'metrics_1.csv'
    pd.DataFrame({
        'timestamp': [1.0, 2.0],
        'status': ['SUCCESS', 'SUCCESS'],
        'cycle_time_sec': [1.0, 3.0],
        't_x': [0.0, 0.0], 't_y': [0.0, 0.0], 't_z': [1.0, 1.0],
        'head_depth_rmse': [0.1, 0.1], 'head_depth_mae': [0.1, 0.1],
        'exo_depth_rmse': [0.1, 0.1], 'exo_depth_mae': [0.1, 0.1],
    }).to_csv(csv_path, index=False)

    plot_main(str(csv_path))

    summary = (tmp_path / 'metrics/eval/metrics_1_summary.txt').read_text()
    assert 'Average:    2.000 s' in summary
    assert 'Minimum:    1.000 s' in summary
    assert 'Maximum:    3.000 s' in summary
