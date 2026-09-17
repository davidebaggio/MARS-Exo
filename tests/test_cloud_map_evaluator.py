import numpy as np
import pandas as pd

from exo_head_slam.cloud_map_evaluator_node import cloud_metrics, merge_voxels
from plot_metrics import main as plot_main, plot_cloud_metrics


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
    metrics = cloud_metrics(points, points, 0.1, extra_thresholds=(0.02, 0.05, 0.10))
    assert all(metrics[f'fscore_{threshold:02d}cm'] == 1.0 for threshold in (2, 5, 10))
    frame = pd.DataFrame([{'voxel_size': 0.1, **metrics}])

    plot_cloud_metrics(str(csv_path), frame)

    output = tmp_path / 'metrics/lightglue/eval'
    assert (output / 'cloud_global_plot.png').exists()
    assert 'F@2cm: 1.0000' in (output / 'cloud_global_summary.txt').read_text()


def test_cycle_time_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('DISPLAY', raising=False)
    csv_path = tmp_path / 'metrics_1.csv'
    pd.DataFrame({
        'timestamp': [1.0, 2.0],
        'status': ['SUCCESS', 'SUCCESS'],
        'cycle_time_sec': [1.0, 3.0],
        'num_2d_matches': [20, 20], 'num_3d_matches': [15, 15],
        'inliers': [12, 12], 'inlier_ratio': [0.8, 0.8],
        'rmse': [0.01, 0.01], 'error_t': [0.1, 0.1],
        'error_r_deg': [1.0, 1.0],
        't_x': [0.0, 0.0], 't_y': [0.0, 0.0], 't_z': [1.0, 1.0],
    }).to_csv(csv_path, index=False)

    plot_main(str(csv_path))

    summary = (
        tmp_path / 'metrics/lightglue/eval/metrics_1_summary.txt'
    ).read_text()
    assert 'Cycle time (s): mean=2.000000' in summary
