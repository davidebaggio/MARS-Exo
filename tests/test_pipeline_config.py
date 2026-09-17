from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_rtabmap_uses_raw_rgb_and_robust_odometry_profile():
    config = yaml.safe_load((ROOT / 'config/exo.yaml').read_text())
    parameters = config['exo_rtabmap']['ros__parameters']
    slam = config['exo_slam']['ros__parameters']

    assert parameters['rgb_topic'] == '/camera/exo/color/image_raw'
    assert parameters['depth_topic'] == '/exo/filtered/depth_raw'
    assert parameters['OdomF2M/BundleAdjustment'] == '0'
    assert parameters['Odom/ResetCountdown'] == '15'
    assert parameters['Vis/MinInliers'] == '8'
    assert parameters['RGBD/OptimizeMaxError'] == '3.0'
    assert parameters['Mem/SaveDepth16Format'] == 'true'
    assert parameters['Vis/FeatureType'] == '8'
    assert parameters['Rtabmap/DetectionRate'] == '1'
    assert slam['Mem/UseOdomFeatures'] == 'false'
    assert slam['Kp/DetectorStrategy'] == '1'
    assert slam['Kp/MaxFeatures'] == '500'
    assert slam['RGBD/LoopClosureReextractFeatures'] == 'true'
    assert slam['Vis/FeatureType'] == '1'
    assert slam['Vis/MaxFeatures'] == '500'

    launch = (ROOT / 'launch/rtabmap_agents_launch.py').read_text()
    assert "'subscribe_rgbd': True" in launch
    assert "('rgbd_image', 'odom_rgbd_image')" in launch


def test_accuracy_defaults_disable_motion_unsafe_depth_filtering():
    head = yaml.safe_load((ROOT / 'config/head.yaml').read_text())[
        'head_depth_preprocessor'
    ]['ros__parameters']
    exo = yaml.safe_load((ROOT / 'config/exo.yaml').read_text())[
        'exo_depth_preprocessor'
    ]['ros__parameters']

    assert head['depth_filter.spatial.enabled'] is False
    assert exo['depth_filter.spatial.enabled'] is True
    assert head['depth_filter.temporal.enabled'] is False
    assert exo['depth_filter.temporal.enabled'] is False
