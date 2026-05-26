import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def load_section(config_path: str, section: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    return (data.get(section, {}) or {}).get('ros__parameters', {})

def generate_launch_description():
    """
    Launches RTAB-Map instances for both head and exoskeleton.
    Assumes rtabmap_ros is installed.
    """
    config_dir = os.path.join(get_package_share_directory('exo_head_slam'), 'config')
    config_path = os.path.join(config_dir, 'head.yaml')
    exo_config_path = os.path.join(config_dir, 'exo.yaml')
    head_params = load_section(config_path, 'head_rtabmap')
    exo_params = load_section(exo_config_path, 'exo_rtabmap')

    head_runtime_params = {
        'subscribe_depth': head_params['subscribe_depth'],
        'frame_id': head_params['frame_id'],
        'approx_sync': head_params['approx_sync'],
        'wait_for_transform': head_params['wait_for_transform'],
    }
    exo_runtime_params = {
        'subscribe_depth': exo_params['subscribe_depth'],
        'frame_id': exo_params['frame_id'],
        'approx_sync': exo_params['approx_sync'],
        'wait_for_transform': exo_params['wait_for_transform'],
    }

    head_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='head_rtabmap',
        namespace='head',
        parameters=[head_runtime_params],
        remappings=[
            ('rgb/image', head_params['rgb_topic']),
            ('depth/image', head_params['depth_topic']),
            ('rgb/camera_info', head_params['camera_info_topic']),
        ],
        output='screen'
    )

    exo_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='exo_rtabmap',
        namespace='exo',
        parameters=[exo_runtime_params],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
        ],
        output='screen'
    )

    return LaunchDescription([
        head_rtabmap,
        exo_rtabmap
    ])
