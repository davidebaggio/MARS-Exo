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
    Launches nvblox nodes for volumetric fusion.
    Assumes nvblox_ros is installed.
    """
    config_dir = os.path.join(get_package_share_directory('exo_head_slam'), 'config')
    config_path = os.path.join(config_dir, 'head.yaml')
    exo_config_path = os.path.join(config_dir, 'exo.yaml')
    head_params = load_section(config_path, 'head_nvblox')
    exo_params = load_section(exo_config_path, 'exo_nvblox')

    head_runtime_params = {
        'global_frame': head_params['global_frame'],
        'use_depth': head_params['use_depth'],
        'use_color': head_params['use_color'],
    }
    exo_runtime_params = {
        'global_frame': exo_params['global_frame'],
        'use_depth': exo_params['use_depth'],
        'use_color': exo_params['use_color'],
    }

    head_nvblox = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='head_nvblox',
        namespace='head',
        parameters=[head_runtime_params],
        remappings=[
            ('depth/image', head_params['depth_topic']),
            ('color/image', head_params['color_topic']),
            ('camera_info', head_params['camera_info_topic']),
        ],
        output='screen'
    )

    exo_nvblox = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='exo_nvblox',
        namespace='exo',
        parameters=[exo_runtime_params],
        remappings=[
            ('depth/image', exo_params['depth_topic']),
            ('color/image', exo_params['color_topic']),
            ('camera_info', exo_params['camera_info_topic']),
        ],
        output='screen'
    )

    return LaunchDescription([
        head_nvblox,
        exo_nvblox
    ])
