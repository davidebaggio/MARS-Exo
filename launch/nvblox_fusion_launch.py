import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def load_section(config_path: str, section: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    return (data.get(section, {}) or {}).get('ros__parameters', {})

def generate_launch_description():
    """
    Launches a single NVBlox node consuming masked depth from both cameras.
    Uses the TF tree (map -> head, map -> head -> exo) for voxel integration.
    """
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_dir = os.path.join(get_package_share_directory('exo_head_slam'), 'config')
    head_config_path = os.path.join(config_dir, 'head.yaml')
    exo_config_path = os.path.join(config_dir, 'exo.yaml')
    head_params = load_section(head_config_path, 'head_nvblox')
    exo_params = load_section(exo_config_path, 'exo_nvblox')

    nvblox_params = {
        'global_frame': head_params.get('global_frame', 'map'),
        'use_sim_time': use_sim_time,
        'num_cameras': 2,
    }

    nvblox_node = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='nvblox',
        parameters=[nvblox_params],
        remappings=[
            ('camera_0/depth/image', head_params['depth_topic']),
            ('camera_0/depth/camera_info', head_params['camera_info_topic']),
            ('camera_1/depth/image', exo_params['depth_topic']),
            ('camera_1/depth/camera_info', exo_params['camera_info_topic']),
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        nvblox_node,
    ])
