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

    global_frame_arg = DeclareLaunchArgument(
        'global_frame',
        default_value=head_params.get('global_frame', 'map'),
        description='Global frame for voxel integration'
    )
    global_frame = LaunchConfiguration('global_frame')

    nvblox_params = {
        'global_frame': global_frame,
        'voxel_size_m': head_params.get('voxel_size_m', 0.05),
        'max_integration_distance_m': head_params.get('max_integration_distance_m', 5.0),
        'mesh_update_period': head_params.get('mesh_update_period', 10),
        'costmap_resolution': head_params.get('costmap_resolution', 0.1),
        'costmap_height_min': head_params.get('costmap_height_min', 0.0),
        'costmap_height_max': head_params.get('costmap_height_max', 1.0),
        'head_depth_topic': head_params['depth_topic'],
        'head_rgb_topic': head_params['rgb_topic'],
        'head_camera_info_topic': head_params['camera_info_topic'],
        'exo_depth_topic': exo_params['depth_topic'],
        'exo_rgb_topic': exo_params['rgb_topic'],
        'exo_camera_info_topic': exo_params['camera_info_topic'],
        'head_frame_id': head_params.get('head_frame_id', 'head_link'),
        'exo_frame_id': exo_params.get('exo_frame_id', 'exo_link'),
        'use_sim_time': use_sim_time,
    }

    nvblox_node = Node(
        package='exo_head_slam',
        executable='nvblox_node',
        name='nvblox',
        parameters=[nvblox_params],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        global_frame_arg,
        nvblox_node,
    ])
