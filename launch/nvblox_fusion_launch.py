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
    Launches nvblox nodes for volumetric fusion.
    Assumes nvblox_ros is installed.
    """
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_dir = os.path.join(get_package_share_directory('exo_head_slam'), 'config')
    config_path = os.path.join(config_dir, 'head.yaml')
    exo_config_path = os.path.join(config_dir, 'exo.yaml')
    head_params = load_section(config_path, 'head_nvblox')
    exo_params = load_section(exo_config_path, 'exo_nvblox')

    head_runtime_params = {
        'global_frame': head_params.get('global_frame', 'head_camera_link'),
        'use_depth': head_params.get('use_depth', True),
        'use_color': head_params.get('use_color', True),
        'use_sim_time': use_sim_time,
    }
    exo_runtime_params = {
        'global_frame': exo_params.get('global_frame', 'exo_camera_link'),
        'use_depth': exo_params.get('use_depth', True),
        'use_color': exo_params.get('use_color', True),
        'use_sim_time': use_sim_time,
    }

    # If configured, launch a single NVBlox node consuming merged topics
    use_merged = head_params.get('use_merged', False) or exo_params.get('use_merged', False)
    if use_merged:
        merged_depth = head_params.get('merged_depth_topic', '/merged/depth/image_raw')
        merged_color = head_params.get('merged_color_topic', '/merged/color/image_raw')
        merged_info = head_params.get('merged_info_topic', '/merged/camera_info')

        merged_params = {
            'global_frame': head_runtime_params['global_frame'],
            'use_depth': head_runtime_params['use_depth'],
            'use_color': head_runtime_params['use_color'],
            'use_sim_time': use_sim_time,
        }

        merged_nvblox = Node(
            package='nvblox_ros',
            executable='nvblox_node',
            name='merged_nvblox',
            parameters=[merged_params],
            remappings=[
                ('depth/image', merged_depth),
                ('color/image', merged_color),
                ('camera_info', merged_info),
            ],
            output='screen'
        )

        return LaunchDescription([use_sim_time_arg, merged_nvblox])

    # Default: per-camera NVBlox nodes
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
        use_sim_time_arg,
        head_nvblox,
        exo_nvblox
    ])
