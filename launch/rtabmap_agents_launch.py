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
    Launches a single RTAB-Map instance on the head camera for VO/SLAM.
    Dense mapping is disabled since NVBlox handles 3D reconstruction.
    """
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_dir = os.path.join(get_package_share_directory('exo_head_slam'), 'config')
    config_path = os.path.join(config_dir, 'head.yaml')
    head_params = load_section(config_path, 'head_rtabmap')

    runtime_params = {
        'subscribe_depth': head_params['subscribe_depth'],
        'frame_id': head_params['frame_id'],
        'approx_sync': head_params['approx_sync'],
        'wait_for_transform': head_params['wait_for_transform'],
        'use_sim_time': use_sim_time,
    }

    head_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='head_rtabmap',
        parameters=[runtime_params],
        remappings=[
            ('rgb/image', head_params['rgb_topic']),
            ('depth/image', head_params['depth_topic']),
            ('rgb/camera_info', head_params['camera_info_topic']),
        ],
        arguments=['-d'],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        head_rtabmap,
    ])
