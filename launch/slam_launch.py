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
    common_config_path = os.path.join(config_dir, 'common.yaml')
    slam_params = load_section(common_config_path, 'slam')

    mode = slam_params.get('mode', 'dual_vo')

    actions = [use_sim_time_arg]

    # Identity map→odom bootstrap (needed for both modes; dual_vo also publishes it)
    static_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom']
    )
    actions.append(static_map_odom)

    if mode == 'dual_vo':
        dual_vo_node = Node(
            package='exo_head_slam',
            executable='dual_vo',
            name='dual_vo',
            parameters=[slam_params, {'use_sim_time': use_sim_time}],
            remappings=[
                ('head/rgb/image', '/head/masked/image_raw'),
                ('head/depth/image', '/head/masked/depth_raw'),
                ('head/rgb/camera_info', '/camera/head/color/camera_info'),
                ('exo/rgb/image', '/exo/masked/image_raw'),
                ('exo/depth/image', '/exo/masked/depth_raw'),
                ('exo/rgb/camera_info', '/camera/exo/color/camera_info'),
            ],
            output='screen'
        )
        actions.append(dual_vo_node)
    else:
        # nvblox_tracking or other modes: connect odom→exo_link so TF tree is complete
        fallback_vo_node = Node(
            package='exo_head_slam',
            executable='fallback_vo',
            name='fallback_vo',
            parameters=[slam_params, {'use_sim_time': use_sim_time}],
            remappings=[
                ('rgb/image', '/exo/masked/image_raw'),
                ('depth/image', '/exo/masked/depth_raw'),
                ('rgb/camera_info', '/camera/exo/color/camera_info'),
            ],
            output='screen'
        )
        actions.append(fallback_vo_node)

    return LaunchDescription(actions)
