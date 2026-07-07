import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction


def load_section(config_path: str, section: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    return (data.get(section, {}) or {}).get('ros__parameters', {})


def generate_launch_description():
    """
    Launches a single RTAB-Map SLAM instance on the exo camera.
    rtabmap_slam computes its own visual odometry internally (no external odom node).
    Publishes map->odom and odom->exo_link TFs.
    """
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    pkg_share = get_package_share_directory('exo_head_slam')
    config_dir = os.path.join(pkg_share, 'config')

    exo_config_path = os.path.join(config_dir, 'exo.yaml')

    exo_params = load_section(exo_config_path, 'exo_rtabmap')

    # Extract topic names for remapping, don't pass them as ROS params
    # (they'd conflict with the remapping mechanism in rtabmap_slam)
    rgb_topic = exo_params.pop('rgb_topic')
    depth_topic = exo_params.pop('depth_topic')
    camera_info_topic = exo_params.pop('camera_info_topic')

    # Base parameters for all nodes
    base_params = {
        'use_sim_time': use_sim_time,
        'qos_image': 2,
        'qos_depth': 2,
        'qos_camera_info': 2,
    }

    # SLAM Node (Exo) with internal visual odometry
    exo_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='exo_rtabmap',
        parameters=[{**base_params, **exo_params}],
        remappings=[
            ('rgb/image', rgb_topic),
            ('depth/image', depth_topic),
            ('rgb/camera_info', camera_info_topic),
            ('depth/camera_info', camera_info_topic),
            ('grid_map', '/map'),
        ],
        output='screen'
    )

    # Map Assembler: subscribes to core SLAM's map_graph and publishes /exo_rtabmap/cloud_map.
    # Delayed 5s so rtabmap/get_map_data service is available at startup (avoids WARN
    # and ensures full cloud map appears immediately rather than growing incrementally).
    map_assembler = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rtabmap_util',
                executable='map_assembler',
                name='map_assembler',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[
                    ('map_graph', '/exo_rtabmap/map_graph'),
                    ('cloud_map', '/exo_rtabmap/cloud_map'),
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        exo_rtabmap,
        map_assembler,
    ])