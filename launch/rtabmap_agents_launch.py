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
    Launches dual RTAB-Map odometry and an EKF to fuse them.
    A single RTAB-Map instance runs on the exo camera for SLAM.
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
    head_config_path = os.path.join(config_dir, 'head.yaml')
    
    exo_params = load_section(exo_config_path, 'exo_rtabmap')
    exo_odom_params = load_section(exo_config_path, 'exo_rgbd_odometry')
    
    head_params = load_section(head_config_path, 'head_rtabmap')
    head_odom_params = load_section(head_config_path, 'head_rgbd_odometry')
    
    # Base parameters for all nodes
    base_params = {
        'use_sim_time': use_sim_time,
        'qos_image': 2,
        'qos_depth': 2,
        'qos_camera_info': 2,
    }

    # Exo Visual Odometry Node (Python Fallback)
    exo_odometry = Node(
        package='exo_head_slam',
        executable='fallback_vo',
        name='exo_rgbd_odometry',
        parameters=[{**base_params, **exo_odom_params}],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
            ('odom', '/exo/odom'),
        ],
        output='screen'
    )

    # SLAM Node (Exo)
    exo_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='exo_rtabmap',
        parameters=[{**base_params, **exo_params}],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
            ('odom', '/exo/odom'),
        ],
        arguments=['-d'],
        output='screen'
    )

    # Head Visual Odometry Node (Python Fallback)
    head_odometry = Node(
        package='exo_head_slam',
        executable='fallback_vo',
        name='head_rgbd_odometry',
        parameters=[{**base_params, **head_odom_params}],
        remappings=[
            ('rgb/image', head_params['rgb_topic']),
            ('depth/image', head_params['depth_topic']),
            ('rgb/camera_info', head_params['camera_info_topic']),
            ('odom', '/head/odom'),
        ],
        output='screen'
    )

    # SLAM Node (Head)
    head_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='head_rtabmap',
        parameters=[{**base_params, **head_params}],
        remappings=[
            ('rgb/image', head_params['rgb_topic']),
            ('depth/image', head_params['depth_topic']),
            ('rgb/camera_info', head_params['camera_info_topic']),
            ('odom', '/head/odom'),
        ],
        arguments=['-d'],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        exo_odometry,
        exo_rtabmap,
        head_odometry,
        head_rtabmap,
    ])

