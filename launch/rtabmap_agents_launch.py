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
    Launches a single RTAB-Map instance on the exo camera for VO/SLAM.
    Dense mapping is disabled since NVBlox handles 3D reconstruction.
    """
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_dir = os.path.join(get_package_share_directory('exo_head_slam'), 'config')
    config_path = os.path.join(config_dir, 'exo.yaml')
    exo_params = load_section(config_path, 'exo_rtabmap')

    # Base parameters for both odom and slam nodes
    base_params = {
        'use_sim_time': use_sim_time,
        'qos_image': 2,
        'qos_depth': 2,
        'qos_camera_info': 2,
    }
    
    # Merge YAML params into runtime_params
    runtime_params = {**base_params, **exo_params}

    # Visual Odometry Node
    exo_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='exo_rgbd_odometry',
        parameters=[runtime_params],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
        ],
        arguments=['-d'],
        output='screen'
    )

    # SLAM Node
    exo_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='exo_rtabmap',
        parameters=[runtime_params],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
            # ('odom', 'odom'), # Explicitly NOT remapping to favor TF if odom_frame_id is set
        ],
        arguments=['-d'],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        exo_odometry,
        exo_rtabmap,
    ])
