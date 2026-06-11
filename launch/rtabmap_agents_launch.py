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
    
    # Exo params
    exo_config_path = os.path.join(config_dir, 'exo.yaml')
    exo_params = load_section(exo_config_path, 'exo_rtabmap')
    
    # Head params
    head_config_path = os.path.join(config_dir, 'head.yaml')
    head_params = load_section(head_config_path, 'head_rtabmap')

    base_params = {
        'use_sim_time': use_sim_time,
        'qos_image': 2,
        'qos_depth': 2,
        'qos_camera_info': 2,
    }

    # Exo Nodes
    exo_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='exo_rgbd_odometry',
        parameters=[{**base_params, **exo_params}],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
        ],
        arguments=['-d'],
        output='screen'
    )

    exo_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='exo_rtabmap',
        parameters=[{**base_params, **exo_params}],
        remappings=[
            ('rgb/image', exo_params['rgb_topic']),
            ('depth/image', exo_params['depth_topic']),
            ('rgb/camera_info', exo_params['camera_info_topic']),
        ],
        arguments=['-d'],
        output='screen'
    )

    # Head Nodes
    head_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='head_rgbd_odometry',
        parameters=[{**base_params, **head_params}],
        remappings=[
            ('rgb/image', head_params['rgb_topic']),
            ('depth/image', head_params['depth_topic']),
            ('rgb/camera_info', head_params['camera_info_topic']),
        ],
        arguments=['-d'],
        output='screen'
    )

    head_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='head_rtabmap',
        parameters=[{**base_params, **head_params}],
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
        exo_odometry,
        exo_rtabmap,
        head_odometry,
        head_rtabmap,
    ])
