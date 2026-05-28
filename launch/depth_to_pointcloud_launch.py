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
    config_path = os.path.join(config_dir, 'head.yaml')
    exo_config_path = os.path.join(config_dir, 'exo.yaml')

    head_masker_params = load_section(config_path, 'head_semantic_masker')
    exo_masker_params = load_section(exo_config_path, 'exo_semantic_masker')

    head_params = {
        'depth_topic': head_masker_params.get('output_depth_topic', '/head/masked/depth_raw'),
        'camera_info_topic': '/camera/head/color/camera_info',
        'pointcloud_topic': '/head/pointcloud',
        'pointcloud_frame_id': 'head_camera_link',
        'depth_unit_scale': 0.001,
        'max_depth': 5.0,
        'downsample_factor': 4,
        'use_sim_time': use_sim_time,
    }

    exo_params = {
        'depth_topic': exo_masker_params.get('output_depth_topic', '/exo/masked/depth_raw'),
        'camera_info_topic': '/camera/exo/color/camera_info',
        'pointcloud_topic': '/exo/pointcloud',
        'pointcloud_frame_id': 'exo_camera_link',
        'depth_unit_scale': 0.001,
        'max_depth': 5.0,
        'downsample_factor': 4,
        'use_sim_time': use_sim_time,
    }

    head_pc = Node(
        package='exo_head_slam',
        executable='depth_to_pointcloud',
        name='head_depth_to_pointcloud',
        parameters=[head_params],
        output='screen',
    )

    exo_pc = Node(
        package='exo_head_slam',
        executable='depth_to_pointcloud',
        name='exo_depth_to_pointcloud',
        parameters=[exo_params],
        output='screen',
    )

    return LaunchDescription([
        use_sim_time_arg,
        head_pc,
        exo_pc,
    ])
