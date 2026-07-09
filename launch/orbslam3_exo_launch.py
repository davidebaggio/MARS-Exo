import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    orbslam_mode = LaunchConfiguration('orbslam_mode')

    pkg_share = get_package_share_directory('exo_head_slam')
    params_path = os.path.join(pkg_share, 'config', 'orbslam3_exo.yaml')
    settings_path = os.path.join(pkg_share, 'config', 'orbslam3_exo_settings.yaml')
    workspace_root = os.path.abspath(os.path.join(pkg_share, '..', '..', '..', '..'))
    default_orbslam_root = os.path.join(workspace_root, 'third_party', 'ORB_SLAM3')
    vocabulary_path = os.path.join(
        os.environ.get('ORB_SLAM3_ROOT', default_orbslam_root),
        'Vocabulary',
        'ORBvoc.txt',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('orbslam_mode', default_value='rgbd_imu'),
        Node(
            package='orbslam3_ros2',
            executable='orbslam3_rgbd_imu',
            name='orbslam3_exo',
            parameters=[
                params_path,
                {
                    'use_sim_time': use_sim_time,
                    'mode': orbslam_mode,
                    'settings_path': settings_path,
                    'vocabulary_path': vocabulary_path,
                },
            ],
            output='screen',
        ),
    ])
