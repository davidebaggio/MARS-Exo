from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('exo_head_slam')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    orbslam3_vocabulary_path_arg = DeclareLaunchArgument(
        'orbslam3_vocabulary_path',
        default_value='third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        description='Path to the ORB-SLAM3 vocabulary file'
    )
    orbslam3_vocabulary_path = LaunchConfiguration('orbslam3_vocabulary_path')

    orbslam3_settings_path_arg = DeclareLaunchArgument(
        'orbslam3_settings_path',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'orbslam3_exo.yaml']),
        description='Path to the ORB-SLAM3 settings file'
    )
    orbslam3_settings_path = LaunchConfiguration('orbslam3_settings_path')

    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/camera/exo/imu',
        description='Raw exo IMU topic'
    )
    imu_topic = LaunchConfiguration('imu_topic')

    slam_mode_arg = DeclareLaunchArgument(
        'slam_mode',
        default_value='IMU_RGBD',
        description='ORB-SLAM3 mode: IMU_RGBD (with IMU) or RGBD (visual only)'
    )
    slam_mode = LaunchConfiguration('slam_mode')

    map_start_z_arg = DeclareLaunchArgument(
        'map_start_z',
        default_value='1',
        description='Initial map height in the RViz ground frame, meters'
    )
    map_start_z = LaunchConfiguration('map_start_z')

    dense_map_voxel_size_arg = DeclareLaunchArgument(
        'dense_map_voxel_size',
        default_value='0.03',
        description='Voxel size in meters for the accumulated global cloud'
    )
    dense_map_voxel_size = LaunchConfiguration('dense_map_voxel_size')

    dense_map_max_points_arg = DeclareLaunchArgument(
        'dense_map_max_points',
        default_value='250000',
        description='Maximum accumulated points in the global cloud'
    )
    dense_map_max_points = LaunchConfiguration('dense_map_max_points')

    dense_map_downsample_factor_arg = DeclareLaunchArgument(
        'dense_map_downsample_factor',
        default_value='2',
        description='Per-frame downsample factor before projection'
    )
    dense_map_downsample_factor = LaunchConfiguration('dense_map_downsample_factor')

    dense_map_min_depth_arg = DeclareLaunchArgument(
        'dense_map_min_depth',
        default_value='0.1',
        description='Minimum depth accepted by the dense accumulator'
    )
    dense_map_min_depth = LaunchConfiguration('dense_map_min_depth')

    dense_map_max_depth_arg = DeclareLaunchArgument(
        'dense_map_max_depth',
        default_value='10.0',
        description='Maximum depth accepted by the dense accumulator'
    )
    dense_map_max_depth = LaunchConfiguration('dense_map_max_depth')

    viz_ground_to_map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='viz_ground_to_map_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', map_start_z,
            '--frame-id', 'viz_ground',
            '--child-frame-id', 'map',
        ],
        output='screen'
    )

    orbslam3_exo = Node(
        package='orbslam3_ros2',
        executable='orbslam3_rgbd_imu',
        name='orbslam3_exo',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'vocabulary_path': orbslam3_vocabulary_path,
            'settings_path': orbslam3_settings_path,
            'rgb_topic': '/camera/exo/color/image_raw',
            'depth_topic': '/exo/filtered/depth_raw',
            'imu_topic': imu_topic,
            'camera_info_topic': '/camera/exo/color/camera_info',
            'map_frame_id': 'map',
            'base_frame_id': 'exo_link',
            'camera_frame_id': '',
            'publish_tf': True,
            'depth_scale': 1.0,
            'imu_timeout_sec': 15.0,
            'imu_max_lag_sec': 0.02,
            'sync_max_delta_sec': 0.02,
            'pose_jump_translation_m': 2.0,
            'pose_jump_rotation_rad': 1.0,
            'slam_mode': slam_mode,
        }]
    )

    dense_global_map = Node(
        package='exo_head_slam',
        executable='dense_global_map',
        name='dense_global_map_accumulator',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'rgb_topic': '/exo/masked/image_raw',
            'depth_topic': '/exo/masked/depth_raw',
            'camera_info_topic': '/camera/exo/color/camera_info',
            'map_frame_id': 'map',
            'voxel_size': dense_map_voxel_size,
            'max_points': dense_map_max_points,
            'downsample_factor': dense_map_downsample_factor,
            'min_depth': dense_map_min_depth,
            'max_depth': dense_map_max_depth,
            'lookup_timeout_sec': 0.1,
            'warn_interval_sec': 5.0,
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        orbslam3_vocabulary_path_arg,
        orbslam3_settings_path_arg,
        imu_topic_arg,
        map_start_z_arg,
        slam_mode_arg,
        dense_map_voxel_size_arg,
        dense_map_max_points_arg,
        dense_map_downsample_factor_arg,
        dense_map_min_depth_arg,
        dense_map_max_depth_arg,
        viz_ground_to_map_tf,
        orbslam3_exo,
        dense_global_map,
    ])
