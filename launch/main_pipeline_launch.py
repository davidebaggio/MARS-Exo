from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    pkg_share = FindPackageShare('exo_head_slam')
    head_config = PathJoinSubstitution([pkg_share, 'config', 'head.yaml'])
    exo_config = PathJoinSubstitution([pkg_share, 'config', 'exo.yaml'])
    common_config = PathJoinSubstitution([pkg_share, 'config', 'common.yaml'])

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (bag) time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    publish_debug_pcl_arg = DeclareLaunchArgument(
        'publish_debug_pcl',
        default_value='true',
        description='Publish per-image PointCloud2 in map frame for debugging'
    )
    publish_debug_pcl = LaunchConfiguration('publish_debug_pcl')

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

    common_params = {'use_sim_time': use_sim_time}

    head_depth_preprocessor = Node(
        package='exo_head_slam',
        executable='depth_preprocessor',
        name='head_depth_preprocessor',
        parameters=[head_config, common_params],
    )

    exo_depth_preprocessor = Node(
        package='exo_head_slam',
        executable='depth_preprocessor',
        name='exo_depth_preprocessor',
        parameters=[exo_config, common_params],
    )
    
    # Head Masker
    head_masker = Node(
        package='exo_head_slam',
        executable='semantic_masker',
        name='head_semantic_masker',
        parameters=[head_config, common_params],
    )
    
    # Exo Masker
    exo_masker = Node(
        package='exo_head_slam',
        executable='semantic_masker',
        name='exo_semantic_masker',
        parameters=[exo_config, common_params],
    )
    
    # Extrinsic Solver
    extrinsic_solver = Node(
        package='exo_head_slam',
        executable='extrinsic_solver',
        name='extrinsic_solver',
        parameters=[common_config, exo_config, common_params],
    )

    # Debug PointCloud Publishers
    head_pcl_pub = Node(
        package='exo_head_slam',
        executable='pointcloud_publisher',
        name='head_pcl_publisher',
        parameters=[{
            'input_rgb_topic': '/head/masked/image_raw',
            'input_depth_topic': '/head/combined/depth_raw',
            'input_camera_info_topic': '/camera/head/color/camera_info',
            'output_pcl_topic': '/head/debug_pcl',
            'downsample_factor': 2,
            'use_sim_time': use_sim_time
        }],
        condition=IfCondition(publish_debug_pcl)
    )

    exo_pcl_pub = Node(
        package='exo_head_slam',
        executable='pointcloud_publisher',
        name='exo_pcl_publisher',
        parameters=[{
            'input_rgb_topic': '/exo/masked/image_raw',
            'input_depth_topic': '/exo/combined/depth_raw',
            'input_camera_info_topic': '/camera/exo/color/camera_info',
            'output_pcl_topic': '/exo/debug_pcl',
            'downsample_factor': 2,
            'use_sim_time': use_sim_time
        }],
        condition=IfCondition(publish_debug_pcl)
    )

    actions = [
        use_sim_time_arg,
        publish_debug_pcl_arg,
        orbslam3_vocabulary_path_arg,
        orbslam3_settings_path_arg,
        imu_topic_arg,
        slam_mode_arg,
        map_start_z_arg,
        dense_map_voxel_size_arg,
        dense_map_max_points_arg,
        dense_map_downsample_factor_arg,
        dense_map_min_depth_arg,
        dense_map_max_depth_arg,
    ]

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_share, 'launch', 'orbslam3_exo_launch.py'])
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'orbslam3_vocabulary_path': orbslam3_vocabulary_path,
                'orbslam3_settings_path': orbslam3_settings_path,
                'imu_topic': imu_topic,
                'slam_mode': slam_mode,
                'map_start_z': map_start_z,
                'dense_map_voxel_size': dense_map_voxel_size,
                'dense_map_max_points': dense_map_max_points,
                'dense_map_downsample_factor': dense_map_downsample_factor,
                'dense_map_min_depth': dense_map_min_depth,
                'dense_map_max_depth': dense_map_max_depth,
            }.items(),
        )
    )

    actions.extend([
        head_depth_preprocessor,
        exo_depth_preprocessor,
        head_masker,
        exo_masker,
        extrinsic_solver,
        head_pcl_pub,
        exo_pcl_pub,
    ])

    return LaunchDescription(actions)
