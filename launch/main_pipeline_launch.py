from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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

    dataset_mode_arg = DeclareLaunchArgument(
        'dataset_mode',
        default_value='false',
        description='Enable exoskeleton_dataset frame isolation and evaluation'
    )
    dataset_mode = LaunchConfiguration('dataset_mode')

    depth_unit_scale_arg = DeclareLaunchArgument(
        'depth_unit_scale',
        default_value='0.001',
        description='Scale applied to incoming depth values'
    )
    depth_unit_scale = LaunchConfiguration('depth_unit_scale')

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

    metrics_csv_path_arg = DeclareLaunchArgument(
        'metrics_csv_path',
        default_value='extrinsic_metrics.csv',
        description='Path to the extrinsic metrics CSV file'
    )
    metrics_csv_path = LaunchConfiguration('metrics_csv_path')

    gt_parent_frame_arg = DeclareLaunchArgument('gt_parent_frame', default_value='')
    gt_parent_frame = LaunchConfiguration('gt_parent_frame')
    gt_child_frame_arg = DeclareLaunchArgument('gt_child_frame', default_value='')
    gt_child_frame = LaunchConfiguration('gt_child_frame')
    gt_tf_static_topic_arg = DeclareLaunchArgument('gt_tf_static_topic', default_value='')
    gt_tf_static_topic = LaunchConfiguration('gt_tf_static_topic')

    evaluation_enabled_arg = DeclareLaunchArgument(
        'evaluation_enabled',
        default_value='false',
        description='Run trajectory and map benchmark evaluator'
    )
    evaluation_enabled = LaunchConfiguration('evaluation_enabled')
    benchmark_output_prefix_arg = DeclareLaunchArgument(
        'benchmark_output_prefix',
        default_value='metrics/eval/benchmark',
        description='Benchmark output path without extension'
    )
    benchmark_output_prefix = LaunchConfiguration('benchmark_output_prefix')

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
        parameters=[
            head_config, common_params,
            {'depth_filter.depth_unit_scale': depth_unit_scale},
        ],
    )

    exo_depth_preprocessor = Node(
        package='exo_head_slam',
        executable='depth_preprocessor',
        name='exo_depth_preprocessor',
        parameters=[
            exo_config, common_params,
            {'depth_filter.depth_unit_scale': depth_unit_scale},
        ],
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
        parameters=[
            common_config,
            exo_config,
            common_params,
            {
                'metrics_csv_path': metrics_csv_path,
                'gt_parent_frame': gt_parent_frame,
                'gt_child_frame': gt_child_frame,
                'gt_tf_static_topic': gt_tf_static_topic,
            },
        ],
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

    dataset_static_transforms = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dataset_exo_optical_tf',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--qx', '0.5', '--qy', '-0.5', '--qz', '0.5', '--qw', '-0.5',
                '--frame-id', 'exo_link',
                '--child-frame-id', 'front_camera_color_optical_frame',
            ],
            condition=IfCondition(dataset_mode),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dataset_exo_imu_tf',
            arguments=[
                '--frame-id', 'exo_link',
                '--child-frame-id', 'front_camera_imu_frame',
            ],
            condition=IfCondition(dataset_mode),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dataset_head_optical_tf',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--qx', '0.5', '--qy', '-0.5', '--qz', '0.5', '--qw', '-0.5',
                '--frame-id', 'head_link',
                '--child-frame-id', 'head_camera_color_optical_frame',
            ],
            condition=IfCondition(dataset_mode),
        ),
    ]

    benchmark_evaluator = Node(
        package='exo_head_slam',
        executable='benchmark_evaluator',
        name='benchmark_evaluator',
        parameters=[
            common_params,
            {
                'output_prefix': benchmark_output_prefix,
                'map_start_z': ParameterValue(map_start_z, value_type=float),
                'estimated_odom_topic': '/exo/odom',
                'estimated_map_topic': '/orbslam/cloud_map',
            },
        ],
        condition=IfCondition(evaluation_enabled),
        output='screen',
    )
    actions = [
        use_sim_time_arg,
        dataset_mode_arg,
        depth_unit_scale_arg,
        publish_debug_pcl_arg,
        orbslam3_vocabulary_path_arg,
        orbslam3_settings_path_arg,
        metrics_csv_path_arg,
        gt_parent_frame_arg,
        gt_child_frame_arg,
        gt_tf_static_topic_arg,
        evaluation_enabled_arg,
        benchmark_output_prefix_arg,
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
        *dataset_static_transforms,
        benchmark_evaluator,
    ])

    return LaunchDescription(actions)
