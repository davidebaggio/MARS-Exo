from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
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

    metrics_csv_path_arg = DeclareLaunchArgument(
        'metrics_csv_path',
        default_value='extrinsic_metrics.csv',
        description='Path to the metrics CSV file'
    )
    metrics_csv_path = LaunchConfiguration('metrics_csv_path')

    gt_parent_frame_arg = DeclareLaunchArgument('gt_parent_frame', default_value='')
    gt_parent_frame = LaunchConfiguration('gt_parent_frame')
    gt_child_frame_arg = DeclareLaunchArgument('gt_child_frame', default_value='')
    gt_child_frame = LaunchConfiguration('gt_child_frame')
    gt_tf_topic_arg = DeclareLaunchArgument('gt_tf_topic', default_value='')
    gt_tf_topic = LaunchConfiguration('gt_tf_topic')

    exo_pitch_deg_arg = DeclareLaunchArgument('exo_pitch_deg', default_value='0')
    exo_pitch_deg = LaunchConfiguration('exo_pitch_deg')
    head_pitch_deg_arg = DeclareLaunchArgument('head_pitch_deg', default_value='0')
    head_pitch_deg = LaunchConfiguration('head_pitch_deg')

    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='false',
        description='Use exo IMU for RTAB-Map odometry gravity initialization'
    )
    use_imu = LaunchConfiguration('use_imu')

    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/camera/exo/imu',
        description='Raw exo IMU topic'
    )
    imu_topic = LaunchConfiguration('imu_topic')

    filtered_imu_topic_arg = DeclareLaunchArgument(
        'filtered_imu_topic',
        default_value='/exo/imu/data',
        description='Madgwick-filtered exo IMU topic for RTAB-Map'
    )
    filtered_imu_topic = LaunchConfiguration('filtered_imu_topic')

    map_start_z_arg = DeclareLaunchArgument(
        'map_start_z',
        default_value='1',
        description='Initial map height in the RViz ground frame, meters'
    )
    map_start_z = LaunchConfiguration('map_start_z')

    exoskeleton_dataset_arg = DeclareLaunchArgument(
        'exoskeleton_dataset',
        default_value='false',
        description='Use frame adapters for the exoskeleton_dataset bag'
    )
    exoskeleton_dataset = LaunchConfiguration('exoskeleton_dataset')

    coupled_sequence_dataset_arg = DeclareLaunchArgument(
        'coupled_sequence_dataset',
        default_value='false',
        description='Pair adjacent frames from a single-camera RGB-D sequence'
    )
    coupled_sequence_dataset = LaunchConfiguration('coupled_sequence_dataset')

    tum_ground_truth_arg = DeclareLaunchArgument(
        'tum_ground_truth',
        default_value='false',
        description='Evaluate against /ground_truth/odom from a TUM bag'
    )
    tum_ground_truth = LaunchConfiguration('tum_ground_truth')

    benchmark_output_prefix_arg = DeclareLaunchArgument(
        'benchmark_output_prefix',
        default_value='metrics/eval/benchmark',
        description='Output prefix for trajectory and map benchmark files',
    )
    benchmark_output_prefix = LaunchConfiguration('benchmark_output_prefix')

    sliding_window_size_arg = DeclareLaunchArgument(
        'sliding_window_size', default_value='4',
        description='VGGT extrinsic-solver temporal window size',
    )
    sliding_window_size = LaunchConfiguration('sliding_window_size')

    common_params = {'use_sim_time': use_sim_time}

    sequence_pair_adapter = Node(
        package='exo_head_slam',
        executable='sequence_pair_adapter',
        name='sequence_pair_adapter',
        parameters=[common_params],
        condition=IfCondition(coupled_sequence_dataset),
    )

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
    
    semantic_masker = Node(
        package='exo_head_slam',
        executable='semantic_masker',
        name='semantic_masker',
        parameters=[common_config, common_params],
    )
    
    # Extrinsic Solver
    extrinsic_solver = Node(
        package='exo_head_slam',
        executable='extrinsic_solver',
        name='extrinsic_solver',
        parameters=[common_config, exo_config, common_params, {
            'metrics_csv_path': metrics_csv_path,
            'sliding_window_size': ParameterValue(sliding_window_size, value_type=int),
            'gt_parent_frame': gt_parent_frame,
            'gt_child_frame': gt_child_frame,
            'gt_tf_topic': gt_tf_topic,
            'validate_rig_geometry': ParameterValue(
                PythonExpression([
                    "'", coupled_sequence_dataset, "'.lower() != 'true'"
                ]),
                value_type=bool,
            ),
            'min_solver_interval': ParameterValue(
                PythonExpression([
                    "0.0 if '", coupled_sequence_dataset,
                    "'.lower() == 'true' else 0.2"
                ]),
                value_type=float,
            ),
        }],
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

    # Start RTAB-Map early so it initializes before data flows (service ready
    # by the time map_assembler fires after its 5s TimerAction delay).
    actions = [
        use_sim_time_arg,
        publish_debug_pcl_arg,
        metrics_csv_path_arg,
        gt_parent_frame_arg,
        gt_child_frame_arg,
        gt_tf_topic_arg,
        exo_pitch_deg_arg,
        head_pitch_deg_arg,
        use_imu_arg,
        imu_topic_arg,
        filtered_imu_topic_arg,
        map_start_z_arg,
        exoskeleton_dataset_arg,
        coupled_sequence_dataset_arg,
        tum_ground_truth_arg,
        benchmark_output_prefix_arg,
        sliding_window_size_arg,
    ]

    # The dataset uses camera-link names while the pipeline owns exo_link/head_link.
    for name, parent, child, translation, quaternion in (
        ('exo_color_tf', 'exo_link', 'front_camera_color_optical_frame', ('0', '0', '0'), ('0.5', '-0.5', '0.5', '-0.5')),
        ('exo_imu_tf', 'exo_link', 'front_camera_imu_frame', ('0', '0', '0'), ('0', '0', '0', '1')),
        ('head_color_tf', 'head_link', 'head_camera_color_optical_frame', ('0', '0', '0'), ('0.5', '-0.5', '0.5', '-0.5')),
    ):
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=name,
            arguments=[
                '--x', translation[0], '--y', translation[1], '--z', translation[2],
                '--qx', quaternion[0], '--qy', quaternion[1],
                '--qz', quaternion[2], '--qw', quaternion[3],
                '--frame-id', parent, '--child-frame-id', child,
            ],
            condition=IfCondition(
                exoskeleton_dataset if name == 'exo_imu_tf' else PythonExpression([
                    "'", exoskeleton_dataset, "'.lower() == 'true' or '",
                    coupled_sequence_dataset, "'.lower() == 'true'",
                ])
            ),
        ))

    for name, child, translation, pitch_deg in (
        ('gt_exo_tf', 'gt_exo_link', ('0.07', '0', '0'), exo_pitch_deg),
        ('gt_head_tf', 'gt_head_link', ('0.07', '0', '0.75'), head_pitch_deg),
    ):
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=name,
            arguments=[
                '--x', translation[0], '--y', translation[1], '--z', translation[2],
                '--roll', '0',
                '--pitch', PythonExpression([pitch_deg, ' * 0.017453292519943295']),
                '--yaw', '0',
                '--frame-id', 'gt_waist_link', '--child-frame-id', child,
            ],
            condition=IfCondition(exoskeleton_dataset),
        ))

    try:
        for required_package in ('rtabmap_slam', 'rtabmap_odom', 'rtabmap_util'):
            get_package_share_directory(required_package)
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_share, 'launch', 'rtabmap_agents_launch.py'])
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'use_imu': use_imu,
                    'imu_topic': imu_topic,
                    'filtered_imu_topic': filtered_imu_topic,
                    'map_start_z': map_start_z,
                    'coupled_sequence_dataset': coupled_sequence_dataset,
                }.items(),
            )
        )
        actions.append(Node(
            package='exo_head_slam',
            executable='benchmark_evaluator',
            name='benchmark_evaluator',
            parameters=[{
                'use_sim_time': use_sim_time,
                'output_prefix': benchmark_output_prefix,
                'map_start_z': ParameterValue(map_start_z, value_type=float),
                'waist_to_exo_pitch_deg': ParameterValue(
                    exo_pitch_deg, value_type=float
                ),
                'ground_truth_odom_topic': PythonExpression([
                    "'/ground_truth/odom' if '", tum_ground_truth,
                    "'.lower() == 'true' else '/exoskeleton/odom'",
                ]),
                'ground_truth_is_camera_pose': ParameterValue(
                    tum_ground_truth, value_type=bool
                ),
                'require_ground_truth_map': ParameterValue(
                    PythonExpression([
                        "'", tum_ground_truth, "'.lower() != 'true'"
                    ]),
                    value_type=bool,
                ),
            }],
            condition=IfCondition(PythonExpression([
                "'", exoskeleton_dataset, "'.lower() == 'true' or '",
                tum_ground_truth, "'.lower() == 'true'",
            ])),
            output='screen',
        ))
    except PackageNotFoundError:
        actions.append(LogInfo(msg='Required RTAB-Map packages not found, skipping RTAB-Map launch.'))

    actions.extend([
        sequence_pair_adapter,
        head_depth_preprocessor,
        exo_depth_preprocessor,
        semantic_masker,
        extrinsic_solver,
        head_pcl_pub,
        exo_pcl_pub,
    ])

    return LaunchDescription(actions)
