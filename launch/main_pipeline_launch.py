from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
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

    metrics_csv_path_arg = DeclareLaunchArgument(
        'metrics_csv_path',
        default_value='extrinsic_metrics.csv',
        description='Path to the metrics CSV file'
    )
    metrics_csv_path = LaunchConfiguration('metrics_csv_path')

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
        parameters=[common_config, exo_config, common_params, {'metrics_csv_path': metrics_csv_path}],
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
        use_imu_arg,
        imu_topic_arg,
        filtered_imu_topic_arg,
        map_start_z_arg,
    ]

    try:
        get_package_share_directory('rtabmap_slam')
        get_package_share_directory('rtabmap_odom')
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
                }.items(),
            )
        )
    except PackageNotFoundError:
        actions.append(LogInfo(msg='rtabmap_slam or rtabmap_odom not found, skipping RTAB-Map launch.'))

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
