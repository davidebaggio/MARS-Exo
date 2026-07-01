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
        default_value='false',
        description='Publish per-image PointCloud2 in map frame for debugging'
    )
    publish_debug_pcl = LaunchConfiguration('publish_debug_pcl')

    global_frame_arg = DeclareLaunchArgument(
        'global_frame',
        default_value='map',
        description='Global frame for point clouds and rviz (e.g., map or odom)'
    )
    global_frame = LaunchConfiguration('global_frame')

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

    # IMU Integrator
    imu_integrator = Node(
        package='exo_head_slam',
        executable='imu_integrator',
        name='imu_integrator',
        parameters=[common_config, exo_config, common_params],
    )

    # Debug PointCloud Publishers
    head_pcl_pub = Node(
        package='exo_head_slam',
        executable='pointcloud_publisher',
        name='head_pcl_publisher',
        parameters=[{
            'input_rgb_topic': '/head/masked/image_raw',
            'input_depth_topic': '/head/masked/depth_raw',
            'input_camera_info_topic': '/camera/head/color/camera_info',
            'output_pcl_topic': '/head/debug_pcl',
            'global_frame': global_frame,
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
            'input_depth_topic': '/exo/masked/depth_raw',
            'input_camera_info_topic': '/camera/exo/color/camera_info',
            'output_pcl_topic': '/exo/debug_pcl',
            'global_frame': global_frame,
            'downsample_factor': 2,
            'use_sim_time': use_sim_time
        }],
        condition=IfCondition(publish_debug_pcl)
    )

    # Static TFs linking exo_link/head_link to camera optical frames
    static_tf_exo_color = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_exo_color',
        arguments=['--x', '-0.000374', '--y', '0.014791', '--z', '0.000141',
                   '--qx', '0.005959', '--qy', '0.002870', '--qz', '-0.001444', '--qw', '0.999977',
                   '--frame-id', 'exo_link', '--child-frame-id', 'exo_color_frame']
    )

    static_tf_exo_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_exo_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                   '--frame-id', 'exo_color_frame', '--child-frame-id', 'exo_color_optical_frame']
    )

    static_tf_head_color = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_head_color',
        arguments=['--x', '-0.000417', '--y', '0.014662', '--z', '-0.000104',
                   '--qx', '-0.004870', '--qy', '-0.000842', '--qz', '0.000748', '--qw', '0.999987',
                   '--frame-id', 'head_link', '--child-frame-id', 'head_color_frame']
    )

    static_tf_head_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_head_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                   '--frame-id', 'head_color_frame', '--child-frame-id', 'head_color_optical_frame']
    )

    actions = [
        use_sim_time_arg,
        publish_debug_pcl_arg,
        global_frame_arg,
        static_tf_exo_color,
        static_tf_exo_optical,
        static_tf_head_color,
        static_tf_head_optical,
        head_depth_preprocessor,
        exo_depth_preprocessor,
        head_masker,
        exo_masker,
        imu_integrator,
        extrinsic_solver,
        head_pcl_pub,
        exo_pcl_pub,
    ]

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_share, 'launch', 'slam_launch.py'])
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_share, 'launch', 'nvblox_fusion_launch.py'])
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'global_frame': global_frame
            }.items(),
        )
    )

    return LaunchDescription(actions)
