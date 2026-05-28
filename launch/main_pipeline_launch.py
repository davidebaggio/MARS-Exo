from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument

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
        parameters=[common_config, common_params],
    )

    # Static transforms to link optical frames to camera_link frames (identity placeholders)
    head_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_head_optical',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'head_camera_link',
            '--child-frame-id', 'head_color_optical_frame',
        ],
        parameters=[common_params],
    )

    exo_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_exo_optical',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'exo_camera_link',
            '--child-frame-id', 'exo_color_optical_frame',
        ],
        parameters=[common_params],
    )

    actions = [
        use_sim_time_arg,
        head_depth_preprocessor,
        exo_depth_preprocessor,
        head_masker,
        exo_masker,
        extrinsic_solver,
        head_static_tf,
        exo_static_tf,
    ]

    try:
        get_package_share_directory('rtabmap_slam')
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_share, 'launch', 'rtabmap_agents_launch.py'])
                ),
                launch_arguments={'use_sim_time': use_sim_time}.items(),
            )
        )
    except PackageNotFoundError:
        actions.append(LogInfo(msg='rtabmap_slam not found, skipping RTAB-Map launch.'))

    try:
        get_package_share_directory('nvblox_ros')
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_share, 'launch', 'nvblox_fusion_launch.py'])
                ),
                launch_arguments={'use_sim_time': use_sim_time}.items(),
            )
        )
    except PackageNotFoundError:
        actions.append(LogInfo(msg='nvblox_ros not found, skipping NVBlox launch.'))

    return LaunchDescription(actions)
