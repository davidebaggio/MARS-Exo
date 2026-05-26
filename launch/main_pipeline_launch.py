from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare('exo_head_slam')
    head_config = PathJoinSubstitution([pkg_share, 'config', 'head.yaml'])
    exo_config = PathJoinSubstitution([pkg_share, 'config', 'exo.yaml'])
    common_config = PathJoinSubstitution([pkg_share, 'config', 'common.yaml'])

    head_depth_preprocessor = Node(
        package='exo_head_slam',
        executable='depth_preprocessor',
        name='head_depth_preprocessor',
        parameters=[head_config],
    )

    exo_depth_preprocessor = Node(
        package='exo_head_slam',
        executable='depth_preprocessor',
        name='exo_depth_preprocessor',
        parameters=[exo_config],
    )
    
    # Head Masker
    head_masker = Node(
        package='exo_head_slam',
        executable='semantic_masker',
        name='head_semantic_masker',
        parameters=[head_config],
    )
    
    # Exo Masker
    exo_masker = Node(
        package='exo_head_slam',
        executable='semantic_masker',
        name='exo_semantic_masker',
        parameters=[exo_config],
    )
    
    # Extrinsic Solver
    extrinsic_solver = Node(
        package='exo_head_slam',
        executable='extrinsic_solver',
        name='extrinsic_solver',
        parameters=[common_config],
    )

    actions = [
        head_depth_preprocessor,
        exo_depth_preprocessor,
        head_masker,
        exo_masker,
        extrinsic_solver,
    ]

    try:
        get_package_share_directory('rtabmap_slam')
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_share, 'launch', 'rtabmap_agents_launch.py'])
                )
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
                )
            )
        )
    except PackageNotFoundError:
        actions.append(LogInfo(msg='nvblox_ros not found, skipping NVBlox launch.'))

    return LaunchDescription(actions)
