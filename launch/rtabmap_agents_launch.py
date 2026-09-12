import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition


def load_section(config_path: str, section: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    return (data.get(section, {}) or {}).get('ros__parameters', {})


def generate_launch_description():
    """
    Launches RGB-D odometry plus RTAB-Map on the exo camera.
    TF ownership: dynamic map->odom from RTAB-Map, dynamic odom->exo_link from odometry.
    """
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='false',
        description='Use IMU orientation to initialize RTAB-Map odometry'
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
        description='Madgwick-filtered IMU topic'
    )
    filtered_imu_topic = LaunchConfiguration('filtered_imu_topic')

    map_start_z_arg = DeclareLaunchArgument(
        'map_start_z',
        default_value='1',
        description='Initial map height in the RViz ground frame, meters'
    )
    map_start_z = LaunchConfiguration('map_start_z')

    coupled_sequence_dataset_arg = DeclareLaunchArgument(
        'coupled_sequence_dataset', default_value='false'
    )
    coupled_sequence_dataset = LaunchConfiguration('coupled_sequence_dataset')

    pkg_share = get_package_share_directory('exo_head_slam')
    config_dir = os.path.join(pkg_share, 'config')

    exo_config_path = os.path.join(config_dir, 'exo.yaml')
    common_config_path = os.path.join(config_dir, 'common.yaml')

    exo_params = load_section(exo_config_path, 'exo_rtabmap')

    # Extract topic names for remapping, don't pass them as ROS params
    # (they'd conflict with the remapping mechanism in rtabmap_slam)
    rgb_topic = exo_params.pop('rgb_topic')
    depth_topic = exo_params.pop('depth_topic')
    camera_info_topic = exo_params.pop('camera_info_topic')
    exo_params['approx_sync'] = ParameterValue(
        PythonExpression([
            "'", coupled_sequence_dataset, "'.lower() != 'true'"
        ]),
        value_type=bool,
    )

    # Base parameters for all nodes
    base_params = {
        'use_sim_time': use_sim_time,
        'qos_image': 2,
        'qos_depth': 2,
        'qos_camera_info': 2,
        'qos_imu': 2,
    }
    odom_topic = '/exo_rtabmap/odom'

    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='exo_imu_filter',
        parameters=[{
            'use_sim_time': use_sim_time,
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
        }],
        remappings=[
            ('imu/data_raw', imu_topic),
            ('imu/data', filtered_imu_topic),
        ],
        condition=IfCondition(use_imu),
        output='screen',
    )

    viz_ground_to_map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='viz_ground_to_map_tf',
        arguments=['--x', '0', '--y', '0', '--z', map_start_z, '--frame-id', 'viz_ground', '--child-frame-id', 'map'],
        output='screen'
    )

    exo_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='exo_rgbd_odometry',
        parameters=[{**base_params, **exo_params, 'publish_tf': True, 'wait_imu_to_init': use_imu, 'always_check_imu_tf': True}],
        remappings=[
            ('rgb/image', rgb_topic),
            ('depth/image', depth_topic),
            ('rgb/camera_info', camera_info_topic),
            ('odom', odom_topic),
            ('imu', filtered_imu_topic),
        ],
        output='screen'
    )

    # SLAM Node (Exo): consumes odometry and owns the corrected map->odom TF.
    exo_rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='exo_rtabmap',
        parameters=[{**base_params, **exo_params, 'publish_tf': True, 'subscribe_odom_info': True}],
        arguments=['-d'],
        remappings=[
            ('rgb/image', rgb_topic),
            ('depth/image', depth_topic),
            ('rgb/camera_info', camera_info_topic),
            ('depth/camera_info', camera_info_topic),
            ('odom', odom_topic),
            ('grid_map', '/map'),
        ],
        output='screen'
    )

    # Map Assembler: subscribes to core SLAM's map_graph and publishes /exo_rtabmap/cloud_map.
    # Delayed 5s so rtabmap/get_map_data service is available at startup (avoids WARN
    # and ensures full cloud map appears immediately rather than growing incrementally).
    map_assembler = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rtabmap_util',
                executable='map_assembler',
                name='map_assembler',
                parameters=[common_config_path, {'use_sim_time': use_sim_time}],
                remappings=[
                    ('map_graph', '/exo_rtabmap/map_graph'),
                    ('cloud_map', '/exo_rtabmap/cloud_map'),
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_imu_arg,
        imu_topic_arg,
        filtered_imu_topic_arg,
        map_start_z_arg,
        coupled_sequence_dataset_arg,
        viz_ground_to_map_tf,
        imu_filter,
        exo_odometry,
        exo_rtabmap,
        map_assembler,
    ])
