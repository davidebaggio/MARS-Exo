import os
import re

import rosbag2_py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_evaluation(context):
    bag_path = LaunchConfiguration('bag_path').perform(context)
    metrics_csv_path = LaunchConfiguration('metrics_csv_path')
    playback_rate = LaunchConfiguration('playback_rate')
    global_mode = LaunchConfiguration('global')
    config = PathJoinSubstitution([FindPackageShare('exo_head_slam'), 'config', 'common.yaml'])
    metadata = rosbag2_py.Info().read_metadata(bag_path, 'mcap')
    exo_pitch_deg = metadata.custom_data.get('exo_pitch_deg')
    if exo_pitch_deg is None:
        match = re.fullmatch(
            r'metrics_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_'
            r'(-?\d+(?:\.\d+)?)_(\d+)_cloud_bag',
            os.path.basename(os.path.normpath(bag_path)),
        )
        if match:
            exo_pitch_deg = match.group(2)
    if exo_pitch_deg is None:
        exo_pitch_deg = LaunchConfiguration('exo_pitch_deg').perform(context)
    if not exo_pitch_deg:
        raise RuntimeError(
            'Evaluation bag has no exo_pitch_deg metadata; pass exo_pitch_deg for old bags.'
        )

    ground_truth_odom_topic = metadata.custom_data.get(
        'ground_truth_odom_topic', '/exoskeleton/odom'
    )
    ground_truth_cloud_is_local = metadata.custom_data.get(
        'ground_truth_cloud_is_local', 'false'
    ).lower() == 'true'
    ground_truth_is_camera_pose = metadata.custom_data.get(
        'ground_truth_is_camera_pose', 'false'
    ).lower() == 'true'

    evaluator = Node(
        package='exo_head_slam',
        executable='cloud_map_evaluator',
        name='cloud_map_evaluator',
        sigterm_timeout='300',
        sigkill_timeout='30',
        parameters=[config, {
            'use_sim_time': True,
            'metrics_csv_path': metrics_csv_path,
            'waist_to_exo_pitch_deg': float(exo_pitch_deg),
            'ground_truth_odom_topic': ground_truth_odom_topic,
            'ground_truth_cloud_is_local': ground_truth_cloud_is_local,
            'ground_truth_is_camera_pose': ground_truth_is_camera_pose,
            'global': global_mode,
        }],
    )
    player = ExecuteProcess(cmd=[
        'ros2', 'bag', 'play', '-i', bag_path, 'mcap',
        '--clock', '--disable-keyboard-controls', '--rate', playback_rate,
    ])

    return [
        evaluator,
        TimerAction(period=1.0, actions=[player]),
        RegisterEventHandler(OnProcessExit(
            target_action=player,
            on_exit=[TimerAction(
                period=5.0,
                actions=[EmitEvent(event=Shutdown(reason='Offline cloud evaluation complete'))],
            )],
        )),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag_path'),
        DeclareLaunchArgument('metrics_csv_path', default_value='cloud_metrics.csv'),
        DeclareLaunchArgument('exo_pitch_deg', default_value=''),
        DeclareLaunchArgument('playback_rate', default_value='3.0'),
        DeclareLaunchArgument('global', default_value='true'),
        OpaqueFunction(function=_launch_evaluation),
    ])
