#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of TUM nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

"""Convert a ROS 1 TUM RGB-D bag to ROS 2 and add PointCloud2 topics."""

import argparse
import os
from pathlib import Path

import numpy as np
from rosbags.convert.converter import create_connections_converters
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import CompressionFormat, CompressionMode, Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp


CAMERA_TOPICS = {
    '/camera/depth/camera_info',
    '/camera/depth/image',
    '/camera/rgb/camera_info',
    '/camera/rgb/image_color',
}

LINK_FROM_OPTICAL = np.eye(4)
LINK_FROM_OPTICAL[:3, :3] = Rotation.from_quat(
    [0.5, -0.5, 0.5, -0.5]
).as_matrix()
OPTICAL_FROM_LINK = np.linalg.inv(LINK_FROM_OPTICAL)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def image_array(msg):
    """Return an Image as an unpadded NumPy array."""
    formats = {
        'mono8': (np.uint8, 1),
        '8UC1': (np.uint8, 1),
        'rgb8': (np.uint8, 3),
        'bgr8': (np.uint8, 3),
        'rgba8': (np.uint8, 4),
        'bgra8': (np.uint8, 4),
        '16UC1': (np.uint16, 1),
        'mono16': (np.uint16, 1),
        '32FC1': (np.float32, 1),
    }
    try:
        base_dtype, channels = formats[msg.encoding]
    except KeyError as exc:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}') from exc

    dtype = np.dtype(base_dtype).newbyteorder('>' if msg.is_bigendian else '<')
    row_values = msg.step // dtype.itemsize
    image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
    image = image[:, :msg.width * channels]
    return image.reshape(msg.height, msg.width, channels) if channels > 1 else image


def rgb_and_mono(msg):
    image = image_array(msg)
    if image.ndim == 2:
        mono = image.astype(np.uint8)
        rgb = np.repeat(mono[..., None], 3, axis=2)
    else:
        rgb = image[..., :3]
        if msg.encoding.startswith('bgr'):
            rgb = rgb[..., ::-1]
        mono = np.rint(
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).astype(np.uint8)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(mono)


def depth_metres(msg):
    depth = image_array(msg).astype(np.float32)
    if msg.encoding in {'16UC1', 'mono16'}:
        depth *= 0.001
    return depth


def load_groundtruth(path):
    values = np.loadtxt(path, comments='#', dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f'Expected 8 ground-truth columns in {path}')
    return values


def interpolate_groundtruth(groundtruth, timestamp_ns, max_gap=0.05):
    timestamp = timestamp_ns / 1e9
    index = np.searchsorted(groundtruth[:, 0], timestamp)
    if index == 0 or index == len(groundtruth):
        nearest = 0 if index == 0 else len(groundtruth) - 1
        return (None if abs(groundtruth[nearest, 0] - timestamp) > max_gap
                else groundtruth[nearest].copy())
    before, after = groundtruth[index - 1], groundtruth[index]
    if min(timestamp - before[0], after[0] - timestamp) > max_gap:
        return None
    alpha = (timestamp - before[0]) / (after[0] - before[0])
    pose = np.empty(8, dtype=np.float64)
    pose[0] = timestamp
    pose[1:4] = before[1:4] + alpha * (after[1:4] - before[1:4])
    pose[4:8] = Slerp(
        [before[0], after[0]], Rotation.from_quat([before[4:8], after[4:8]])
    )([timestamp]).as_quat()[0]
    return pose


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[4:8]).as_matrix()
    matrix[:3, 3] = pose[1:4]
    return matrix


def paired_groundtruth(previous_points, previous_pose, current_points, current_pose):
    """Return measured pair cloud and head pose in current exo-link coordinates."""
    world_from_previous_optical = pose_matrix(previous_pose)
    world_from_current_optical = pose_matrix(current_pose)
    world_from_previous_link = world_from_previous_optical @ OPTICAL_FROM_LINK
    world_from_current_link = world_from_current_optical @ OPTICAL_FROM_LINK
    current_link_from_world = np.linalg.inv(world_from_current_link)
    current_link_from_previous_optical = (
        current_link_from_world @ world_from_previous_optical
    )
    previous_local = (
        previous_points @ current_link_from_previous_optical[:3, :3].T
        + current_link_from_previous_optical[:3, 3]
    )
    current_local = (
        current_points @ LINK_FROM_OPTICAL[:3, :3].T + LINK_FROM_OPTICAL[:3, 3]
    )
    current_link_from_previous_link = (
        current_link_from_world @ world_from_previous_link
    )
    return np.vstack([previous_local, current_local]), current_link_from_previous_link


def merge_world_voxels(voxels, camera_points, camera_pose, voxel_size=0.05):
    """Accumulate a bounded-density ground-truth map in TUM world coordinates."""
    world_from_camera = pose_matrix(camera_pose)
    points = (
        camera_points @ world_from_camera[:3, :3].T
        + world_from_camera[:3, 3]
    )
    keys = np.floor(points / voxel_size).astype(np.int64)
    for key, point in zip(map(tuple, keys), points):
        voxels.setdefault(key, point.astype(np.float32))


def make_odometry(store, stamp, pose):
    Header = store.types['std_msgs/msg/Header']
    Point = store.types['geometry_msgs/msg/Point']
    Quaternion = store.types['geometry_msgs/msg/Quaternion']
    Pose = store.types['geometry_msgs/msg/Pose']
    PoseWithCovariance = store.types['geometry_msgs/msg/PoseWithCovariance']
    Vector3 = store.types['geometry_msgs/msg/Vector3']
    Twist = store.types['geometry_msgs/msg/Twist']
    TwistWithCovariance = store.types['geometry_msgs/msg/TwistWithCovariance']
    Odometry = store.types['nav_msgs/msg/Odometry']
    zero_covariance = np.zeros(36, dtype=np.float64)
    return Odometry(
        Header(stamp, 'world'), 'front_camera_color_optical_frame',
        PoseWithCovariance(
            Pose(Point(*pose[1:4]), Quaternion(*pose[4:8])), zero_covariance
        ),
        TwistWithCovariance(
            Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)),
            zero_covariance.copy(),
        ),
    )


def make_transform_message(store, stamp, matrix):
    Header = store.types['std_msgs/msg/Header']
    Vector3 = store.types['geometry_msgs/msg/Vector3']
    Quaternion = store.types['geometry_msgs/msg/Quaternion']
    Transform = store.types['geometry_msgs/msg/Transform']
    TransformStamped = store.types['geometry_msgs/msg/TransformStamped']
    TFMessage = store.types['tf2_msgs/msg/TFMessage']
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    transform = TransformStamped(
        Header(stamp, 'gt_exo_link'), 'gt_head_link',
        Transform(Vector3(*matrix[:3, 3]), Quaternion(*quaternion)),
    )
    return TFMessage([transform])


def make_xyz_cloud(store, stamp, points, frame_id='exo_link'):
    Header = store.types['std_msgs/msg/Header']
    PointCloud2 = store.types['sensor_msgs/msg/PointCloud2']
    PointField = store.types['sensor_msgs/msg/PointField']
    points = np.asarray(points, dtype='<f4').reshape(-1, 3)
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    return PointCloud2(
        Header(stamp, frame_id), 1, len(points), fields, False, 12,
        12 * len(points), points.view(np.uint8).reshape(-1), True,
    )


def make_clouds(store, depth_msg, rgb_msg, camera_info):
    if depth_msg.height != rgb_msg.height or depth_msg.width != rgb_msg.width:
        raise ValueError('Depth and RGB image dimensions differ; images must be registered')

    depth = depth_metres(depth_msg)
    rgb, mono = rgb_and_mono(rgb_msg)
    height, width = depth.shape
    intrinsics = getattr(camera_info, 'k', None)
    if intrinsics is None:
        intrinsics = camera_info.K
    fx, fy, cx, cy = intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]
    if not fx or not fy:
        raise ValueError('Depth camera focal length is zero')

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)

    xyz = np.empty((height, width), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4')
    ])
    xyz['x'] = (u - cx) * z / fx
    xyz['y'] = (v - cy) * z / fy
    xyz['z'] = z

    xyzrgb = np.empty((height, width), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')
    ])
    for name in ('x', 'y', 'z'):
        xyzrgb[name] = xyz[name]
    xyzrgb['rgb'] = (
        rgb[..., 0].astype(np.uint32) << 16
        | rgb[..., 1].astype(np.uint32) << 8
        | rgb[..., 2].astype(np.uint32)
    )

    Image = store.types['sensor_msgs/msg/Image']
    PointCloud2 = store.types['sensor_msgs/msg/PointCloud2']
    PointField = store.types['sensor_msgs/msg/PointField']
    mono_msg = Image(
        rgb_msg.header, height, width, 'mono8', 0, width, mono.reshape(-1)
    )
    xyz_fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    depth_cloud = PointCloud2(
        depth_msg.header, height, width, xyz_fields, False, 12, 12 * width,
        xyz.view(np.uint8).reshape(-1), False,
    )
    rgb_cloud = PointCloud2(
        depth_msg.header, height, width,
        xyz_fields + [PointField('rgb', 12, PointField.UINT32, 1)],
        False, 16, 16 * width, xyzrgb.view(np.uint8).reshape(-1), False,
    )
    reference_points = np.column_stack([
        xyz['x'][valid], xyz['y'][valid], xyz['z'][valid]
    ]).astype(np.float32)
    reference_points = reference_points[
        (reference_points[:, 2] > 0.1) & (reference_points[:, 2] < 6.0)
    ][::2]
    return mono_msg, depth_cloud, rgb_cloud, reference_points


def ros2_store():
    distro = os.environ.get('ROS_DISTRO', '').upper()
    return get_typestore(getattr(Stores, f'ROS2_{distro}', Stores.LATEST))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=float, default=0.0,
                        help='skip first N seconds (default: 0)')
    parser.add_argument('--duration', type=float,
                        help='process at most N seconds')
    parser.add_argument('--nth', type=int, default=1,
                        help='save every Nth RGB-D frame (default: 1)')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip N selected frames initially (default: 0)')
    parser.add_argument('--compress', action='store_true',
                        help='use ROS 2 zstd message compression')
    parser.add_argument('--groundtruth', type=Path,
                        help='TUM trajectory file (default: sole gt*.txt beside input)')
    parser.add_argument('inputbag', type=Path, help='ROS 1 or ROS 2 input bag')
    parser.add_argument('outputbag', type=Path, nargs='?',
                        help='ROS 2 output bag directory')
    args = parser.parse_args()
    if args.start < 0 or args.duration is not None and args.duration <= 0:
        parser.error('--start must be nonnegative and --duration must be positive')
    if args.nth <= 0 or args.skip < 0:
        parser.error('--nth must be positive and --skip must be nonnegative')
    if args.outputbag is None:
        args.outputbag = args.inputbag.with_suffix('').with_name(
            args.inputbag.stem + '-points'
        )
    if args.groundtruth is None:
        matches = list(args.inputbag.parent.glob('gt*.txt'))
        if len(matches) == 1:
            args.groundtruth = matches[0]
    if args.groundtruth is not None and not args.groundtruth.is_file():
        parser.error(f'ground-truth file not found: {args.groundtruth}')
    return args


def main():
    args = parse_args()
    print(f'Input:  {args.inputbag}')
    print(f'Output: {args.outputbag} (ROS 2)')
    groundtruth = load_groundtruth(args.groundtruth) if args.groundtruth else None
    print(f'Ground truth: {args.groundtruth or "not found"}')

    store = ros2_store()
    writer = Writer(args.outputbag, version=Writer.VERSION_LATEST)
    if args.compress:
        writer.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)

    with AnyReader([args.inputbag]) as reader, writer:
        connmap, converters = create_connections_converters(
            reader.connections, store, reader, writer
        )
        mono_conn = writer.add_connection(
            '/camera/rgb/image_mono', 'sensor_msgs/msg/Image', typestore=store
        )
        depth_cloud_conn = writer.add_connection(
            '/camera/depth/points', 'sensor_msgs/msg/PointCloud2', typestore=store
        )
        rgb_cloud_conn = writer.add_connection(
            '/camera/rgb/points', 'sensor_msgs/msg/PointCloud2', typestore=store
        )
        gt_conn = writer.add_connection(
            '/ground_truth/odom', 'nav_msgs/msg/Odometry', typestore=store
        ) if groundtruth is not None else None
        gt_cloud_conn = writer.add_connection(
            '/ground_truth/visible_cloud', 'sensor_msgs/msg/PointCloud2',
            typestore=store
        ) if groundtruth is not None else None
        gt_map_conn = writer.add_connection(
            '/ground_truth/visible_map', 'sensor_msgs/msg/PointCloud2',
            typestore=store
        ) if groundtruth is not None else None
        gt_pair_tf_conn = writer.add_connection(
            '/ground_truth/pair_tf', 'tf2_msgs/msg/TFMessage', typestore=store
        ) if groundtruth is not None else None

        cached = {}
        transforms = {}
        previous_reference = None
        groundtruth_voxels = {}
        last_map_stamp = None
        frame = saved = groundtruth_saved = paired_saved = 0
        groundtruth_index = 0
        start_ns = None
        first_ns = int(args.start * 1e9)
        end_ns = None if args.duration is None else int(
            (args.start + args.duration) * 1e9
        )

        def write_source(connection, timestamp, raw):
            writer.write(
                connmap[(connection.id, connection.owner)], timestamp,
                converters[connection.msgtype](raw),
            )

        def write_groundtruth_until(timestamp):
            nonlocal groundtruth_index, groundtruth_saved
            if gt_conn is None:
                return
            while (groundtruth_index < len(groundtruth)
                   and int(round(groundtruth[groundtruth_index, 0] * 1e9)) <= timestamp):
                pose = groundtruth[groundtruth_index]
                pose_ns = int(round(pose[0] * 1e9))
                pose_stamp = store.types['builtin_interfaces/msg/Time'](
                    pose_ns // 1_000_000_000, pose_ns % 1_000_000_000
                )
                odometry = make_odometry(store, pose_stamp, pose)
                writer.write(gt_conn, pose_ns, store.serialize_cdr(
                    odometry, gt_conn.msgtype
                ))
                groundtruth_index += 1
                groundtruth_saved += 1

        for connection, timestamp, raw in reader.messages():
            if start_ns is None:
                start_ns = timestamp
                if groundtruth is not None:
                    groundtruth_index = int(np.searchsorted(
                        groundtruth[:, 0], (start_ns + first_ns) / 1e9
                    ))
            elapsed = timestamp - start_ns
            if elapsed < first_ns:
                continue
            if end_ns is not None and elapsed > end_ns:
                break

            write_groundtruth_until(timestamp)

            topic = connection.topic
            if topic == '/tf':
                tf_msg = reader.deserialize(raw, connection.msgtype)
                for transform in tf_msg.transforms:
                    transforms[(transform.header.frame_id, transform.child_frame_id)] = transform
                continue
            if topic in CAMERA_TOPICS:
                cached[topic] = (connection, timestamp, raw)
            elif topic == '/imu':
                continue
            else:
                write_source(connection, timestamp, raw)
                continue

            if topic != '/camera/depth/image' or not CAMERA_TOPICS.issubset(cached):
                continue

            depth_item = cached['/camera/depth/image']
            rgb_item = cached['/camera/rgb/image_color']
            depth_msg = reader.deserialize(depth_item[2], depth_item[0].msgtype)
            rgb_msg = reader.deserialize(rgb_item[2], rgb_item[0].msgtype)
            if abs(stamp_ns(depth_msg.header.stamp) - stamp_ns(rgb_msg.header.stamp)) > 1e9 / 30:
                continue

            frame += 1
            if frame % args.nth == 0:
                if args.skip:
                    args.skip -= 1
                else:
                    if transforms:
                        tf_type = store.types['tf2_msgs/msg/TFMessage']
                        tf_conn = next(c for c in writer.connections if c.topic == '/tf')
                        writer.write(tf_conn, timestamp, store.serialize_cdr(
                            tf_type(list(transforms.values())), tf_conn.msgtype
                        ))
                        transforms.clear()

                    for camera_topic in CAMERA_TOPICS:
                        item = cached[camera_topic]
                        if camera_topic == '/camera/depth/image':
                            message = depth_msg
                        elif camera_topic == '/camera/rgb/image_color':
                            message = rgb_msg
                        else:
                            message = reader.deserialize(item[2], item[0].msgtype)
                        message.header.stamp = rgb_msg.header.stamp
                        output_connection = connmap[(item[0].id, item[0].owner)]
                        normalized_raw = reader.typestore.serialize_ros1(
                            message, item[0].msgtype
                        )
                        writer.write(
                            output_connection, timestamp,
                            converters[item[0].msgtype](normalized_raw),
                        )

                    info_item = cached['/camera/depth/camera_info']
                    info = reader.deserialize(info_item[2], info_item[0].msgtype)
                    mono, depth_cloud, rgb_cloud, reference_points = make_clouds(
                        store, depth_msg, rgb_msg, info
                    )
                    writer.write(mono_conn, timestamp,
                                 store.serialize_cdr(mono, mono_conn.msgtype))
                    writer.write(depth_cloud_conn, timestamp,
                                 store.serialize_cdr(depth_cloud, depth_cloud_conn.msgtype))
                    writer.write(rgb_cloud_conn, timestamp,
                                 store.serialize_cdr(rgb_cloud, rgb_cloud_conn.msgtype))
                    if gt_conn is not None:
                        pair_stamp = rgb_msg.header.stamp
                        pose = interpolate_groundtruth(
                            groundtruth, stamp_ns(pair_stamp)
                        )
                        if pose is not None:
                            merge_world_voxels(
                                groundtruth_voxels, reference_points, pose
                            )
                            last_map_stamp = pair_stamp
                        if pose is not None and previous_reference is not None:
                            previous_points, previous_pose = previous_reference
                            paired_cloud, pair_transform = paired_groundtruth(
                                previous_points, previous_pose, reference_points, pose
                            )
                            pair_tf = make_transform_message(
                                store, pair_stamp, pair_transform
                            )
                            visible_cloud = make_xyz_cloud(
                                store, pair_stamp, paired_cloud
                            )
                            writer.write(gt_pair_tf_conn, timestamp, store.serialize_cdr(
                                pair_tf, gt_pair_tf_conn.msgtype
                            ))
                            writer.write(gt_cloud_conn, timestamp, store.serialize_cdr(
                                visible_cloud, gt_cloud_conn.msgtype
                            ))
                            paired_saved += 1
                        previous_reference = (
                            (reference_points, pose) if pose is not None else None
                        )
                    saved += 1

            cached.pop('/camera/depth/image', None)
            cached.pop('/camera/rgb/image_color', None)

        if groundtruth_voxels and last_map_stamp is not None:
            visible_map = make_xyz_cloud(
                store, last_map_stamp,
                np.asarray(list(groundtruth_voxels.values())), frame_id='world',
            )
            writer.write(gt_map_conn, timestamp, store.serialize_cdr(
                visible_map, gt_map_conn.msgtype
            ))

    print(f'Saved {saved} RGB-D frames to ROS 2 bag {args.outputbag}')
    if groundtruth is not None:
        print(f'Saved {groundtruth_saved} ground-truth poses on /ground_truth/odom')
        print(f'Saved {paired_saved} paired reference clouds and transforms')


if __name__ == '__main__':
    main()
