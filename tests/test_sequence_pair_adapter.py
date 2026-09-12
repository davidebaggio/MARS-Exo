from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image

from exo_head_slam.sequence_pair_adapter_node import retag_frame, sliding_pair


def test_sliding_pairs_overlap_and_retag_without_mutating_inputs():
    frames = [
        (Image(), Image(), CameraInfo()),
        (Image(), Image(), CameraInfo()),
        (Image(), Image(), CameraInfo()),
    ]

    previous, pair = sliding_pair(None, frames[0])
    assert pair is None
    previous, pair = sliding_pair(previous, frames[1])
    assert pair[0] is frames[0] and pair[1] is frames[1]
    previous, pair = sliding_pair(previous, frames[2])
    assert pair[0] is frames[1] and pair[1] is frames[2]

    stamp = Time(sec=12, nanosec=34)
    tagged = retag_frame(pair[0], stamp, 'head_camera_color_optical_frame')
    assert all(message.header.stamp == stamp for message in tagged)
    assert all(message.header.frame_id == 'head_camera_color_optical_frame'
               for message in tagged)
    assert all(message.header.frame_id == '' for message in pair[0])
