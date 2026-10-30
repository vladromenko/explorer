"""Small CPU-only ROS camera cache used by the local web panel."""
import io
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage

ROOT = Path('/home/vlad/Explorer')


def atomic(path, content):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


class CameraCache(Node):
    def __init__(self):
        super().__init__('explorer_camera_cache', enable_rosout=False)
        self.last_write = 0.0
        self.create_subscription(RosImage, '/camera/color/image_raw', self.receive, qos_profile_sensor_data)

    def receive(self, message):
        now = time.monotonic()
        if now - self.last_write < 0.12 or message.encoding.lower() not in ('rgb8', 'bgr8'):
            return
        if message.width <= 0 or message.height <= 0 or message.step < message.width * 3:
            return
        array = np.ndarray((message.height, message.width, 3), dtype=np.uint8,
                           buffer=message.data, strides=(message.step, 3, 1)).copy()
        if message.encoding.lower() == 'bgr8':
            array = array[:, :, ::-1]
        output = io.BytesIO()
        Image.fromarray(array).save(output, format='JPEG', quality=82)
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        metadata = json.dumps({
            'at': time.time(), 'image_stamp': stamp, 'frame': message.header.frame_id,
            'width': message.width, 'height': message.height, 'source': 'ros_color_cpu_cache',
        }, ensure_ascii=False, separators=(',', ':')).encode()
        atomic(ROOT / 'data/frame-raw.jpg', output.getvalue())
        atomic(ROOT / 'data/camera-frame.json', metadata)
        self.last_write = now


def main():
    rclpy.init()
    node = CameraCache()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
