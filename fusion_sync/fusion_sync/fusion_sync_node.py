import rclpy
from rclpy.node import Node

import os
import numpy as np
import cv2

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

from message_filters import Subscriber, ApproximateTimeSynchronizer

from cv_bridge import CvBridge

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
# from fusion_sync.msg import SyncedSensors
# from fusion_sync import bag_extraction as be 

class SyncNode(Node):

    def __init__(self):

        super().__init__("sync_node")

        self.declare_parameter("mode", "realtime")
        self.declare_parameter("input_dir", "")
        self.declare_parameter("output_dir", "./dataset")

        self.mode = self.get_parameter("mode").value

        self.bridge = CvBridge()

        # if self.mode == "realtime":
        #     self.start_realtime()
        # else:
        #     self.input_dir = self.get_parameter("input_dir").value
        #     self.output_dir = self.get_parameter("output_dir").value

        self.input_dir = self.get_parameter("input_dir").value
        self.output_dir = self.get_parameter("output_dir").value
        self.make_folders()

    # =========================
    # REALTIME MODE
    # =========================

    # def start_realtime(self):

    #     self.get_logger().info("Realtime synchronization mode")

    #     self.rgb_sub = Subscriber(self, Image, "/harrier_cam_node/image_raw")
    #     self.thermal_sub = Subscriber(self, Image, "/cellplus_cam_node/color/image_raw")
    #     self.thermal_raw_sub = Subscriber(self, Image, "/cellplus_cam_node/thermal/image_raw")
    #     self.lidar_sub = Subscriber(self, PointCloud2, "/livox/lidar")

    #     self.sync = ApproximateTimeSynchronizer(
    #         [self.rgb_sub, self.thermal_sub, self.thermal_raw_sub, self.lidar_sub],
    #         queue_size=20,
    #         slop=0.1
    #     )

    #     self.sync.registerCallback(self.sync_callback)

    #     self.publisher = self.create_publisher(
    #         SyncedSensors,
    #         "/synced_sensors",
    #         10
    #     )

    # def sync_callback(self, rgb, thermal, thermal_raw, lidar):

    #     msg = SyncedSensors()

    #     msg.header = Header()
    #     msg.header.stamp = self.get_clock().now().to_msg()

    #     msg.rgb = rgb
    #     msg.thermal = thermal
    #     msg.thermal_raw = thermal_raw
    #     msg.lidar = lidar

    #     self.publisher.publish(msg)

    # =========================
    # ROSBAG MODE
    # =========================

    def make_folders(self):
        for task_name in os.listdir(self.input_dir):
            task_path = os.path.join(self.input_dir, task_name)
            if not os.path.isdir(task_path):
                continue
            for pan_tilt in os.listdir(task_path):
                pan_tilt_path = os.path.join(task_path, pan_tilt)
                if not os.path.isdir(pan_tilt_path):
                    continue
                for bag in os.listdir(pan_tilt_path) :
                    bag_path = os.path.join(pan_tilt_path, bag)
                    if not os.path.isdir(bag_path):
                        continue

                    save_path = os.path.join(self.output_dir, task_name, pan_tilt, bag)
                    self.start_rosbag(bag_path, save_path)

    def start_rosbag(self, bag_path, save_path):

        self.get_logger().info(f"Reading bag: {bag_path}")

        os.makedirs(save_path + "/rgb", exist_ok=True)
        os.makedirs(save_path + "/thermal", exist_ok=True)
        os.makedirs(save_path + "/thermal_raw", exist_ok=True)

        reader = rosbag2_py.SequentialReader()

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path,
            storage_id="mcap"
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        rgb_msgs = []
        thermal_msgs = []
        thermal_raw_msgs = []

        while reader.has_next():

            topic, data, t = reader.read_next()

            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == "/harrier_cam_node/image_raw":
                rgb_msgs.append(msg)

            elif topic == "/cellplus_cam_node/color/image_raw":
                thermal_msgs.append(msg)

            elif topic == "/cellplus_cam_node/thermal/image_raw":
                thermal_raw_msgs.append(msg)

        self.sync_msgs(save_path, rgb_msgs, thermal_msgs, thermal_raw_msgs)

    def to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def sync_msgs(self, save_path, rgb, thermal, thermal_raw):

        frame = 0

        for l in thermal:

            t = self.to_sec(l.header.stamp)

            rgb_match = min(rgb, key=lambda x: abs(self.to_sec(x.header.stamp) - t))
            thermal_match = min(thermal, key=lambda x: abs(self.to_sec(x.header.stamp) - t))
            thermal_raw_match = min(thermal_raw, key=lambda x: abs(self.to_sec(x.header.stamp) - t))

            self.save_frame(frame, save_path, rgb_match, thermal_match, thermal_raw_match)

            frame += 1

    def save_frame(self, idx, save_path, rgb, thermal, thermal_raw):

        rgb_img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        cv2.imwrite(f"{save_path}/rgb/{idx:06d}.png", rgb_img)

        thermal_img = self.bridge.imgmsg_to_cv2(thermal)
        cv2.imwrite(f"{save_path}/thermal/{idx:06d}.png", thermal_img)

        thermal_raw_img = self.bridge.imgmsg_to_cv2(thermal_raw, desired_encoding='passthrough')
        np.save(f"{save_path}/thermal_raw/{idx:06d}.npy", thermal_raw_img)

        self.get_logger().info(f"Saved frame {idx}")


def main():

    rclpy.init()
    node = SyncNode()

    # be.BagExtractor(node)

    node.destroy_node()
    rclpy.shutdown()