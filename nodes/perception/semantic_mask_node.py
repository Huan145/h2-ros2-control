#!/usr/bin/env python3
"""
Converts Isaac Sim semantic segmentation (32SC1) to a binary mask (mono8)
for use with FoundationPose.

Looks up the target object's semantic ID dynamically by label name,
so it works even when Isaac Sim reassigns IDs across sessions.

Subscribes: /semantic_segmentation         (sensor_msgs/Image, 32SC1)
            /semantic_labels               (std_msgs/String, JSON label map)
Publishes:  /segmentation_mask             (sensor_msgs/Image, mono8)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import json


# ── Change this to match your semantic label (set in Semantics Schema Editor)
TARGET_LABEL = 'klt_box'   # the 'class' value you assigned in Isaac Sim
# ─────────────────────────────────────────────────────────────────────────────


class SemanticMaskNode(Node):

    def __init__(self):
        super().__init__('semantic_mask_node')

        # The current semantic ID for our target — discovered at runtime
        self.target_id = None

        # Subscribe to the segmentation image
        self.sub_seg = self.create_subscription(
            Image,
            '/semantic_segmentation',
            self.seg_callback,
            10
        )

        # Subscribe to the label map that Isaac Sim publishes alongside the image
        # This topic publishes a JSON string mapping ID → label
        self.sub_labels = self.create_subscription(
            String,
            '/semantic_labels',
            self.labels_callback,
            10
        )

        self.pub = self.create_publisher(
            Image,
            '/segmentation_mask',
            10
        )

        self.get_logger().info(
            f'SemanticMaskNode started. Looking for label="{TARGET_LABEL}"'
        )

    def labels_callback(self, msg: String):
        """
        Isaac Sim publishes a JSON label map alongside the segmentation image.
        Example: {"0": {"class": "BACKGROUND"}, "2": {"class": "klt_box"}, ...}
        We parse this to find which ID corresponds to our target label.
        """
        try:
            label_map = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse label map JSON')
            return

        new_id = None
        for id_str, label_dict in label_map.items():
            # label_dict can be {'class': 'klt_box'} or {'klt_box_1': 'klt_box', 'class': '...'}
            # Check all values in the dict for our target label
            values = list(label_dict.values()) if isinstance(label_dict, dict) else [label_dict]
            for val in values:
                if TARGET_LABEL.lower() in str(val).lower():
                    new_id = int(id_str)
                    break
            if new_id is not None:
                break

        if new_id is None:
            self.get_logger().warn(
                f'Label "{TARGET_LABEL}" not found in label map. '
                f'Available labels: {label_map}',
                throttle_duration_sec=5.0
            )
            return

        if new_id != self.target_id:
            self.get_logger().info(
                f'Found "{TARGET_LABEL}" → semantic ID={new_id} '
                f'(was {self.target_id})'
            )
            self.target_id = new_id

    def seg_callback(self, msg: Image):
        """Convert semantic segmentation image to binary mask."""

        if self.target_id is None:
            self.get_logger().warn(
                f'Waiting for label map to find ID for "{TARGET_LABEL}". '
                f'Checking if /semantic_labels is published...',
                throttle_duration_sec=5.0
            )
            # Fallback: print unique IDs to help debugging
            raw = np.frombuffer(msg.data, dtype=np.int32)
            unique_ids = np.unique(raw)
            self.get_logger().info(
                f'Unique semantic IDs in current frame: {unique_ids}',
                throttle_duration_sec=5.0
            )
            return

        # Convert raw bytes → numpy int32 array
        raw = np.frombuffer(msg.data, dtype=np.int32)
        seg = raw.reshape((msg.height, msg.width))

        # Binary mask: target object = 255, everything else = 0
        mask = np.where(seg == self.target_id, 255, 0).astype(np.uint8)

        num_pixels = int(np.sum(mask > 0))
        if num_pixels == 0:
            self.get_logger().warn(
                f'No pixels found for "{TARGET_LABEL}" (ID={self.target_id}). '
                f'Is the KLT box in the camera frame?',
                throttle_duration_sec=5.0
            )

        # Build output message
        out = Image()
        out.header       = msg.header
        out.height       = msg.height
        out.width        = msg.width
        out.encoding     = 'mono8'
        out.is_bigendian = 0
        out.step         = msg.width
        out.data         = mask.tobytes()

        self.pub.publish(out)

        self.get_logger().info(
            f'Mask published: {num_pixels}/{msg.width * msg.height} pixels '
            f'are "{TARGET_LABEL}" (ID={self.target_id})',
            throttle_duration_sec=2.0
        )


def main():
    rclpy.init()
    node = SemanticMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
