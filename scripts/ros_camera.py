"""ROS 2 Jazzy image subscription and latest-frame buffering."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageSnapshot:
    rgb: np.ndarray
    source_timestamp_ns: int
    receipt_monotonic_ns: int
    sequence: int


class LatestImageStore:
    """Thread-safe single-slot image store with freshness checks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: ImageSnapshot | None = None
        self._sequence = 0

    def update(
        self,
        rgb: np.ndarray,
        source_timestamp_ns: int,
        receipt_monotonic_ns: int | None = None,
    ) -> int:
        image = np.asarray(rgb)
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError("ROS image must be RGB uint8 with shape (480, 640, 3)")
        receipt = time.monotonic_ns() if receipt_monotonic_ns is None else int(receipt_monotonic_ns)
        with self._lock:
            self._sequence += 1
            self._snapshot = ImageSnapshot(
                rgb=image.copy(),
                source_timestamp_ns=int(source_timestamp_ns),
                receipt_monotonic_ns=receipt,
                sequence=self._sequence,
            )
            return self._sequence

    def snapshot(self) -> ImageSnapshot | None:
        with self._lock:
            return self._snapshot

    @staticmethod
    def accept(
        snapshot: ImageSnapshot | None,
        previous_sequence: int | None,
        now_monotonic_ns: int,
        max_age_s: float,
    ) -> bool:
        if snapshot is None:
            return False
        if previous_sequence is not None and snapshot.sequence <= int(previous_sequence):
            return False
        age_s = (int(now_monotonic_ns) - snapshot.receipt_monotonic_ns) / 1e9
        return 0.0 <= age_s <= float(max_age_s)


class RosImageSubscriber:
    """Lazy ROS 2 Jazzy adapter; importable on hosts without ROS installed."""

    def __init__(
        self,
        store: LatestImageStore,
        topic: str = "/camera/camera/color/image_raw",
    ) -> None:
        self.store = store
        self.topic = topic
        self.error: Exception | None = None
        self._rclpy: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_rclpy = False

    def start(self) -> None:
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSHistoryPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Jazzy, sensor_msgs, and cv_bridge must be sourced on the robot host"
            ) from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        store = self.store
        bridge = CvBridge()

        class ImageNode(Node):
            def __init__(self) -> None:
                super().__init__("pushbox_keyboard_image_subscriber")
                qos = QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    durability=QoSDurabilityPolicy.VOLATILE,
                )
                self.subscription = self.create_subscription(
                    Image,
                    self_topic,
                    self.callback,
                    qos,
                )

            def callback(self, message: Any) -> None:
                try:
                    if message.encoding != "rgb8":
                        raise ValueError(f"expected rgb8, received {message.encoding!r}")
                    if message.width != 640 or message.height != 480:
                        raise ValueError(
                            f"expected 640x480, received {message.width}x{message.height}"
                        )
                    rgb = np.asarray(
                        bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"),
                        dtype=np.uint8,
                    )
                    source_ns = int(message.header.stamp.sec) * 1_000_000_000
                    source_ns += int(message.header.stamp.nanosec)
                    store.update(rgb, source_ns, time.monotonic_ns())
                except Exception as exc:  # keep executor alive; main loop checks error
                    subscriber.error = exc

        self_topic = self.topic
        subscriber = self
        self._node = ImageNode()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            name="ros-image-executor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._owns_rclpy and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        self._executor = None
        self._node = None
        self._thread = None
