"""ROS 2 Jazzy image subscription, latest-frame buffering, and stream health."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Sequence

import numpy as np


# Stream-health gate defaults, derived from measured episode-survival math
# (2026-07-11): a 150-tick episode survives with p >= 0.9 only when the newest
# frame is older than the freshness bound for <= ~0.05% of the time. Delivery
# hiccups of 100-370 ms are routine on a healthy rig
# (docs/hardware-api-reference.md), so each probe tolerates one routine
# hiccup's worth of stale time on top of the fractional budget; a gap beyond
# the routine ceiling or a median gap above dataset_spec.md's 50 ms degraded-
# stream bound fails regardless of probe length.
HEALTH_MAX_AGE_S = 0.20
HEALTH_MAX_STALE_FRACTION = 0.0005
HEALTH_ROUTINE_HICCUP_MAX_S = 0.370
HEALTH_MAX_GAP_S = 0.500
HEALTH_MAX_MEDIAN_GAP_S = 0.050
HEALTH_MIN_FPS = 20.0

# Contract with the external RealSense publisher (dataset_spec.md): rgb8,
# 640x480 at 60 fps, subscribed best-effort/volatile keeping only the newest
# frame. Every consumer must validate against these constants so the health
# probe, the live subscriber, and the store can never drift apart.
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480


@dataclass(frozen=True)
class ImageSnapshot:
    rgb: np.ndarray
    source_timestamp_ns: int
    receipt_monotonic_ns: int
    sequence: int


def decode_rgb8(message: Any) -> np.ndarray:
    """Decode a sensor_msgs/msg/Image rgb8 payload with NumPy only.

    Jazzy's cv_bridge extension is compiled against NumPy 1.x and crashes
    under this project's NumPy >= 2 pin; an rgb8 payload needs no color
    conversion, only a bounded reshape that honors the row stride.
    """
    if message.encoding != "rgb8":
        raise ValueError(f"expected rgb8, received {message.encoding!r}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    if step < width * 3:
        raise ValueError(f"row step {step} is too small for width {width}")
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size != height * step:
        raise ValueError(f"payload of {data.size} bytes does not match {height}x{step}")
    return data.reshape(height, step)[:, : width * 3].reshape(height, width, 3)


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
        if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"ROS image must be RGB uint8 with shape ({EXPECTED_HEIGHT}, {EXPECTED_WIDTH}, 3)"
            )
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


def _require_ros() -> tuple[Any, Any, Any]:
    """Import the ROS 2 stack lazily so this module loads on dev machines."""
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Jazzy and sensor_msgs must be sourced on the robot host"
        ) from exc
    return rclpy, SingleThreadedExecutor, Image


def _image_qos() -> Any:
    """Subscription QoS matching the RealSense publisher: best-effort, newest frame only."""
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class RosImageSubscriber:
    """Lazy ROS 2 Jazzy adapter; importable on hosts without ROS installed."""

    def __init__(
        self,
        store: LatestImageStore,
        topic: str = "/camera/camera/color/image_raw",
    ) -> None:
        self.store = store
        self.topic = topic
        self._error_lock = threading.Lock()
        self._error: Exception | None = None
        self._rclpy: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_rclpy = False

    def start(self) -> None:
        rclpy, SingleThreadedExecutor, Image = _require_ros()
        from rclpy.node import Node

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        store = self.store

        class ImageNode(Node):
            def __init__(self) -> None:
                super().__init__("pushbox_keyboard_image_subscriber")
                self.subscription = self.create_subscription(
                    Image,
                    self_topic,
                    self.callback,
                    _image_qos(),
                )

            def callback(self, message: Any) -> None:
                try:
                    if message.width != EXPECTED_WIDTH or message.height != EXPECTED_HEIGHT:
                        raise ValueError(
                            f"expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT},"
                            f" received {message.width}x{message.height}"
                        )
                    rgb = decode_rgb8(message)
                    source_ns = int(message.header.stamp.sec) * 1_000_000_000
                    source_ns += int(message.header.stamp.nanosec)
                    store.update(rgb, source_ns, time.monotonic_ns())
                except Exception as exc:  # keep executor alive; main loop checks error
                    subscriber._record_error(exc)

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

    def _record_error(self, exc: Exception) -> None:
        with self._error_lock:
            self._error = exc

    def take_error(self) -> Exception | None:
        """Return and clear the most recent subscriber callback error."""
        with self._error_lock:
            error, self._error = self._error, None
            return error

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


@dataclass(frozen=True)
class StreamHealthReport:
    message_count: int
    duration_s: float
    fps: float
    median_gap_ms: float
    max_gap_ms: float
    stale_fraction: float
    problems: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = [
            f"camera stream: {self.message_count} msgs in {self.duration_s:.1f} s"
            f" -> {self.fps:.1f} fps",
            f"gaps: median {self.median_gap_ms:.1f} ms, max {self.max_gap_ms:.0f} ms;"
            f" stale-time {self.stale_fraction * 100:.3f}%",
            "verdict: HEALTHY" if self.healthy else "verdict: UNHEALTHY",
        ]
        lines.extend(f"  problem: {problem}" for problem in self.problems)
        return "\n".join(lines)


def evaluate_stream_health(
    receipt_monotonic_ns: Sequence[int],
    duration_s: float,
    *,
    probe_end_monotonic_ns: int | None = None,
    max_age_s: float = HEALTH_MAX_AGE_S,
    max_stale_fraction: float = HEALTH_MAX_STALE_FRACTION,
    max_gap_s: float = HEALTH_MAX_GAP_S,
    max_median_gap_s: float = HEALTH_MAX_MEDIAN_GAP_S,
    min_fps: float = HEALTH_MIN_FPS,
    extra_problems: Sequence[str] = (),
) -> StreamHealthReport:
    """Judge measured frame receipt times against the episode-survival gate.

    Rates and stale time are measured over the span from the first receipt to
    `probe_end_monotonic_ns` (when given): time before the first frame is
    subscription discovery and does not count against the stream, but silence
    after the last frame does — a stream that dies mid-probe must fail.

    `stale_fraction` is the share of that span during which the newest
    available frame was already older than `max_age_s` — exactly the condition
    that makes a 5 Hz recording tick discard the active episode. The stale
    budget is `max_stale_fraction` of the span plus one routine hiccup
    (HEALTH_ROUTINE_HICCUP_MAX_S), so a single documented-normal delivery
    hiccup does not fail an otherwise healthy stream regardless of probe
    length.
    """
    problems = list(extra_problems)
    receipts = np.asarray(receipt_monotonic_ns, dtype=np.int64)
    if receipts.size < 2:
        problems.append(
            f"received {receipts.size} messages in {duration_s:.1f} s; expected a stream"
        )
        return StreamHealthReport(
            message_count=int(receipts.size),
            duration_s=float(duration_s),
            fps=0.0,
            median_gap_ms=float("nan"),
            max_gap_ms=float("nan"),
            stale_fraction=1.0,
            problems=tuple(problems),
        )

    end_ns = int(receipts[-1])
    if probe_end_monotonic_ns is not None:
        end_ns = max(end_ns, int(probe_end_monotonic_ns))
    cadence_gaps_s = np.diff(receipts) / 1e9
    gaps_s = cadence_gaps_s
    tail_gap_s = (end_ns - int(receipts[-1])) / 1e9
    if tail_gap_s > 0.0:
        gaps_s = np.append(gaps_s, tail_gap_s)
    total_s = float(end_ns - int(receipts[0])) / 1e9
    fps = receipts.size / total_s if total_s > 0 else 0.0
    stale_s = float(np.clip(gaps_s - max_age_s, 0.0, None).sum())
    stale_fraction = stale_s / total_s if total_s > 0 else 1.0
    stale_budget_s = max_stale_fraction * total_s + max(HEALTH_ROUTINE_HICCUP_MAX_S - max_age_s, 0.0)
    max_gap_s_measured = float(gaps_s.max())
    median_gap_s = float(np.median(cadence_gaps_s))

    if fps < min_fps:
        problems.append(f"rate {fps:.1f} fps is below the {min_fps:.0f} fps minimum")
    if max_gap_s_measured > max_gap_s:
        problems.append(
            f"largest delivery gap {max_gap_s_measured * 1000:.0f} ms exceeds"
            f" {max_gap_s * 1000:.0f} ms"
        )
    if median_gap_s > max_median_gap_s:
        problems.append(
            f"median delivery gap {median_gap_s * 1000:.1f} ms exceeds"
            f" {max_median_gap_s * 1000:.0f} ms; the stream is degraded"
        )
    if stale_s > stale_budget_s:
        problems.append(
            f"newest frame older than {max_age_s:.2f} s for {stale_s:.2f} s of"
            f" {total_s:.1f} s (budget {stale_budget_s:.2f} s ="
            f" {max_stale_fraction * 100:.3f}% plus one routine hiccup);"
            " episodes would frequently be discarded"
        )

    return StreamHealthReport(
        message_count=int(receipts.size),
        duration_s=float(duration_s),
        fps=float(fps),
        median_gap_ms=median_gap_s * 1000,
        max_gap_ms=max_gap_s_measured * 1000,
        stale_fraction=stale_fraction,
        problems=tuple(problems),
    )


def measure_stream_health(
    topic: str = "/camera/camera/color/image_raw",
    duration_s: float = 15.0,
    *,
    max_age_s: float = HEALTH_MAX_AGE_S,
) -> StreamHealthReport:
    """Subscribe for `duration_s` and gate the live stream (own ROS context).

    Uses a private rclpy context so it can run before, and independently of,
    `RosImageSubscriber`. Also validates the first message's encoding/shape.
    """
    rclpy, SingleThreadedExecutor, Image = _require_ros()

    context = rclpy.Context()
    rclpy.init(args=None, context=context)
    receipts: list[int] = []
    format_problems: list[str] = []

    def callback(message: Any) -> None:
        receipts.append(time.monotonic_ns())
        if len(receipts) == 1:
            try:
                if message.width != EXPECTED_WIDTH or message.height != EXPECTED_HEIGHT:
                    raise ValueError(
                        f"resolution {message.width}x{message.height} is not"
                        f" {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
                    )
                decode_rgb8(message)
            except ValueError as exc:
                format_problems.append(str(exc))

    node = rclpy.create_node("pushbox_camera_health_probe", context=context)
    executor = SingleThreadedExecutor(context=context)
    try:
        node.create_subscription(Image, topic, callback, _image_qos())
        executor.add_node(node)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        probe_end_ns = time.monotonic_ns()
    finally:
        executor.shutdown()
        node.destroy_node()
        context.try_shutdown()

    return evaluate_stream_health(
        receipts,
        duration_s,
        probe_end_monotonic_ns=probe_end_ns,
        max_age_s=max_age_s,
        extra_problems=format_problems,
    )
