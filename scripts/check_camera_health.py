"""Preflight camera stream health gate for keyboard XY collection.

Run before a collection session (with the RealSense ROS node up and the
machine in its collection-time state — no browser or other heavy load):

    uv run scripts/check_camera_health.py [--duration 60]

Exits 0 when the stream satisfies the episode-survival gate: stale time within
the 0.05% budget plus at most one routine hiccup (<= 370 ms), no delivery gap
over 500 ms, median gap <= 50 ms, and >= 20 fps measured from the first frame
to the end of the probe (so a stream that dies mid-probe fails). Otherwise it
prints the problems and exits 1. The same gate runs automatically at collector
startup; `--skip-camera-check` there bypasses it.
"""

from __future__ import annotations

import argparse
from typing import Sequence

try:  # package import for tests; direct-file import for ``uv run scripts/...``
    from scripts.ros_camera import HEALTH_MAX_AGE_S, measure_stream_health
except ImportError:  # pragma: no cover - exercised by direct script execution
    from ros_camera import HEALTH_MAX_AGE_S, measure_stream_health


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--duration", type=float, default=60.0, help="probe seconds")
    parser.add_argument(
        "--max-age",
        type=float,
        default=HEALTH_MAX_AGE_S,
        help="freshness bound the collector will use (IMAGE_MAX_AGE_S)",
    )
    args = parser.parse_args(argv)
    print(f"probing {args.topic} for {args.duration:.0f} s ...")
    report = measure_stream_health(args.topic, args.duration, max_age_s=args.max_age)
    print(report.summary())
    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
