"""Preflight camera stream health gate for keyboard XY collection.

Run before a collection session (with the RealSense ROS node up and the
machine in its collection-time state — no browser or other heavy load):

    uv run scripts/check_camera_health.py [--duration 60]

Exits 0 when the stream satisfies the episode-survival gate (newest frame
older than the freshness bound for at most 0.05% of the time, no delivery gap
over 300 ms), otherwise prints the problems and exits 1. The same gate runs
automatically at collector startup; `--skip-camera-check` there bypasses it.
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
