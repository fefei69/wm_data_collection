"""ROS 2 + Pygame keyboard collector for fixed-height follower-arm pushing.

The module keeps validation and episode transaction helpers importable on a
development machine.  Hardware, ROS 2, Pygame, and the dataset writer are
loaded only from :func:`main` so unit tests do not need the robot host stack.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # package import for tests; direct-file import for ``uv run scripts/...``
    from scripts.collector_core import (
        ImageTransformProfile,
        build_episode_columns,
        validate_magnitudes,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution
    from collector_core import ImageTransformProfile, build_episode_columns, validate_magnitudes


ACTION_CAP_M = 0.010
TICK_S = 0.2
# 0.20 (was 0.10): measured 2026-07-11 at 640x480x60 — typical frame age is
# 17-35 ms, but delivery hiccups of 100-200 ms are routine, and one tick is
# the hard ceiling anyway (each tick also requires a NEW frame). The logged
# image_receipt/command timestamps let QA audit the true lag per row.
IMAGE_MAX_AGE_S = 0.20
RESET_STEP_M = 0.005
FIXED_ORIENTATION = np.array([0.0, math.pi / 2.0, 0.0], dtype=np.float64)
DEFAULT_MAGNITUDES = {1: 0.0025, 2: 0.005, 3: 0.010}


def accepted_target(
    current_xy: np.ndarray,
    requested_delta: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a requested XY target and return ``(target, actual_delta)``."""
    current = np.asarray(current_xy, dtype=np.float64)
    delta = np.asarray(requested_delta, dtype=np.float64)
    if current.shape != (2,) or delta.shape != (2,):
        raise ValueError("current_xy and requested_delta must have shape (2,)")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(delta)):
        raise ValueError("current_xy and requested_delta must be finite")
    lower = np.array([bounds[0][0], bounds[1][0]], dtype=np.float64)
    upper = np.array([bounds[0][1], bounds[1][1]], dtype=np.float64)
    if np.any(lower >= upper):
        raise ValueError("workspace bounds must be strictly increasing")
    target = np.clip(current + delta, lower, upper)
    return target, target - current


@dataclass(frozen=True)
class CollectorConfig:
    output_h5: Path = Path("pushbox_keyboard.h5")
    output_video_dir: Path = Path("output_videos")
    transform_profile_path: Path = Path()
    camera_params_path: Path = Path()
    transform_profile: ImageTransformProfile = field(repr=False, default=None)  # type: ignore[assignment]
    follower_ip: str = "192.168.1.3"
    fixed_z: float = 0.20
    safe_z: float = 0.30
    start_xy: tuple[float, float] = (0.30, 0.0)
    x_bounds: tuple[float, float] = (0.15, 0.45)
    y_bounds: tuple[float, float] = (-0.25, 0.25)
    trajectory_check_samples: int = 10
    magnitudes: Mapping[int, float] = field(default_factory=lambda: dict(DEFAULT_MAGNITUDES))

    @classmethod
    def from_paths(
        cls,
        transform_profile: str | Path,
        camera_params: str | Path,
        **kwargs: Any,
    ) -> "CollectorConfig":
        profile_path = Path(transform_profile)
        camera_path = Path(camera_params)
        if not profile_path.is_file():
            raise ValueError(f"transform profile file does not exist: {profile_path}")
        if not camera_path.is_file():
            raise ValueError(f"camera parameter dump does not exist: {camera_path}")
        try:
            profile_data = json.loads(profile_path.read_text())
            profile = ImageTransformProfile.from_dict(profile_data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid transform profile: {profile_path}") from exc
        magnitudes = dict(kwargs.pop("magnitudes", DEFAULT_MAGNITUDES))
        validate_magnitudes(magnitudes, ACTION_CAP_M)
        config = cls(
            transform_profile_path=profile_path,
            camera_params_path=camera_path,
            transform_profile=profile,
            magnitudes=magnitudes,
            **kwargs,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.follower_ip:
            raise ValueError("follower_ip must not be empty")
        values = [self.fixed_z, self.safe_z, *self.start_xy, *self.x_bounds, *self.y_bounds]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("pose and workspace values must be finite")
        if self.safe_z <= self.fixed_z:
            raise ValueError("safe_z must be above fixed_z")
        if self.x_bounds[0] >= self.x_bounds[1] or self.y_bounds[0] >= self.y_bounds[1]:
            raise ValueError("workspace bounds must be strictly increasing")
        if not (self.x_bounds[0] <= self.start_xy[0] <= self.x_bounds[1]):
            raise ValueError("start x is outside workspace bounds")
        if not (self.y_bounds[0] <= self.start_xy[1] <= self.y_bounds[1]):
            raise ValueError("start y is outside workspace bounds")
        if self.trajectory_check_samples < 1:
            raise ValueError("trajectory_check_samples must be positive")
        validate_magnitudes(self.magnitudes, ACTION_CAP_M)


class EpisodeRecorder:
    """Transactional per-episode row and preview-frame buffer."""

    def __init__(self, output_h5: str | Path | None = None, video_path: str | Path | None = None, episode_id: int = 0):
        self.output_h5 = Path(output_h5) if output_h5 is not None else None
        self.video_path = Path(video_path) if video_path is not None else None
        self.episode_id = int(episode_id)
        self.rows: list[Mapping[str, Any]] = []
        self.video_frames: list[np.ndarray] = []

    def start(self) -> None:
        self.rows.clear()
        self.video_frames.clear()

    def append(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))
        if "pixels" in row:
            self.video_frames.append(np.asarray(row["pixels"], dtype=np.uint8).copy())

    def send_and_commit(self, row: Mapping[str, Any], send: Callable[[], Any]) -> None:
        """Send first; commit the row only if the driver call succeeds."""
        try:
            send()
        except Exception:
            raise
        self.append(row)

    def finish(self, save: bool = True) -> dict[str, np.ndarray] | None:
        if not self.rows:
            return None
        if "action" in self.rows[-1] and not np.allclose(self.rows[-1]["action"], 0.0):
            raise ValueError("episode must end with a zero action row")
        columns = build_episode_columns(self.rows, self.episode_id) if "pixels" in self.rows[0] else None
        if save and columns is not None and self.output_h5 is not None:
            self.output_h5.parent.mkdir(parents=True, exist_ok=True)
            import stable_worldmodel as swm

            with swm.data.HDF5Writer(str(self.output_h5), mode="append") as writer:
                writer.write_episode(columns)
            if self.video_path is not None and self.video_frames:
                _write_preview_video(self.video_path, self.video_frames)
        return columns


def _write_preview_video(path: Path, frames: Sequence[np.ndarray]) -> None:
    import imageio_ffmpeg
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "224x224", "-r", "5", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
        if process.wait(timeout=10) != 0:
            raise RuntimeError("ffmpeg failed to write preview video")


class FollowerArmAdapter:
    """Small adapter isolating the pinned Trossen API from the collector loop."""

    def __init__(self, driver: Any, trossen_module: Any, config: CollectorConfig):
        self.driver = driver
        self.trossen = trossen_module
        self.config = config

    @classmethod
    def connect(cls, config: CollectorConfig) -> "FollowerArmAdapter":
        import trossen_arm

        driver = trossen_arm.TrossenArmDriver()
        driver.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_follower, config.follower_ip, False)
        driver.set_all_modes(trossen_arm.Mode.position)
        return cls(driver, trossen_arm, config)

    def home(self, positions: np.ndarray, goal_time: float = 2.0) -> None:
        self.driver.set_all_positions(np.asarray(positions, dtype=np.float64), goal_time, True)

    def pose(self) -> np.ndarray:
        return np.asarray(self.driver.get_cartesian_positions(), dtype=np.float64).reshape(-1)[:6]

    def velocities(self) -> np.ndarray:
        return np.asarray(self.driver.get_cartesian_velocities(), dtype=np.float64).reshape(-1)[:6]

    def send_cartesian(self, target6: np.ndarray, *, goal_time: float = TICK_S, blocking: bool = False) -> None:
        target = np.asarray(target6, dtype=np.float64).reshape(6)
        interpolation = self.trossen.InterpolationSpace.cartesian
        self.driver.set_cartesian_positions(
            target,
            interpolation,
            float(goal_time),
            bool(blocking),
            None,
            None,
            int(self.config.trajectory_check_samples),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("pushbox_keyboard.h5"))
    parser.add_argument("--video-dir", type=Path, default=Path("output_videos"))
    parser.add_argument("--transform-profile", required=True, type=Path)
    parser.add_argument("--camera-params", required=True, type=Path)
    parser.add_argument("--follower-ip", default="192.168.1.3")
    parser.add_argument("--fixed-z", required=True, type=float)
    parser.add_argument("--safe-z", required=True, type=float)
    parser.add_argument("--start-x", required=True, type=float)
    parser.add_argument("--start-y", required=True, type=float)
    parser.add_argument("--x-min", required=True, type=float)
    parser.add_argument("--x-max", required=True, type=float)
    parser.add_argument("--y-min", required=True, type=float)
    parser.add_argument("--y-max", required=True, type=float)
    parser.add_argument("--trajectory-check-samples", required=True, type=int)
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        help="skip the startup camera stream health gate",
    )
    parser.add_argument(
        "--camera-check-seconds",
        type=float,
        default=15.0,
        help="duration of the startup camera stream health gate",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> CollectorConfig:
    return CollectorConfig.from_paths(
        args.transform_profile,
        args.camera_params,
        output_h5=args.output,
        output_video_dir=args.video_dir,
        follower_ip=args.follower_ip,
        fixed_z=args.fixed_z,
        safe_z=args.safe_z,
        start_xy=(args.start_x, args.start_y),
        x_bounds=(args.x_min, args.x_max),
        y_bounds=(args.y_min, args.y_max),
        trajectory_check_samples=args.trajectory_check_samples,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        from scripts.pygame_input import HeldInput, PygameInputBackend
        from scripts.ros_camera import LatestImageStore, RosImageSubscriber, measure_stream_health
    except ImportError:  # direct-file execution
        from pygame_input import HeldInput, PygameInputBackend
        from ros_camera import LatestImageStore, RosImageSubscriber, measure_stream_health

    if not args.skip_camera_check:
        # Gate the camera stream before any arm connection or motion; a lossy
        # stream makes the 5 Hz freshness check discard nearly every episode.
        report = measure_stream_health(
            duration_s=args.camera_check_seconds, max_age_s=IMAGE_MAX_AGE_S
        )
        print(report.summary())
        if not report.healthy:
            print("camera stream failed the health gate; fix the stream or rerun with --skip-camera-check")
            return 1

    image_store = LatestImageStore()
    ros = RosImageSubscriber(image_store)
    pygame_input = PygameInputBackend(HeldInput(config.magnitudes))
    arm = FollowerArmAdapter.connect(config)
    home_positions = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0])
    recording = False
    quit_requested = False
    pending_stop = False
    normal_exit = False
    target_xy = np.array(config.start_xy, dtype=np.float64)
    previous_sequence: int | None = None
    episode_id = 0
    recorder = EpisodeRecorder(output_h5=config.output_h5)
    preview = np.zeros((224, 224, 3), dtype=np.uint8)

    try:
        arm.home(home_positions)
        safe_pose = np.r_[target_xy, config.safe_z, FIXED_ORIENTATION]
        arm.send_cartesian(safe_pose, goal_time=2.0, blocking=True)
        arm.send_cartesian(np.r_[target_xy, config.fixed_z, FIXED_ORIENTATION], goal_time=2.0, blocking=True)
        ros.start()
        pygame_input.start()
        next_tick = time.monotonic()
        while not quit_requested:
            events = pygame_input.poll()
            if events.quit_requested:
                pending_stop = recording
                quit_requested = not recording
            if events.discard_requested:
                recorder.start()
                recording = False
                pending_stop = False
            if events.save_toggle:
                if recording:
                    pending_stop = True
                else:
                    recorder = EpisodeRecorder(
                        output_h5=config.output_h5,
                        video_path=config.output_video_dir / f"ep_{episode_id:03d}.mp4",
                        episode_id=episode_id,
                    )
                    episode_id += 1
                    recording = True
                    recorder.start()
            if events.reset_requested and not recording:
                reset_delta = np.array(config.start_xy, dtype=np.float64) - target_xy
                reset_norm = float(np.linalg.norm(reset_delta))
                if reset_norm > RESET_STEP_M:
                    reset_delta *= RESET_STEP_M / reset_norm
                target_xy, _ = accepted_target(target_xy, reset_delta, (config.x_bounds, config.y_bounds))
                arm.send_cartesian(np.r_[target_xy, config.fixed_z, FIXED_ORIENTATION], goal_time=1.0, blocking=False)
            if events.focus_lost:
                if recording:
                    recorder.start()
                    recording = False
                pending_stop = False
                # Clear held motion and immediately command a fixed-target hold.
                arm.send_cartesian(np.r_[target_xy, config.fixed_z, FIXED_ORIENTATION])
            if pending_stop and recording:
                # Final zero row is collected at the next valid tick.
                pass

            now = time.monotonic()
            if now >= next_tick:
                next_tick += TICK_S
                held_action = pygame_input.action()
                if pending_stop and recording:
                    held_action = np.zeros(2, dtype=np.float32)
                target, actual = accepted_target(target_xy, held_action, (config.x_bounds, config.y_bounds))
                if recording:
                    snapshot = image_store.snapshot()
                    fresh = snapshot is not None and LatestImageStore.accept(snapshot, previous_sequence, time.monotonic_ns(), IMAGE_MAX_AGE_S)
                    if not fresh:
                        recorder.start()
                        recording = False
                        pending_stop = False
                    else:
                        previous_sequence = snapshot.sequence
                        pixels = config.transform_profile.apply(snapshot.rgb)
                        preview = pixels
                        pose = arm.pose()
                        row = {
                            "pixels": pixels,
                            "proprio": np.r_[pose[:2], arm.velocities()[:2]].astype(np.float32),
                            "state": np.r_[pose[:2], np.full(4, np.nan)].astype(np.float32),
                            "action": actual.astype(np.float32),
                            "image_timestamp_ns": snapshot.source_timestamp_ns,
                            "image_receipt_monotonic_ns": snapshot.receipt_monotonic_ns,
                            "command_monotonic_ns": time.monotonic_ns(),
                        }
                        recorder.send_and_commit(
                            row,
                            lambda: arm.send_cartesian(np.r_[target, config.fixed_z, FIXED_ORIENTATION]),
                        )
                        target_xy = target
                        if pending_stop:
                            recorder.finish(save=True)
                            recording = False
                            pending_stop = False
                            if events.quit_requested:
                                quit_requested = True
                elif np.linalg.norm(actual) > 0.0:
                    # Idle arrow motion positions the EE but is not logged.
                    arm.send_cartesian(np.r_[target, config.fixed_z, FIXED_ORIENTATION])
                    target_xy = target
                if not recording and quit_requested:
                    break
            pygame_input.render(preview, "recording" if recording else "idle")
        normal_exit = True
    finally:
        try:
            if recording:
                recorder.finish(save=False)
            if normal_exit:
                pygame_input.confirm_shutdown()
            arm.send_cartesian(np.r_[target_xy, config.safe_z, FIXED_ORIENTATION], goal_time=2.0, blocking=True)
            arm.home(home_positions)
        finally:
            pygame_input.close()
            ros.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
