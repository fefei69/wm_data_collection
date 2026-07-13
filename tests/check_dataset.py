"""Offline QA gate for collected push-box HDF5 datasets (dataset_spec.md §8).

Not a pytest module — run it directly. With no paths it discovers every
``*.h5``/``*.hdf5`` file below ``data/``:

    uv run --no-project --with h5py --with numpy --with imageio-ffmpeg \
        python tests/check_dataset.py

Explicit paths remain available for targeted checks:

    uv run --no-project --with h5py --with numpy --with imageio-ffmpeg \
        python tests/check_dataset.py data/pushbox_keyboard_20260711_172610.h5

Checks the schema/writer contract (§4, §7) and every machine-checkable QA
item (§8): action-proprio alignment, 5 Hz cadence, image freshness, the state
NaN contract, terminal zero actions, episode-length bounds, action coverage,
pixel continuity, and matching preview videos. Exits 0 when no episode FAILs
(WARNs allowed; `--strict` promotes them), 1 otherwise. Visual semantics and
coverage quality still require operator review; this script surfaces the
statistics and likely visual discontinuities that make that review efficient.

Constants are pinned to dataset_spec.md rather than imported from the
collector so the dataset is judged against the spec, not against whatever the
collector currently does. Video checks are skipped with a warning when the
optional ``imageio-ffmpeg`` package is unavailable.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


TICK_S = 0.2  # §2: 5 Hz decision rate
DEFAULT_GOAL_HOURS = 4.0
TICK_TOLERANCE_S = 0.02  # §2: uniform to +/-10%
ACTION_CAP_M = 0.010  # §3: norm cap per step
CAP_EPSILON_M = 1e-6  # float32 storage slack on the cap
MAX_IMAGE_AGE_S = 0.20  # §2: freshness bound at command time
MEDIAN_LATENCY_WARN_S = 0.05  # §8.2: degraded-stream flag
MIN_ALIGNMENT_CORRELATION = 0.9  # §8.1, on contact-free segments
EPISODE_HARD_FLOOR = 50  # §5
EPISODE_TARGET_RANGE = (120, 150)  # §5
EPISODE_SOFT_CEILING = 250  # §5
VIDEO_FPS = 5.0
PIXEL_JUMP_REVIEW_MAD = 10.0
EXPECTED_SPEEDS_MM = (2.5, 5.0, 10.0)
EXPECTED_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+X+Y", "+X-Y", "-X+Y", "-X-Y")

EXPECTED_COLUMNS = {
    # §4/§7 writer contract: name -> (trailing shape, dtype)
    "pixels": ((224, 224, 3), np.uint8),
    "action": ((2,), np.float32),
    "proprio": ((4,), np.float32),
    "state": ((6,), np.float32),
    "episode_idx": ((), np.int64),
    "step_idx": ((), np.int64),
    "image_timestamp_ns": ((), np.int64),
    "image_receipt_monotonic_ns": ((), np.int64),
    "command_monotonic_ns": ((), np.int64),
}


@dataclass
class CheckResult:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: "CheckResult") -> None:
        self.failures.extend(other.failures)
        self.warnings.extend(other.warnings)


@dataclass
class EpisodeStats:
    index: int
    episode_id: int
    length: int
    zero_actions: int
    nonterminal_actions: int
    speed_counts: Counter[str]
    direction_counts: Counter[str]
    alignment: float
    latency_ms: np.ndarray
    achieved_ratios: np.ndarray
    pixel_mad: np.ndarray
    peak_pixel_transition: int | None
    brightness: np.ndarray
    exact_duplicate_pairs: int


@dataclass
class DatasetStats:
    path: Path
    episodes: list[EpisodeStats]
    total_rows: int
    command_interval_ms: np.ndarray
    source_interval_ms: np.ndarray
    latency_ms: np.ndarray
    speed_counts: Counter[str]
    direction_counts: Counter[str]
    zero_actions: int
    nonterminal_actions: int
    alignments: np.ndarray
    achieved_ratios: np.ndarray
    pixel_mad: np.ndarray
    brightness: np.ndarray
    exact_duplicate_pairs: int
    xy_min: np.ndarray
    xy_max: np.ndarray


@dataclass
class VideoStats:
    path: Path
    episode_id: int
    frames: int
    fps: float
    size: tuple[int, int]


def check_schema(f: h5py.File) -> tuple[CheckResult, int]:
    """Validate columns, dtypes, and the writer's ep_len/ep_offset bookkeeping."""
    result = CheckResult()
    lengths = {}
    for name, (trailing, dtype) in EXPECTED_COLUMNS.items():
        if name not in f:
            result.fail(f"missing column: {name}")
            continue
        dataset = f[name]
        if tuple(dataset.shape[1:]) != trailing:
            result.fail(f"{name}: shape {dataset.shape} does not end with {trailing}")
        if dataset.dtype != dtype:
            result.fail(f"{name}: dtype {dataset.dtype} is not {np.dtype(dtype)}")
        lengths[name] = dataset.shape[0]
    if len(set(lengths.values())) > 1:
        result.fail(f"columns disagree on row count: {lengths}")
    total = next(iter(lengths.values()), 0)

    bookkeeping_valid = True
    for name in ("ep_len", "ep_offset"):
        if name not in f:
            result.fail(f"missing writer bookkeeping dataset: {name}")
            bookkeeping_valid = False
            continue
        dataset = f[name]
        if dataset.ndim != 1:
            result.fail(f"{name}: bookkeeping dataset must be one-dimensional")
            bookkeeping_valid = False
        if not np.issubdtype(dataset.dtype, np.integer):
            result.fail(f"{name}: dtype {dataset.dtype} is not an integer dtype")
            bookkeeping_valid = False
    if not bookkeeping_valid:
        return result, total

    ep_len = np.asarray(f["ep_len"], dtype=np.int64)
    ep_offset = np.asarray(f["ep_offset"], dtype=np.int64)
    if ep_len.shape != ep_offset.shape:
        result.fail(f"ep_len {ep_len.shape} and ep_offset {ep_offset.shape} disagree")
        return result, total
    if np.any(ep_len <= 0):
        result.fail("ep_len values must all be positive")
    if np.any(ep_offset < 0):
        result.fail("ep_offset values must all be nonnegative")
    if ep_len.size and int(ep_len.sum()) != total:
        result.fail(f"sum(ep_len)={int(ep_len.sum())} does not cover the {total} rows")
    expected_offsets = np.concatenate(([0], np.cumsum(ep_len)[:-1])) if ep_len.size else ep_offset
    if not np.array_equal(ep_offset, expected_offsets):
        result.fail("ep_offset is not the running sum of ep_len (episodes overlap or gap)")
    return result, total


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def alignment_correlation(proprio: np.ndarray, action: np.ndarray, shift: int = 0) -> float:
    """§8.1: correlate achieved EE displacement with action[t + shift], both axes pooled."""
    displacement = np.diff(proprio[:, :2], axis=0)  # rows t -> t+1, length T-1
    commanded = action[:-1] if shift == 0 else action[1:] if shift == 1 else action[:-2]
    if shift == -1:
        displacement = displacement[1:]
    elif shift == 1:
        displacement = displacement[: commanded.shape[0]]
    return _correlation(displacement.reshape(-1), commanded.reshape(-1))


def check_episode(
    index: int,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    action: np.ndarray,
    proprio: np.ndarray,
    state: np.ndarray,
    pixels_sample: np.ndarray,
    image_timestamp_ns: np.ndarray,
    image_receipt_ns: np.ndarray,
    command_ns: np.ndarray,
) -> CheckResult:
    result = CheckResult()
    prefix = f"episode {index} (idx {int(episode_idx[0])}, {len(step_idx)} steps)"

    # Structure (§4, §5)
    if not np.all(episode_idx == episode_idx[0]):
        result.fail(f"{prefix}: episode_idx is not constant within the episode")
    if not np.array_equal(step_idx, np.arange(len(step_idx))):
        result.fail(f"{prefix}: step_idx is not 0..T-1")
    length = len(step_idx)
    if length < EPISODE_HARD_FLOOR:
        result.fail(f"{prefix}: length {length} is below the hard floor of {EPISODE_HARD_FLOOR}")
    elif not (EPISODE_TARGET_RANGE[0] <= length <= EPISODE_TARGET_RANGE[1]):
        if length > EPISODE_SOFT_CEILING:
            result.warn(f"{prefix}: length {length} exceeds the soft ceiling of {EPISODE_SOFT_CEILING}")
        else:
            result.warn(
                f"{prefix}: length {length} is outside the {EPISODE_TARGET_RANGE[0]}-"
                f"{EPISODE_TARGET_RANGE[1]} target window"
            )

    # Actions (§3)
    norms = np.linalg.norm(action, axis=1)
    if float(norms.max(initial=0.0)) > ACTION_CAP_M + CAP_EPSILON_M:
        result.fail(f"{prefix}: max action norm {norms.max():.4f} m exceeds the {ACTION_CAP_M} m cap")
    if not np.allclose(action[-1], 0.0):
        result.fail(f"{prefix}: final action row is not zero padding")
    if not np.all(np.isfinite(action)):
        result.fail(f"{prefix}: action contains non-finite values")

    # State contract (§4, §8.3)
    if not np.all(np.isnan(state[:, 2:])):
        result.fail(f"{prefix}: state[:, 2:] must be all-NaN (box pose is unmeasured)")
    if not np.all(np.isfinite(state[:, :2])):
        result.fail(f"{prefix}: state EE X/Y contains non-finite values")
    if not np.all(np.isfinite(proprio)):
        result.fail(f"{prefix}: proprio contains non-finite values")
    if not np.allclose(state[:, :2], proprio[:, :2], atol=1e-6):
        result.fail(f"{prefix}: state[:, :2] does not match proprio[:, :2]")

    # Timing (§2, §8.2)
    tick_s = np.diff(command_ns) / 1e9
    bad_ticks = int(np.count_nonzero(np.abs(tick_s - TICK_S) > TICK_TOLERANCE_S))
    if bad_ticks:
        result.fail(
            f"{prefix}: {bad_ticks} tick(s) outside {TICK_S:.2f}+/-{TICK_TOLERANCE_S:.2f} s"
            f" (worst {tick_s.max():.3f} s / {tick_s.min():.3f} s)"
        )
    if not np.all(np.diff(image_timestamp_ns) > 0):
        result.fail(f"{prefix}: image source stamps are not strictly increasing")
    if not np.all(np.diff(image_receipt_ns) > 0):
        result.fail(f"{prefix}: image receipt times are not strictly increasing (frame reuse?)")
    latency_s = (command_ns - image_receipt_ns) / 1e9
    if float(latency_s.min(initial=0.0)) < 0.0:
        result.fail(f"{prefix}: command precedes image receipt (min latency {latency_s.min():.3f} s)")
    if float(latency_s.max(initial=0.0)) > MAX_IMAGE_AGE_S:
        result.fail(
            f"{prefix}: image-to-command latency up to {latency_s.max():.3f} s exceeds"
            f" the {MAX_IMAGE_AGE_S:.2f} s freshness bound"
        )
    median_latency = float(np.median(latency_s))
    if median_latency > MEDIAN_LATENCY_WARN_S:
        result.warn(
            f"{prefix}: median image-to-command latency {median_latency * 1000:.0f} ms exceeds"
            f" {MEDIAN_LATENCY_WARN_S * 1000:.0f} ms (degraded stream)"
        )

    # Alignment (§8.1) — the spec threshold applies to contact-free segments,
    # which are not labeled, so pooled correlation only warns.
    correlation = alignment_correlation(proprio, action)
    commanded_motion = bool(
        np.any(np.linalg.norm(action[:-1], axis=1) > CAP_EPSILON_M)
    )
    if np.isnan(correlation):
        if commanded_motion:
            result.fail(
                f"{prefix}: displacement/action correlation is undefined despite"
                " nonzero commands"
            )
    elif correlation < MIN_ALIGNMENT_CORRELATION:
        hint = ""
        shifted = {s: alignment_correlation(proprio, action, shift=s) for s in (-1, 1)}
        best_shift = max(shifted, key=lambda s: np.nan_to_num(shifted[s], nan=-2.0))
        if np.nan_to_num(shifted[best_shift], nan=-2.0) > correlation + 0.1:
            hint = (
                f"; correlates better with action[t{best_shift:+d}]"
                f" (r={shifted[best_shift]:.2f}) — logging may be off by one tick"
            )
        result.warn(
            f"{prefix}: displacement/action correlation r={correlation:.2f} is below"
            f" {MIN_ALIGNMENT_CORRELATION}{hint}"
        )

    # Pixels (§8.4 is manual; this only catches dead frames)
    if pixels_sample.size and int(pixels_sample.max()) == int(pixels_sample.min()):
        result.fail(f"{prefix}: sampled frames are constant-valued (dead camera?)")

    return result


def check_file(path: Path) -> tuple[CheckResult, str]:
    result = CheckResult()
    with h5py.File(path, "r") as f:
        schema_result, total = check_schema(f)
        result.merge(schema_result)
        if schema_result.failures:
            return result, f"{path}: schema invalid; skipped episode checks"

        ep_len = np.asarray(f["ep_len"], dtype=np.int64)
        ep_offset = np.asarray(f["ep_offset"], dtype=np.int64)
        episode_ids = []
        for index, (offset, length) in enumerate(zip(ep_offset, ep_len)):
            rows = slice(int(offset), int(offset + length))
            # Three sampled frames per episode keep pixel I/O trivial.
            frame_rows = sorted({rows.start, rows.start + int(length) // 2, rows.stop - 1})
            result.merge(
                check_episode(
                    index,
                    episode_idx=np.asarray(f["episode_idx"][rows]),
                    step_idx=np.asarray(f["step_idx"][rows]),
                    action=np.asarray(f["action"][rows]),
                    proprio=np.asarray(f["proprio"][rows]),
                    state=np.asarray(f["state"][rows]),
                    pixels_sample=np.stack([f["pixels"][row] for row in frame_rows]),
                    image_timestamp_ns=np.asarray(f["image_timestamp_ns"][rows]),
                    image_receipt_ns=np.asarray(f["image_receipt_monotonic_ns"][rows]),
                    command_ns=np.asarray(f["command_monotonic_ns"][rows]),
                )
            )
            episode_ids.append(int(f["episode_idx"][rows.start]))
        if episode_ids != sorted(set(episode_ids)):
            result.fail(f"{path}: episode_idx is not strictly increasing across the file")

        minutes = total * TICK_S / 60.0
        summary = (
            f"{path}: {len(ep_len)} episodes, {total} steps"
            f" ({minutes:.1f} min of robot time at 5 Hz)"
        )
    return result, summary


def discover_dataset_paths(paths: Sequence[Path], data_dir: Path) -> list[Path]:
    """Return explicit paths, or every HDF5 dataset below ``data_dir``."""
    if paths:
        return list(paths)
    discovered = {*data_dir.rglob("*.h5"), *data_dir.rglob("*.hdf5")}
    return sorted(discovered)


def action_direction(action: np.ndarray) -> str:
    """Map a planar action to zero, a cardinal, or a diagonal label."""
    x, y = np.asarray(action, dtype=np.float64).reshape(2)
    if float(np.linalg.norm((x, y))) <= CAP_EPSILON_M:
        return "zero"
    x_label = "+X" if x > CAP_EPSILON_M else "-X" if x < -CAP_EPSILON_M else ""
    y_label = "+Y" if y > CAP_EPSILON_M else "-Y" if y < -CAP_EPSILON_M else ""
    return x_label + y_label


def action_speed_bucket(action: np.ndarray) -> str:
    """Bucket a moving action by the commissioned magnitudes, in millimetres."""
    norm_mm = float(np.linalg.norm(action)) * 1000.0
    for expected in EXPECTED_SPEEDS_MM:
        if abs(norm_mm - expected) <= 0.05:
            return f"{expected:g}"
    return "other"


def _concatenate(arrays: Sequence[np.ndarray]) -> np.ndarray:
    populated = [np.asarray(array) for array in arrays if np.asarray(array).size]
    return np.concatenate(populated) if populated else np.empty(0, dtype=np.float64)


def collect_dataset_stats(path: Path, *, scan_pixels: bool = True) -> DatasetStats:
    """Collect detailed timing, action, tracking, and optional pixel statistics."""
    episodes: list[EpisodeStats] = []
    command_intervals: list[np.ndarray] = []
    source_intervals: list[np.ndarray] = []
    all_latency: list[np.ndarray] = []
    all_ratios: list[np.ndarray] = []
    all_pixel_mad: list[np.ndarray] = []
    all_brightness: list[np.ndarray] = []
    speed_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    zero_actions = 0
    nonterminal_actions = 0
    exact_duplicate_pairs = 0

    with h5py.File(path, "r") as f:
        ep_len = np.asarray(f["ep_len"], dtype=np.int64)
        ep_offset = np.asarray(f["ep_offset"], dtype=np.int64)
        all_proprio = np.asarray(f["proprio"], dtype=np.float64)

        for index, (offset_value, length_value) in enumerate(zip(ep_offset, ep_len)):
            offset = int(offset_value)
            length = int(length_value)
            rows = slice(offset, offset + length)
            action = np.asarray(f["action"][rows], dtype=np.float64)
            proprio = all_proprio[rows]
            command_ns = np.asarray(f["command_monotonic_ns"][rows], dtype=np.int64)
            source_ns = np.asarray(f["image_timestamp_ns"][rows], dtype=np.int64)
            receipt_ns = np.asarray(f["image_receipt_monotonic_ns"][rows], dtype=np.int64)

            command_intervals.append(np.diff(command_ns) / 1e6)
            source_intervals.append(np.diff(source_ns) / 1e6)
            latency_ms = (command_ns - receipt_ns) / 1e6
            all_latency.append(latency_ms)

            moving_action = action[:-1]
            moving_norm = np.linalg.norm(moving_action, axis=1)
            zero_mask = moving_norm <= CAP_EPSILON_M
            episode_zero = int(np.count_nonzero(zero_mask))
            zero_actions += episode_zero
            nonterminal_actions += len(moving_action)

            episode_speeds: Counter[str] = Counter()
            episode_directions: Counter[str] = Counter()
            for value, is_zero in zip(moving_action, zero_mask):
                if is_zero:
                    continue
                episode_speeds[action_speed_bucket(value)] += 1
                episode_directions[action_direction(value)] += 1
            speed_counts.update(episode_speeds)
            direction_counts.update(episode_directions)

            displacement = np.diff(proprio[:, :2], axis=0)
            alignment = _correlation(displacement.reshape(-1), moving_action.reshape(-1))
            active = ~zero_mask
            achieved_ratios = (
                np.linalg.norm(displacement[active], axis=1) / moving_norm[active]
                if np.any(active)
                else np.empty(0, dtype=np.float64)
            )
            all_ratios.append(achieved_ratios)

            pixel_mad_values: list[float] = []
            brightness_values: list[float] = []
            episode_duplicates = 0
            peak_transition: int | None = None
            if scan_pixels:
                previous: np.ndarray | None = None
                for row in range(offset, offset + length):
                    frame = np.asarray(f["pixels"][row])
                    brightness_values.append(float(frame.mean()))
                    if previous is not None:
                        if np.array_equal(frame, previous):
                            episode_duplicates += 1
                        pixel_mad_values.append(
                            float(
                                np.mean(
                                    np.abs(
                                        frame.astype(np.int16)
                                        - previous.astype(np.int16)
                                    )
                                )
                            )
                        )
                    previous = frame
                if pixel_mad_values:
                    peak_transition = int(np.argmax(pixel_mad_values)) + 1

            pixel_mad = np.asarray(pixel_mad_values, dtype=np.float64)
            brightness = np.asarray(brightness_values, dtype=np.float64)
            all_pixel_mad.append(pixel_mad)
            all_brightness.append(brightness)
            exact_duplicate_pairs += episode_duplicates

            episodes.append(
                EpisodeStats(
                    index=index,
                    episode_id=int(f["episode_idx"][offset]),
                    length=length,
                    zero_actions=episode_zero,
                    nonterminal_actions=len(moving_action),
                    speed_counts=episode_speeds,
                    direction_counts=episode_directions,
                    alignment=alignment,
                    latency_ms=latency_ms,
                    achieved_ratios=achieved_ratios,
                    pixel_mad=pixel_mad,
                    peak_pixel_transition=peak_transition,
                    brightness=brightness,
                    exact_duplicate_pairs=episode_duplicates,
                )
            )

    return DatasetStats(
        path=path,
        episodes=episodes,
        total_rows=sum(episode.length for episode in episodes),
        command_interval_ms=_concatenate(command_intervals),
        source_interval_ms=_concatenate(source_intervals),
        latency_ms=_concatenate(all_latency),
        speed_counts=speed_counts,
        direction_counts=direction_counts,
        zero_actions=zero_actions,
        nonterminal_actions=nonterminal_actions,
        alignments=np.asarray([episode.alignment for episode in episodes]),
        achieved_ratios=_concatenate(all_ratios),
        pixel_mad=_concatenate(all_pixel_mad),
        brightness=_concatenate(all_brightness),
        exact_duplicate_pairs=exact_duplicate_pairs,
        xy_min=np.min(all_proprio[:, :2], axis=0),
        xy_max=np.max(all_proprio[:, :2], axis=0),
    )


def stats_review_warnings(stats: DatasetStats) -> CheckResult:
    """Flag likely visual discontinuities for manual video review."""
    result = CheckResult()
    for episode in stats.episodes:
        if not episode.pixel_mad.size:
            continue
        peak = float(np.max(episode.pixel_mad))
        if peak > PIXEL_JUMP_REVIEW_MAD:
            transition = episode.peak_pixel_transition
            result.warn(
                f"episode {episode.index} (idx {episode.episode_id}): adjacent-frame "
                f"pixel MAD peaks at {peak:.2f} near step {transition - 1}->{transition}; "
                "review for a person, reset, lighting change, or camera discontinuity"
            )
    return result


def _video_directory(dataset_path: Path, video_root: Path) -> Path | None:
    timestamp = re.search(r"(\d{8}_\d{6})$", dataset_path.stem)
    return video_root / timestamp.group(1) if timestamp else None


def check_matching_videos(
    stats: DatasetStats,
    video_root: Path,
) -> tuple[CheckResult, list[VideoStats]]:
    """Decode matching preview videos and compare their metadata to HDF5 episodes."""
    result = CheckResult()
    video_stats: list[VideoStats] = []
    directory = _video_directory(stats.path, video_root)
    if directory is None:
        result.warn(f"{stats.path}: cannot infer timestamped preview-video directory")
        return result, video_stats
    if not directory.is_dir():
        result.warn(f"{stats.path}: matching preview-video directory is missing: {directory}")
        return result, video_stats

    try:
        import imageio_ffmpeg
    except ImportError:
        result.warn(
            "preview-video checks skipped: install imageio-ffmpeg or run with "
            "--with imageio-ffmpeg"
        )
        return result, video_stats

    expected = {episode.episode_id: episode.length for episode in stats.episodes}
    discovered: dict[int, Path] = {}
    for path in sorted(directory.glob("ep_*.mp4")):
        match = re.fullmatch(r"ep_(\d+)\.mp4", path.name)
        if match:
            discovered[int(match.group(1))] = path

    for episode_id in sorted(expected.keys() - discovered.keys()):
        result.warn(f"{stats.path}: preview video missing for episode id {episode_id}")
    for episode_id in sorted(discovered.keys() - expected.keys()):
        result.warn(
            f"{stats.path}: preview video has no matching HDF5 episode: "
            f"{discovered[episode_id]}"
        )

    for episode_id in sorted(expected.keys() & discovered.keys()):
        path = discovered[episode_id]
        reader: Any = None
        try:
            reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
            metadata = next(reader)
            frames = sum(1 for _ in reader)
            size_value = metadata.get("size") or metadata.get("source_size") or (0, 0)
            size = (int(size_value[0]), int(size_value[1]))
            fps = float(metadata.get("fps", 0.0))
        except Exception as exc:
            result.fail(f"{path}: video decode failed: {exc}")
            continue
        finally:
            if reader is not None:
                reader.close()

        video_stats.append(
            VideoStats(
                path=path,
                episode_id=episode_id,
                frames=frames,
                fps=fps,
                size=size,
            )
        )
        if frames != expected[episode_id]:
            result.fail(
                f"{path}: {frames} frames does not match HDF5 episode length "
                f"{expected[episode_id]}"
            )
        if size != (224, 224):
            result.fail(f"{path}: video size {size} is not (224, 224)")
        if abs(fps - VIDEO_FPS) > 0.01:
            result.fail(f"{path}: video frame rate {fps:.3f} is not {VIDEO_FPS:.1f} fps")

    return result, video_stats


def combine_dataset_stats(items: Sequence[DatasetStats]) -> DatasetStats:
    """Combine file summaries without merging or rewriting their HDF5 data."""
    return DatasetStats(
        path=Path("ALL DATASETS"),
        episodes=[episode for item in items for episode in item.episodes],
        total_rows=sum(item.total_rows for item in items),
        command_interval_ms=_concatenate([item.command_interval_ms for item in items]),
        source_interval_ms=_concatenate([item.source_interval_ms for item in items]),
        latency_ms=_concatenate([item.latency_ms for item in items]),
        speed_counts=sum((item.speed_counts for item in items), Counter()),
        direction_counts=sum((item.direction_counts for item in items), Counter()),
        zero_actions=sum(item.zero_actions for item in items),
        nonterminal_actions=sum(item.nonterminal_actions for item in items),
        alignments=_concatenate([item.alignments for item in items]),
        achieved_ratios=_concatenate([item.achieved_ratios for item in items]),
        pixel_mad=_concatenate([item.pixel_mad for item in items]),
        brightness=_concatenate([item.brightness for item in items]),
        exact_duplicate_pairs=sum(item.exact_duplicate_pairs for item in items),
        xy_min=np.min(np.stack([item.xy_min for item in items]), axis=0),
        xy_max=np.max(np.stack([item.xy_max for item in items]), axis=0),
    )


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _counter_text(counter: Counter[str], order: Sequence[str]) -> str:
    ordered = [f"{key}={counter[key]}" for key in order if counter[key]]
    extras = sorted(set(counter) - set(order))
    ordered.extend(f"{key}={counter[key]}" for key in extras)
    return ", ".join(ordered) if ordered else "none"


def format_duration(seconds: float) -> str:
    """Format a duration for the collection-goal summary."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def print_collection_goal(total_rows: int, goal_hours: float) -> None:
    """Put overall nominal dataset time and goal progress at the end of the report."""
    collected_seconds = total_rows * TICK_S
    goal_seconds = goal_hours * 3600.0
    progress_percent = collected_seconds / goal_seconds * 100.0
    remaining_seconds = max(0.0, goal_seconds - collected_seconds)
    target_rows = math.ceil(goal_seconds / TICK_S)
    remaining_rows = max(0, target_rows - total_rows)

    print(f"\n{goal_hours:g}-HOUR DATA GOAL")
    print(
        f"  Collected: {format_duration(collected_seconds)} of "
        f"{format_duration(goal_seconds)} ({progress_percent:.1f}%)"
    )
    if remaining_seconds:
        print(
            f"  Remaining: {format_duration(remaining_seconds)} "
            f"({remaining_rows:,} samples at 5 Hz)"
        )
    else:
        exceeded_seconds = collected_seconds - goal_seconds
        print(f"  Goal reached; exceeded by {format_duration(exceeded_seconds)}")


def print_dataset_stats(stats: DatasetStats, *, show_episodes: bool = True) -> None:
    """Print a compact, human-readable statistical report."""
    moving = stats.nonterminal_actions - stats.zero_actions
    zero_fraction = stats.zero_actions / stats.nonterminal_actions if stats.nonterminal_actions else 0.0
    duration_minutes = stats.total_rows * TICK_S / 60.0
    print(
        f"  STATS {len(stats.episodes)} episodes, {stats.total_rows} rows, "
        f"{duration_minutes:.2f} min at 5 Hz"
    )
    if stats.command_interval_ms.size:
        values = stats.command_interval_ms
        print(
            "    command interval: "
            f"mean={np.mean(values):.3f} ms std={np.std(values):.3f} "
            f"min={np.min(values):.3f} p95={np.percentile(values, 95):.3f} "
            f"max={np.max(values):.3f}"
        )
    if stats.source_interval_ms.size:
        outside = int(
            np.count_nonzero(
                np.abs(stats.source_interval_ms - TICK_S * 1000.0)
                > TICK_TOLERANCE_S * 1000.0
            )
        )
        print(
            "    selected-image interval: "
            f"mean={np.mean(stats.source_interval_ms):.3f} ms "
            f"min={np.min(stats.source_interval_ms):.3f} "
            f"max={np.max(stats.source_interval_ms):.3f}; "
            f"outside 200+/-20 ms={outside}/{stats.source_interval_ms.size}"
        )
    if stats.latency_ms.size:
        print(
            "    image latency: "
            f"median={np.median(stats.latency_ms):.3f} ms "
            f"p95={np.percentile(stats.latency_ms, 95):.3f} "
            f"max={np.max(stats.latency_ms):.3f}"
        )
    print(
        f"    actions: zero={stats.zero_actions}/{stats.nonterminal_actions} "
        f"({zero_fraction * 100:.1f}%), moving={moving}"
    )
    print(
        "    moving speeds (mm): "
        + _counter_text(stats.speed_counts, tuple(f"{value:g}" for value in EXPECTED_SPEEDS_MM))
    )
    print(
        "    moving directions: "
        + _counter_text(stats.direction_counts, EXPECTED_DIRECTIONS)
    )
    print(
        f"    measured XY range: x=[{stats.xy_min[0]:.4f}, {stats.xy_max[0]:.4f}] m "
        f"y=[{stats.xy_min[1]:.4f}, {stats.xy_max[1]:.4f}] m"
    )
    alignments = _finite(stats.alignments)
    if alignments.size:
        print(
            "    alignment r: "
            f"min={np.min(alignments):.3f} median={np.median(alignments):.3f} "
            f"max={np.max(alignments):.3f}"
        )
    if stats.achieved_ratios.size:
        print(
            "    achieved/commanded displacement: "
            f"p05={np.percentile(stats.achieved_ratios, 5):.3f} "
            f"median={np.median(stats.achieved_ratios):.3f} "
            f"p95={np.percentile(stats.achieved_ratios, 95):.3f}"
        )
    if stats.pixel_mad.size:
        print(
            "    adjacent pixel MAD: "
            f"p05={np.percentile(stats.pixel_mad, 5):.3f} "
            f"median={np.median(stats.pixel_mad):.3f} "
            f"p95={np.percentile(stats.pixel_mad, 95):.3f} "
            f"max={np.max(stats.pixel_mad):.3f}; "
            f"exact duplicates={stats.exact_duplicate_pairs}"
        )
    if stats.brightness.size:
        print(
            "    frame brightness: "
            f"mean={np.mean(stats.brightness):.2f} "
            f"std-over-frames={np.std(stats.brightness):.2f} "
            f"min={np.min(stats.brightness):.2f} max={np.max(stats.brightness):.2f}"
        )
    if show_episodes:
        print("    per episode:")
        for episode in stats.episodes:
            zero_percent = (
                100.0 * episode.zero_actions / episode.nonterminal_actions
                if episode.nonterminal_actions
                else 0.0
            )
            peak = float(np.max(episode.pixel_mad)) if episode.pixel_mad.size else float("nan")
            print(
                f"      ep index={episode.index:02d} id={episode.episode_id:03d}: "
                f"{episode.length:3d} steps/{episode.length * TICK_S:4.1f} s, "
                f"zero={zero_percent:4.1f}%, alignment={episode.alignment:.3f}, "
                f"latency median/max={np.median(episode.latency_ms):.1f}/"
                f"{np.max(episode.latency_ms):.1f} ms, pixel-peak={peak:.2f}"
            )


def print_coverage_notes(stats: DatasetStats) -> None:
    """Print heuristic coverage flags without changing the QA exit status."""
    moving = stats.nonterminal_actions - stats.zero_actions
    notes: list[str] = []
    if stats.nonterminal_actions and stats.zero_actions / stats.nonterminal_actions > 0.25:
        notes.append("more than 25% of nonterminal actions are zero")
    if moving:
        dominant_speed, dominant_count = stats.speed_counts.most_common(1)[0]
        if dominant_count / moving > 0.60:
            notes.append(
                f"{dominant_speed} mm actions dominate {dominant_count / moving * 100:.1f}% "
                "of moving steps"
            )
        sparse = [
            direction
            for direction in EXPECTED_DIRECTIONS
            if stats.direction_counts[direction] / moving < 0.02
        ]
        if sparse:
            notes.append("directions below 2% of moving steps: " + ", ".join(sparse))
    if notes:
        print("  COVERAGE FLAGS (informational):")
        for note in notes:
            print(f"    - {note}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="dataset file(s); defaults to every .h5/.hdf5 below --data-dir",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory recursively scanned when no explicit paths are given",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("output_videos"),
        help="root containing timestamped preview-video directories",
    )
    parser.add_argument(
        "--no-video-check",
        action="store_true",
        help="skip preview-video discovery and full decode checks",
    )
    parser.add_argument(
        "--no-pixel-scan",
        action="store_true",
        help="skip the full pixel continuity/brightness scan for a faster report",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    parser.add_argument(
        "--goal-hours",
        type=float,
        default=DEFAULT_GOAL_HOURS,
        help=f"collection-time goal in hours (default: {DEFAULT_GOAL_HOURS:g})",
    )
    args = parser.parse_args(argv)
    if args.goal_hours <= 0:
        parser.error("--goal-hours must be greater than zero")

    paths = discover_dataset_paths(args.paths, args.data_dir)
    if not paths:
        print(f"FAIL no .h5/.hdf5 datasets found below {args.data_dir}")
        return 1

    print(f"Discovered {len(paths)} dataset(s)")
    total_result = CheckResult()
    collected_stats: list[DatasetStats] = []
    decoded_videos = 0
    for path in paths:
        print(f"\n{path}")
        if not path.is_file():
            message = f"{path}: file does not exist"
            print(f"  FAIL {message}")
            total_result.fail(message)
            continue

        try:
            result, summary = check_file(path)
        except (OSError, ValueError) as exc:
            message = f"{path}: could not read dataset: {exc}"
            print(f"  FAIL {message}")
            total_result.fail(message)
            continue

        stats: DatasetStats | None = None
        if "schema invalid" not in summary:
            try:
                stats = collect_dataset_stats(path, scan_pixels=not args.no_pixel_scan)
                result.merge(stats_review_warnings(stats))
                if not args.no_video_check:
                    video_result, videos = check_matching_videos(stats, args.video_root)
                    result.merge(video_result)
                    decoded_videos += len(videos)
            except (OSError, KeyError, ValueError) as exc:
                result.fail(f"{path}: statistics failed: {exc}")

        print(f"  {summary}")
        for message in result.failures:
            print(f"  FAIL {message}")
        for message in result.warnings:
            print(f"  WARN {message}")
        if not result.failures and not result.warnings:
            print("  all hard checks passed; no warnings")
        if stats is not None:
            print_dataset_stats(stats)
            collected_stats.append(stats)
        total_result.merge(result)

    if collected_stats:
        combined = combine_dataset_stats(collected_stats)
        print("\nCOMBINED")
        print_dataset_stats(combined, show_episodes=False)
        print_coverage_notes(combined)
    if not args.no_video_check:
        print(f"\nDecoded {decoded_videos} matching preview video(s)")

    if collected_stats:
        print_collection_goal(combined.total_rows, args.goal_hours)

    print(
        f"RESULT: {len(total_result.failures)} failure(s), "
        f"{len(total_result.warnings)} warning(s)"
    )
    return int(bool(total_result.failures or (args.strict and total_result.warnings)))


if __name__ == "__main__":
    raise SystemExit(main())
