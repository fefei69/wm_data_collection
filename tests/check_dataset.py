"""Offline QA gate for collected push-box HDF5 datasets (dataset_spec.md §8).

Not a pytest module — run it directly against one or more dataset files:

    uv run --no-project --with h5py --with numpy python tests/check_dataset.py \
        data/pushbox_keyboard_20260711_172610.h5

Checks the schema/writer contract (§4, §7) and every machine-checkable QA
item (§8): action-proprio alignment, 5 Hz cadence, image freshness, the state
NaN contract, terminal zero actions, and episode-length bounds. Exits 0 when
no episode FAILs (WARNs allowed; `--strict` promotes them), 1 otherwise.
Visual sanity and coverage review (§8.4-5) remain manual.

Constants are pinned to dataset_spec.md rather than imported from the
collector so this script needs only numpy + h5py (no cv2/ROS) and so the
dataset is judged against the spec, not against whatever the collector
currently does.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


TICK_S = 0.2  # §2: 5 Hz decision rate
TICK_TOLERANCE_S = 0.02  # §2: uniform to +/-10%
ACTION_CAP_M = 0.010  # §3: norm cap per step
CAP_EPSILON_M = 1e-6  # float32 storage slack on the cap
MAX_IMAGE_AGE_S = 0.20  # §2: freshness bound at command time
MEDIAN_LATENCY_WARN_S = 0.05  # §8.2: degraded-stream flag
MIN_ALIGNMENT_CORRELATION = 0.9  # §8.1, on contact-free segments
EPISODE_HARD_FLOOR = 50  # §5
EPISODE_TARGET_RANGE = (120, 150)  # §5
EPISODE_SOFT_CEILING = 250  # §5

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

    for name in ("ep_len", "ep_offset"):
        if name not in f:
            result.fail(f"missing writer bookkeeping dataset: {name}")
            return result, total
    ep_len = np.asarray(f["ep_len"], dtype=np.int64)
    ep_offset = np.asarray(f["ep_offset"], dtype=np.int64)
    if ep_len.shape != ep_offset.shape:
        result.fail(f"ep_len {ep_len.shape} and ep_offset {ep_offset.shape} disagree")
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
    if not np.isnan(correlation) and correlation < MIN_ALIGNMENT_CORRELATION:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="dataset .h5 file(s)")
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        if not path.is_file():
            print(f"FAIL {path}: file does not exist")
            exit_code = 1
            continue
        result, summary = check_file(path)
        print(summary)
        for message in result.failures:
            print(f"  FAIL {message}")
        for message in result.warnings:
            print(f"  WARN {message}")
        if not result.failures and not result.warnings:
            print("  all checks passed")
        if result.failures or (args.strict and result.warnings):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
