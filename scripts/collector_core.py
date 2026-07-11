"""Dependency-light helpers for the ROS 2 keyboard collector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SOURCE_WIDTH = 640
SOURCE_HEIGHT = 480
MODEL_SIZE = 224


def action_from_held(held: Mapping[str, bool], magnitude_m: float) -> np.ndarray:
    """Convert held arrows into a normalized base-frame XY delta."""
    magnitude = float(magnitude_m)
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude_m must be finite and positive")

    direction = np.array(
        [
            int(bool(held.get("up", False))) - int(bool(held.get("down", False))),
            int(bool(held.get("left", False))) - int(bool(held.get("right", False))),
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.zeros(2, dtype=np.float32)
    return (direction * (magnitude / norm)).astype(np.float32)


def validate_magnitudes(magnitudes: Mapping[int, float], cap_m: float) -> None:
    """Validate the three ordered speed levels used by the collector."""
    if set(magnitudes) != {1, 2, 3}:
        raise ValueError("magnitudes must contain exactly speed levels 1, 2, and 3")
    cap = float(cap_m)
    values = [float(magnitudes[level]) for level in (1, 2, 3)]
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap_m must be finite and positive")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("all magnitudes must be finite and positive")
    if values != sorted(values) or len(set(values)) != 3:
        raise ValueError("magnitudes must be strictly increasing")
    if values[-1] > cap:
        raise ValueError("the largest magnitude cannot exceed cap_m")


@dataclass(frozen=True)
class ImageTransformProfile:
    """A fixed, non-distorting 4:3-to-square image transform."""

    mode: str
    output_size: int = MODEL_SIZE
    fill_rgb: tuple[int, int, int] = (0, 0, 0)
    x: int | None = None
    y: int | None = None
    size: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageTransformProfile":
        mode = str(data.get("mode", "")).lower()
        if mode not in {"letterbox", "square_roi"}:
            raise ValueError("mode must be 'letterbox' or 'square_roi'")

        output_key = data.get("size", MODEL_SIZE) if mode == "letterbox" else data.get("output_size", MODEL_SIZE)
        output_size = int(output_key)
        if output_size <= 0:
            raise ValueError("output_size must be positive")

        fill = tuple(int(value) for value in data.get("fill_rgb", (0, 0, 0)))
        if len(fill) != 3 or any(value < 0 or value > 255 for value in fill):
            raise ValueError("fill_rgb must contain three uint8 values")

        if mode == "letterbox":
            return cls(mode=mode, output_size=output_size, fill_rgb=fill)

        required = ("x", "y", "size")
        if any(key not in data for key in required):
            raise ValueError("square_roi requires x, y, and size")
        x, y, size = (int(data[key]) for key in required)
        if x < 0 or y < 0 or size <= 0:
            raise ValueError("square_roi values must be non-negative with positive size")
        if x + size > SOURCE_WIDTH or y + size > SOURCE_HEIGHT:
            raise ValueError("square_roi must fit inside the 640x480 source")
        return cls(
            mode=mode,
            output_size=output_size,
            fill_rgb=fill,
            x=x,
            y=y,
            size=size,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply the configured transform to one RGB uint8 source image."""
        image = np.asarray(rgb)
        if image.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3) or image.dtype != np.uint8:
            raise ValueError("expected an RGB uint8 image with shape (480, 640, 3)")

        if self.mode == "square_roi":
            assert self.x is not None and self.y is not None and self.size is not None
            source = image[self.y : self.y + self.size, self.x : self.x + self.size]
            return cv2.resize(
                source,
                (self.output_size, self.output_size),
                interpolation=cv2.INTER_AREA,
            ).astype(np.uint8, copy=False)

        scale = min(self.output_size / SOURCE_WIDTH, self.output_size / SOURCE_HEIGHT)
        width = max(1, round(SOURCE_WIDTH * scale))
        height = max(1, round(SOURCE_HEIGHT * scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        output = np.empty(
            (self.output_size, self.output_size, 3), dtype=np.uint8
        )
        output[...] = np.asarray(self.fill_rgb, dtype=np.uint8)
        x0 = (self.output_size - width) // 2
        y0 = (self.output_size - height) // 2
        output[y0 : y0 + height, x0 : x0 + width] = resized
        return output


def _stack_rows(rows: Sequence[Mapping[str, Any]], key: str, dtype: np.dtype) -> np.ndarray:
    try:
        values = [np.asarray(row[key], dtype=dtype) for row in rows]
    except KeyError as exc:
        raise ValueError(f"missing row column: {exc.args[0]}") from exc
    return np.stack(values, axis=0)


def build_episode_columns(
    rows: Sequence[Mapping[str, Any]], episode_id: int
) -> dict[str, np.ndarray]:
    """Convert accepted per-tick rows to equal-length HDF5Writer columns."""
    if not rows:
        raise ValueError("cannot write an empty episode")

    pixels = _stack_rows(rows, "pixels", np.uint8)
    action = _stack_rows(rows, "action", np.float32)
    proprio = _stack_rows(rows, "proprio", np.float32)
    if pixels.shape[1:] != (MODEL_SIZE, MODEL_SIZE, 3):
        raise ValueError("pixels must have shape (T, 224, 224, 3)")
    if action.shape[1:] != (2,):
        raise ValueError("action must have shape (T, 2)")
    if proprio.shape[1:] != (4,):
        raise ValueError("proprio must have shape (T, 4)")
    if not np.allclose(action[-1], 0.0):
        raise ValueError("the final episode action must be zero padding")

    state_values = [
        np.asarray(row.get("state", [row["proprio"][0], row["proprio"][1], np.nan, np.nan, np.nan, np.nan]), dtype=np.float32)
        for row in rows
    ]
    state = np.stack(state_values, axis=0)
    if state.shape[1:] != (6,):
        raise ValueError("state must have shape (T, 6)")

    image_timestamp_ns = _stack_rows(rows, "image_timestamp_ns", np.int64).reshape(-1)
    image_receipt_ns = _stack_rows(rows, "image_receipt_monotonic_ns", np.int64).reshape(-1)
    command_ns = _stack_rows(rows, "command_monotonic_ns", np.int64).reshape(-1)
    length = len(rows)
    return {
        "pixels": pixels,
        "action": action,
        "proprio": proprio,
        "state": state,
        "episode_idx": np.full(length, int(episode_id), dtype=np.int64),
        "step_idx": np.arange(length, dtype=np.int64),
        "image_timestamp_ns": image_timestamp_ns,
        "image_receipt_monotonic_ns": image_receipt_ns,
        "command_monotonic_ns": command_ns,
    }
