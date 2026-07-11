from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ros_camera import LatestImageStore, decode_rgb8, evaluate_stream_health


def _image_message(height=480, width=640, step=None, encoding="rgb8", pixels=None):
    step = width * 3 if step is None else step
    if pixels is None:
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
    rows = np.zeros((height, step), dtype=np.uint8)
    if step >= width * 3:
        rows[:, : width * 3] = pixels.reshape(height, width * 3)
    return SimpleNamespace(
        height=height, width=width, step=step, encoding=encoding, data=rows.tobytes()
    )


def test_decode_rgb8_round_trips_pixels_and_honors_row_padding():
    pixels = np.arange(480 * 640 * 3, dtype=np.uint64).astype(np.uint8).reshape(480, 640, 3)
    np.testing.assert_array_equal(decode_rgb8(_image_message(pixels=pixels)), pixels)
    padded = _image_message(step=640 * 3 + 64, pixels=pixels)
    np.testing.assert_array_equal(decode_rgb8(padded), pixels)


def test_decode_rgb8_rejects_wrong_encoding_or_payload():
    with pytest.raises(ValueError, match="rgb8"):
        decode_rgb8(_image_message(encoding="bgr8"))
    truncated = _image_message()
    truncated.data = truncated.data[:-1]
    with pytest.raises(ValueError, match="payload"):
        decode_rgb8(truncated)
    with pytest.raises(ValueError, match="step"):
        decode_rgb8(_image_message(step=100))


def test_latest_image_store_replaces_frame_and_rejects_reuse():
    store = LatestImageStore()
    first = np.zeros((480, 640, 3), dtype=np.uint8)
    second = np.ones((480, 640, 3), dtype=np.uint8)
    store.update(first, source_timestamp_ns=10, receipt_monotonic_ns=100)
    store.update(second, source_timestamp_ns=20, receipt_monotonic_ns=200)
    snapshot = store.snapshot()
    assert snapshot is not None
    assert snapshot.sequence == 2
    assert snapshot.source_timestamp_ns == 20
    np.testing.assert_array_equal(snapshot.rgb, second)
    assert store.accept(snapshot, previous_sequence=1, now_monotonic_ns=250, max_age_s=0.1)
    assert not store.accept(snapshot, previous_sequence=2, now_monotonic_ns=250, max_age_s=0.1)


def test_latest_image_store_rejects_stale_or_future_receipt_time():
    store = LatestImageStore()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    store.update(image, source_timestamp_ns=10, receipt_monotonic_ns=100)
    snapshot = store.snapshot()
    assert snapshot is not None
    assert not store.accept(snapshot, previous_sequence=0, now_monotonic_ns=201_000_001, max_age_s=0.1)
    assert not store.accept(snapshot, previous_sequence=0, now_monotonic_ns=50, max_age_s=0.1)


def _receipts_ns(gaps_s):
    times = [0.0]
    for gap in gaps_s:
        times.append(times[-1] + gap)
    return [int(t * 1e9) for t in times]


def test_stream_health_accepts_a_steady_stream():
    report = evaluate_stream_health(_receipts_ns([1 / 60] * 600), duration_s=10.0)
    assert report.healthy
    assert report.fps == pytest.approx(60.1, abs=0.5)
    assert report.stale_fraction == 0.0


def test_stream_health_rejects_large_gaps_and_stale_time():
    gaps = [1 / 60] * 300 + [0.350] + [1 / 60] * 300
    report = evaluate_stream_health(_receipts_ns(gaps), duration_s=10.35)
    assert not report.healthy
    assert any("gap" in problem for problem in report.problems)

    gaps = ([1 / 60] * 50 + [0.25]) * 20  # repeated 250 ms dropouts
    report = evaluate_stream_health(_receipts_ns(gaps), duration_s=22.0)
    assert not report.healthy
    assert any("older than" in problem for problem in report.problems)


def test_stream_health_rejects_an_empty_or_missing_stream():
    report = evaluate_stream_health([], duration_s=10.0)
    assert not report.healthy
    report = evaluate_stream_health(_receipts_ns([1 / 60] * 60), duration_s=10.0)
    assert not report.healthy  # 61 msgs in 10 s -> 6.1 fps, below the minimum


def test_stream_health_carries_format_problems():
    report = evaluate_stream_health(
        _receipts_ns([1 / 60] * 600), duration_s=10.0,
        extra_problems=("encoding 'bgr8' is not rgb8",),
    )
    assert not report.healthy
