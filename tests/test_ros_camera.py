from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ros_camera import (
    LatestImageStore,
    RosImageSubscriber,
    decode_rgb8,
    evaluate_stream_health,
)


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


def test_subscriber_take_error_returns_and_clears():
    subscriber = RosImageSubscriber(LatestImageStore())
    assert subscriber.take_error() is None
    error = ValueError("bad frame")
    subscriber._record_error(error)
    assert subscriber.take_error() is error
    assert subscriber.take_error() is None


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


def test_stream_health_tolerates_one_routine_hiccup_but_not_two():
    # docs/hardware-api-reference.md: 100-370 ms hiccups are routine on a
    # healthy rig, so a single one must not fail the gate.
    one_hiccup = [1 / 60] * 300 + [0.350] + [1 / 60] * 300
    report = evaluate_stream_health(_receipts_ns(one_hiccup), duration_s=10.35)
    assert report.healthy

    two_hiccups = [1 / 60] * 200 + [0.350] + [1 / 60] * 200 + [0.350] + [1 / 60] * 200
    report = evaluate_stream_health(_receipts_ns(two_hiccups), duration_s=10.7)
    assert not report.healthy
    assert any("older than" in problem for problem in report.problems)


def test_stream_health_rejects_large_gaps_and_stale_time():
    gaps = [1 / 60] * 300 + [0.600] + [1 / 60] * 300
    report = evaluate_stream_health(_receipts_ns(gaps), duration_s=10.6)
    assert not report.healthy
    assert any("gap" in problem for problem in report.problems)

    gaps = ([1 / 60] * 50 + [0.25]) * 20  # repeated 250 ms dropouts
    report = evaluate_stream_health(_receipts_ns(gaps), duration_s=22.0)
    assert not report.healthy
    assert any("older than" in problem for problem in report.problems)


def test_stream_health_rejects_a_degraded_median_cadence():
    # dataset_spec.md: a median gap above 50 ms indicates a degraded stream,
    # even when the average rate still clears the fps minimum.
    gaps = [0.060] * 151 + [0.010] * 150
    report = evaluate_stream_health(_receipts_ns(gaps), duration_s=10.56)
    assert not report.healthy
    assert any("degraded" in problem for problem in report.problems)


def test_stream_health_detects_a_stream_that_dies_mid_probe():
    receipts = _receipts_ns([1 / 60] * 200)  # healthy 60 fps for ~3.3 s
    report = evaluate_stream_health(
        receipts, duration_s=5.0, probe_end_monotonic_ns=int(5.0e9)
    )
    assert not report.healthy
    assert any("gap" in problem for problem in report.problems)


def test_stream_health_ignores_subscription_discovery_latency():
    # First frame lands 3.5 s into a 5 s probe (discovery latency), then a
    # clean 60 fps stream: rates are judged from the first receipt onward.
    receipts = [int((3.5 + i / 60) * 1e9) for i in range(91)]
    report = evaluate_stream_health(
        receipts, duration_s=5.0, probe_end_monotonic_ns=int(5.0e9)
    )
    assert report.healthy
    assert report.fps == pytest.approx(60.7, abs=0.5)


def test_stream_health_rejects_an_empty_or_missing_stream():
    report = evaluate_stream_health([], duration_s=10.0)
    assert not report.healthy
    report = evaluate_stream_health(
        _receipts_ns([1 / 60] * 60), duration_s=10.0, probe_end_monotonic_ns=int(10.0e9)
    )
    assert not report.healthy  # 61 msgs, then silence for 9 s -> 6.1 fps


def test_stream_health_carries_format_problems():
    report = evaluate_stream_health(
        _receipts_ns([1 / 60] * 600), duration_s=10.0,
        extra_problems=("encoding 'bgr8' is not rgb8",),
    )
    assert not report.healthy
