import numpy as np

from scripts.ros_camera import LatestImageStore


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
