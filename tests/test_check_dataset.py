import sys
from types import SimpleNamespace

import h5py
import numpy as np

from tests.check_dataset import (
    EXPECTED_COLUMNS,
    action_direction,
    action_speed_bucket,
    check_episode,
    check_file,
    check_matching_videos,
    check_schema,
    collect_dataset_stats,
    combine_dataset_stats,
    discover_dataset_paths,
    format_duration,
    print_collection_goal,
    stats_review_warnings,
)


def test_commanded_motion_with_undefined_alignment_fails():
    length = 50
    action = np.zeros((length, 2), dtype=np.float32)
    action[:-1, 0] = 0.005
    proprio = np.zeros((length, 4), dtype=np.float32)
    state = np.full((length, 6), np.nan, dtype=np.float32)
    state[:, :2] = proprio[:, :2]
    command_ns = np.arange(length, dtype=np.int64) * 200_000_000 + 10_000_000
    receipt_ns = command_ns - 10_000_000

    result = check_episode(
        0,
        episode_idx=np.zeros(length, dtype=np.int64),
        step_idx=np.arange(length, dtype=np.int64),
        action=action,
        proprio=proprio,
        state=state,
        pixels_sample=np.stack(
            [
                np.zeros((224, 224, 3), dtype=np.uint8),
                np.ones((224, 224, 3), dtype=np.uint8),
            ]
        ),
        image_timestamp_ns=np.arange(length, dtype=np.int64) + 1,
        image_receipt_ns=receipt_ns,
        command_ns=command_ns,
    )

    assert any("undefined" in message for message in result.failures)


def _write_empty_columns(file: h5py.File) -> None:
    for name, (trailing_shape, dtype) in EXPECTED_COLUMNS.items():
        file.create_dataset(name, shape=(0, *trailing_shape), dtype=dtype)


def test_zero_length_episode_is_a_schema_failure_not_an_indexing_error(tmp_path):
    path = tmp_path / "zero-length.h5"
    with h5py.File(path, "w") as file:
        _write_empty_columns(file)
        file.create_dataset("ep_len", data=np.array([0], dtype=np.int64))
        file.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        schema, _ = check_schema(file)

    assert any("positive" in message for message in schema.failures)
    result, _ = check_file(path)
    assert any("positive" in message for message in result.failures)


def test_bookkeeping_arrays_must_be_one_dimensional(tmp_path):
    path = tmp_path / "bookkeeping-2d.h5"
    with h5py.File(path, "w") as file:
        _write_empty_columns(file)
        file.create_dataset("ep_len", data=np.empty((0, 1), dtype=np.int64))
        file.create_dataset("ep_offset", data=np.empty((0, 1), dtype=np.int64))
        schema, _ = check_schema(file)

    assert any("one-dimensional" in message for message in schema.failures)


def _write_stats_fixture(path, *, episode_id=7):
    length = 3
    pixels = np.zeros((length, 224, 224, 3), dtype=np.uint8)
    pixels[1:] = 20
    action = np.array(
        [[0.0025, 0.0], [0.0, 0.005], [0.0, 0.0]], dtype=np.float32
    )
    proprio = np.zeros((length, 4), dtype=np.float32)
    proprio[1, :2] = [0.0025, 0.0]
    proprio[2, :2] = [0.0025, 0.005]
    state = np.full((length, 6), np.nan, dtype=np.float32)
    state[:, :2] = proprio[:, :2]
    command_ns = np.arange(length, dtype=np.int64) * 200_000_000 + 10_000_000
    receipt_ns = command_ns - 10_000_000
    values = {
        "pixels": pixels,
        "action": action,
        "proprio": proprio,
        "state": state,
        "episode_idx": np.full(length, episode_id, dtype=np.int64),
        "step_idx": np.arange(length, dtype=np.int64),
        "image_timestamp_ns": np.arange(length, dtype=np.int64) * 200_000_000,
        "image_receipt_monotonic_ns": receipt_ns,
        "command_monotonic_ns": command_ns,
    }
    with h5py.File(path, "w") as file:
        for name, value in values.items():
            file.create_dataset(name, data=value)
        file.create_dataset("ep_len", data=np.array([length], dtype=np.int64))
        file.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))


def test_discovers_every_hdf5_file_recursively_when_paths_are_omitted(tmp_path):
    data_dir = tmp_path / "data"
    nested = data_dir / "nested"
    nested.mkdir(parents=True)
    first = data_dir / "first.h5"
    second = nested / "second.hdf5"
    ignored = data_dir / "notes.txt"
    first.touch()
    second.touch()
    ignored.touch()

    assert discover_dataset_paths([], data_dir) == [first, second]
    assert discover_dataset_paths([second], data_dir) == [second]


def test_action_direction_and_speed_bucketing():
    assert action_direction(np.array([0.0, 0.0])) == "zero"
    assert action_direction(np.array([0.005, -0.005])) == "+X-Y"
    assert action_direction(np.array([-0.005, 0.005])) == "-X+Y"
    assert action_speed_bucket(np.array([0.0025, 0.0])) == "2.5"
    assert action_speed_bucket(np.array([0.003, 0.0])) == "other"


def test_collection_goal_summary_prioritizes_time_remaining(capsys):
    print_collection_goal(2_132, 4.0)

    assert capsys.readouterr().out == (
        "\n4-HOUR DATA GOAL\n"
        "  Collected: 7m 06s of 4h 00m 00s (3.0%)\n"
        "  Remaining: 3h 52m 54s (69,868 samples at 5 Hz)\n"
    )
    assert format_duration(4 * 3600) == "4h 00m 00s"


def test_collects_timing_action_tracking_and_pixel_stats(tmp_path):
    path = tmp_path / "pushbox_keyboard_20260713_120000.h5"
    _write_stats_fixture(path)

    stats = collect_dataset_stats(path)

    assert stats.total_rows == 3
    assert stats.nonterminal_actions == 2
    assert stats.zero_actions == 0
    assert stats.speed_counts == {"2.5": 1, "5": 1}
    assert stats.direction_counts == {"+X": 1, "+Y": 1}
    np.testing.assert_allclose(stats.command_interval_ms, [200.0, 200.0])
    np.testing.assert_allclose(stats.achieved_ratios, [1.0, 1.0])
    assert stats.alignments[0] == 1.0
    assert stats.exact_duplicate_pairs == 1
    np.testing.assert_allclose(stats.pixel_mad, [20.0, 0.0])
    assert stats.episodes[0].peak_pixel_transition == 1
    warnings = stats_review_warnings(stats)
    assert any("pixel MAD" in message for message in warnings.warnings)

    combined = combine_dataset_stats([stats, stats])
    assert combined.total_rows == 6
    assert combined.speed_counts == {"2.5": 2, "5": 2}


def test_video_check_decodes_and_matches_hdf5_episode(monkeypatch, tmp_path):
    path = tmp_path / "pushbox_keyboard_20260713_120000.h5"
    _write_stats_fixture(path)
    stats = collect_dataset_stats(path, scan_pixels=False)
    video_dir = tmp_path / "videos" / "20260713_120000"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "ep_007.mp4"
    video_path.touch()

    class FakeReader:
        def __init__(self):
            self.values = iter(
                [
                    {"size": (224, 224), "fps": 5.0},
                    b"frame-0",
                    b"frame-1",
                    b"frame-2",
                ]
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.values)

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(read_frames=lambda *_args, **_kwargs: FakeReader()),
    )

    result, videos = check_matching_videos(stats, tmp_path / "videos")

    assert result.failures == []
    assert result.warnings == []
    assert len(videos) == 1
    assert videos[0].frames == 3
    assert videos[0].size == (224, 224)
    assert videos[0].fps == 5.0
