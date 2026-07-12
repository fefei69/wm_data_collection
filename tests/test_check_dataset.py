import h5py
import numpy as np

from tests.check_dataset import (
    EXPECTED_COLUMNS,
    check_episode,
    check_file,
    check_schema,
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
