import json

import h5py
import numpy as np
import pytest

from scripts.merge_datasets import (
    MergeError,
    collection_date,
    group_datasets,
    main,
    merge_date_group,
    output_path,
)


def _write_dataset(path, episode_lengths, *, value_offset=0, action_width=2):
    episode_lengths = np.asarray(episode_lengths, dtype=np.int32)
    total = int(episode_lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(episode_lengths)[:-1])).astype(np.int64)
    episode_idx = np.concatenate(
        [np.full(length, index, dtype=np.int64) for index, length in enumerate(episode_lengths)]
    )
    step_idx = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in episode_lengths]
    )
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "pixels",
            data=np.arange(value_offset, value_offset + total, dtype=np.uint8)[:, None, None, None],
            maxshape=(None, 1, 1, 1),
        )
        file.create_dataset(
            "action",
            data=np.full((total, action_width), value_offset, dtype=np.float32),
            maxshape=(None, action_width),
        )
        file.create_dataset("episode_idx", data=episode_idx, maxshape=(None,))
        file.create_dataset("step_idx", data=step_idx, maxshape=(None,))
        file.create_dataset("ep_len", data=episode_lengths, maxshape=(None,))
        file.create_dataset("ep_offset", data=offsets, maxshape=(None,))


def test_groups_by_filename_date_and_rejects_missing_date(tmp_path):
    first = tmp_path / "pushbox_keyboard_20260713_120000.h5"
    second = tmp_path / "pushbox_keyboard_20260714_090000.hdf5"
    _write_dataset(first, [2])
    _write_dataset(second, [3])

    grouped = group_datasets([first, second])

    assert list(grouped) == ["20260713", "20260714"]
    assert collection_date(first) == "20260713"
    with pytest.raises(MergeError, match="expected exactly one YYYYMMDD"):
        collection_date(tmp_path / "undated.h5")


def test_merge_reindexes_episodes_and_preserves_other_rows(tmp_path):
    first = tmp_path / "pushbox_keyboard_20260713_120000.h5"
    second = tmp_path / "pushbox_keyboard_20260713_130000.h5"
    _write_dataset(first, [2, 1], value_offset=10)
    _write_dataset(second, [3], value_offset=20)
    sources = group_datasets([first, second])["20260713"]
    destination = tmp_path / "merged" / "pushbox_keyboard_20260713.h5"

    result = merge_date_group("20260713", sources, destination)

    assert result.total_rows == 6
    assert result.episode_count == 3
    with h5py.File(destination, "r") as file:
        np.testing.assert_array_equal(file["ep_len"][:], [2, 1, 3])
        np.testing.assert_array_equal(file["ep_offset"][:], [0, 2, 3])
        np.testing.assert_array_equal(file["episode_idx"][:], [0, 0, 1, 2, 2, 2])
        np.testing.assert_array_equal(file["step_idx"][:], [0, 1, 0, 0, 1, 2])
        np.testing.assert_array_equal(file["pixels"][:, 0, 0, 0], [10, 11, 12, 20, 21, 22])
        assert file.attrs["merge_date"] == "20260713"
        assert json.loads(file.attrs["source_files_json"]) == [
            first.as_posix(),
            second.as_posix(),
        ]


def test_schema_mismatch_is_rejected_before_writing(tmp_path):
    first = tmp_path / "pushbox_keyboard_20260713_120000.h5"
    second = tmp_path / "pushbox_keyboard_20260713_130000.h5"
    _write_dataset(first, [2], action_width=2)
    _write_dataset(second, [2], action_width=3)

    with pytest.raises(MergeError, match="schema does not match"):
        group_datasets([first, second])


def test_existing_output_requires_overwrite(tmp_path):
    source = tmp_path / "pushbox_keyboard_20260713_120000.h5"
    _write_dataset(source, [2])
    sources = group_datasets([source])["20260713"]
    destination = output_path(tmp_path / "merged", "20260713")
    merge_date_group("20260713", sources, destination)

    with pytest.raises(MergeError, match="--overwrite"):
        merge_date_group("20260713", sources, destination)
    merge_date_group("20260713", sources, destination, overwrite=True)


def test_dry_run_groups_every_dataset_without_writing(tmp_path, capsys):
    data_dir = tmp_path / "data"
    nested = data_dir / "nested"
    nested.mkdir(parents=True)
    _write_dataset(data_dir / "pushbox_keyboard_20260713_120000.h5", [2])
    _write_dataset(nested / "pushbox_keyboard_20260713_130000.hdf5", [3])
    output_dir = tmp_path / "merged_data"

    assert main(["--data-dir", str(data_dir), "--output-dir", str(output_dir), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "PLAN 20260713: 2 file(s), 2 episode(s), 5 rows" in output
    assert not output_dir.exists()
