#!/usr/bin/env python3
"""Merge collected HDF5 datasets into one training file per collection date.

By default, every ``.h5``/``.hdf5`` file below ``data/`` is grouped by the
``YYYYMMDD`` token in its filename and written to
``merged_data/pushbox_keyboard_YYYYMMDD.h5``. Source files are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


DATE_PATTERN = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
BOOKKEEPING_COLUMNS = frozenset({"ep_len", "ep_offset"})
REQUIRED_EPISODE_COLUMNS = frozenset({"episode_idx", "step_idx"})
TICK_SECONDS = 0.2
COPY_BATCH_ROWS = 64


class MergeError(RuntimeError):
    """Raised when inputs cannot be merged without changing their meaning."""


@dataclass(frozen=True)
class ColumnSpec:
    trailing_shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    date: str
    total_rows: int
    episode_lengths: np.ndarray
    episode_offsets: np.ndarray
    columns: dict[str, ColumnSpec]

    @property
    def episode_count(self) -> int:
        return int(self.episode_lengths.size)


def collection_date(path: Path) -> str:
    """Extract exactly one YYYYMMDD collection date from a filename."""
    dates = sorted(set(DATE_PATTERN.findall(path.stem)))
    if len(dates) != 1:
        raise MergeError(
            f"{path}: expected exactly one YYYYMMDD date in the filename; "
            f"found {dates or 'none'}"
        )
    return dates[0]


def discover_datasets(data_dir: Path) -> list[Path]:
    """Find every HDF5 file recursively below the source directory."""
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not paths:
        raise MergeError(f"no .h5/.hdf5 datasets found below {data_dir}")
    return paths


def inspect_dataset(path: Path, *, date: str | None = None) -> DatasetInfo:
    """Validate structural bookkeeping and return the locked column schema."""
    try:
        with h5py.File(path, "r") as source:
            groups = [name for name, value in source.items() if not isinstance(value, h5py.Dataset)]
            if groups:
                raise MergeError(f"{path}: nested HDF5 groups are unsupported: {groups}")
            missing_metadata = BOOKKEEPING_COLUMNS - set(source.keys())
            if missing_metadata:
                raise MergeError(
                    f"{path}: missing bookkeeping datasets: {sorted(missing_metadata)}"
                )

            episode_lengths = np.asarray(source["ep_len"], dtype=np.int64)
            episode_offsets = np.asarray(source["ep_offset"], dtype=np.int64)
            if episode_lengths.ndim != 1 or episode_offsets.ndim != 1:
                raise MergeError(f"{path}: ep_len and ep_offset must be one-dimensional")
            if episode_lengths.shape != episode_offsets.shape:
                raise MergeError(f"{path}: ep_len and ep_offset shapes do not match")
            if episode_lengths.size == 0 or np.any(episode_lengths <= 0):
                raise MergeError(f"{path}: every dataset must contain positive-length episodes")

            expected_offsets = np.concatenate(
                ([0], np.cumsum(episode_lengths, dtype=np.int64)[:-1])
            )
            if not np.array_equal(episode_offsets, expected_offsets):
                raise MergeError(f"{path}: ep_offset is not the running sum of ep_len")
            total_rows = int(episode_lengths.sum())

            column_names = sorted(set(source.keys()) - BOOKKEEPING_COLUMNS)
            missing_columns = REQUIRED_EPISODE_COLUMNS - set(column_names)
            if missing_columns:
                raise MergeError(
                    f"{path}: missing episode columns needed for safe merging: "
                    f"{sorted(missing_columns)}"
                )
            columns: dict[str, ColumnSpec] = {}
            for name in column_names:
                dataset = source[name]
                if dataset.ndim == 0 or dataset.shape[0] != total_rows:
                    raise MergeError(
                        f"{path}: column {name!r} has shape {dataset.shape}; "
                        f"expected first dimension {total_rows}"
                    )
                columns[name] = ColumnSpec(tuple(dataset.shape[1:]), np.dtype(dataset.dtype))

            for offset_value, length_value in zip(episode_offsets, episode_lengths):
                offset = int(offset_value)
                length = int(length_value)
                rows = slice(offset, offset + length)
                episode_ids = np.asarray(source["episode_idx"][rows])
                if not np.all(episode_ids == episode_ids[0]):
                    raise MergeError(f"{path}: episode_idx changes inside an episode")
                step_ids = np.asarray(source["step_idx"][rows])
                if not np.array_equal(step_ids, np.arange(length, dtype=step_ids.dtype)):
                    raise MergeError(f"{path}: step_idx is not 0..T-1 inside an episode")
    except OSError as exc:
        raise MergeError(f"{path}: cannot read HDF5 dataset: {exc}") from exc

    return DatasetInfo(
        path=path,
        date=date if date is not None else collection_date(path),
        total_rows=total_rows,
        episode_lengths=episode_lengths,
        episode_offsets=episode_offsets,
        columns=columns,
    )


def validate_group_schema(date: str, sources: Sequence[DatasetInfo]) -> None:
    """Require the exact same columns, dtypes, and per-row shapes for a date."""
    reference = sources[0]
    for source in sources[1:]:
        if source.columns != reference.columns:
            reference_names = set(reference.columns)
            source_names = set(source.columns)
            details: list[str] = []
            if reference_names - source_names:
                details.append(f"missing={sorted(reference_names - source_names)}")
            if source_names - reference_names:
                details.append(f"extra={sorted(source_names - reference_names)}")
            for name in sorted(reference_names & source_names):
                if source.columns[name] != reference.columns[name]:
                    details.append(
                        f"{name}: expected {reference.columns[name]}, "
                        f"got {source.columns[name]}"
                    )
            raise MergeError(
                f"{source.path}: schema does not match {reference.path} for {date}: "
                + "; ".join(details)
            )


def group_datasets(paths: Sequence[Path]) -> dict[str, list[DatasetInfo]]:
    """Inspect and group sources chronologically by filename date."""
    grouped: dict[str, list[DatasetInfo]] = defaultdict(list)
    for path in paths:
        date = collection_date(path)
        grouped[date].append(inspect_dataset(path, date=date))
    result = dict(sorted(grouped.items()))
    for date, sources in result.items():
        validate_group_schema(date, sources)
    return result


def output_path(output_dir: Path, date: str) -> Path:
    return output_dir / f"pushbox_keyboard_{date}.h5"


def _create_output_datasets(
    destination: h5py.File,
    sources: Sequence[DatasetInfo],
    *,
    date: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference = sources[0]
    total_rows = sum(source.total_rows for source in sources)
    episode_lengths = np.concatenate([source.episode_lengths for source in sources])
    episode_offsets = np.concatenate(
        ([0], np.cumsum(episode_lengths, dtype=np.int64)[:-1])
    ).astype(np.int64, copy=False)

    for name, spec in reference.columns.items():
        destination.create_dataset(
            name,
            shape=(total_rows, *spec.trailing_shape),
            maxshape=(None, *spec.trailing_shape),
            dtype=spec.dtype,
            chunks=(1, *spec.trailing_shape),
        )
    destination.create_dataset(
        "ep_len",
        data=episode_lengths.astype(np.int32),
        maxshape=(None,),
        chunks=(1024,),
    )
    destination.create_dataset(
        "ep_offset",
        data=episode_offsets,
        maxshape=(None,),
        chunks=(1024,),
    )
    destination.attrs["merge_date"] = date
    destination.attrs["source_file_count"] = len(sources)
    destination.attrs["source_files_json"] = json.dumps(
        [source.path.as_posix() for source in sources]
    )
    return episode_lengths, episode_offsets


def _copy_sources(destination: h5py.File, sources: Sequence[DatasetInfo]) -> None:
    destination_row = 0
    destination_episode = 0
    column_names = tuple(sources[0].columns)

    for info in sources:
        global_episode_ids = np.empty(info.total_rows, dtype=np.int64)
        for offset_value, length_value in zip(info.episode_offsets, info.episode_lengths):
            offset = int(offset_value)
            length = int(length_value)
            global_episode_ids[offset : offset + length] = destination_episode
            destination_episode += 1

        with h5py.File(info.path, "r") as source:
            for start in range(0, info.total_rows, COPY_BATCH_ROWS):
                stop = min(start + COPY_BATCH_ROWS, info.total_rows)
                output_rows = slice(destination_row + start, destination_row + stop)
                input_rows = slice(start, stop)
                for name in column_names:
                    if name == "episode_idx":
                        destination[name][output_rows] = global_episode_ids[input_rows]
                    else:
                        destination[name][output_rows] = source[name][input_rows]
        destination_row += info.total_rows


def merge_date_group(
    date: str,
    sources: Sequence[DatasetInfo],
    destination_path: Path,
    *,
    overwrite: bool = False,
) -> DatasetInfo:
    """Atomically merge one date group and validate the completed file."""
    if destination_path.exists() and not overwrite:
        raise MergeError(
            f"{destination_path} already exists; pass --overwrite to replace it"
        )
    if destination_path.resolve() in {source.path.resolve() for source in sources}:
        raise MergeError(f"refusing to overwrite a source dataset: {destination_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.stem}.",
        suffix=".tmp.h5",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with h5py.File(temporary_path, "w", libver="latest") as destination:
            _create_output_datasets(destination, sources, date=date)
            _copy_sources(destination, sources)
            destination.flush()
        merged_info = inspect_dataset(temporary_path, date=date)
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return DatasetInfo(
        path=destination_path,
        date=merged_info.date,
        total_rows=merged_info.total_rows,
        episode_lengths=merged_info.episode_lengths,
        episode_offsets=merged_info.episode_offsets,
        columns=merged_info.columns,
    )


def _duration_text(rows: int) -> str:
    total_seconds = int(round(rows * TICK_SECONDS))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def print_plan(
    grouped: dict[str, list[DatasetInfo]], output_dir: Path, *, label: str
) -> None:
    for date, sources in grouped.items():
        rows = sum(source.total_rows for source in sources)
        episodes = sum(source.episode_count for source in sources)
        print(
            f"{label} {date}: {len(sources)} file(s), {episodes} episode(s), "
            f"{rows:,} rows ({_duration_text(rows)} at 5 Hz) -> "
            f"{output_path(output_dir, date)}"
        )
        for source in sources:
            print(
                f"  {source.path}: {source.episode_count} episode(s), "
                f"{source.total_rows:,} rows"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="source directory scanned recursively (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("merged_data"),
        help="directory for date-grouped merged files (default: merged_data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing merged file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the merge plan without writing files",
    )
    args = parser.parse_args(argv)

    try:
        paths = discover_datasets(args.data_dir)
        grouped = group_datasets(paths)
        print_plan(grouped, args.output_dir, label="PLAN" if args.dry_run else "MERGE")
        if args.dry_run:
            return 0
        for date, sources in grouped.items():
            merged = merge_date_group(
                date,
                sources,
                output_path(args.output_dir, date),
                overwrite=args.overwrite,
            )
            print(
                f"DONE {merged.path}: {merged.episode_count} episode(s), "
                f"{merged.total_rows:,} rows ({_duration_text(merged.total_rows)})"
            )
    except (MergeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
