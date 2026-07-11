import numpy as np
import pytest

from scripts.collect_keyboard_xy import (
    CollectorConfig,
    EpisodeRecorder,
    accepted_target,
)


def test_config_rejects_missing_commissioning_files(tmp_path):
    with pytest.raises(ValueError, match="transform profile"):
        CollectorConfig.from_paths(
            transform_profile=tmp_path / "missing.json",
            camera_params=tmp_path / "camera.yaml",
        )


def test_accepted_target_clips_workspace_and_reports_actual_delta():
    target, actual = accepted_target(
        np.array([0.095, -0.02]),
        np.array([0.01, -0.01]),
        bounds=((0.0, 0.1), (-0.05, 0.05)),
    )
    np.testing.assert_allclose(target, [0.1, -0.03])
    np.testing.assert_allclose(actual, [0.005, -0.01])


def test_accepted_target_does_not_clip_when_bounds_are_disabled():
    target, actual = accepted_target(
        np.array([0.095, -0.02]),
        np.array([0.01, -0.01]),
        bounds=None,
    )
    np.testing.assert_allclose(target, [0.105, -0.03])
    np.testing.assert_allclose(actual, [0.01, -0.01])


def test_config_allows_start_outside_bounds_when_limits_are_disabled(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text('{"mode":"square_roi","x":80,"y":0,"size":480}')
    camera = tmp_path / "camera.yaml"
    camera.write_text("camera: params")

    config = CollectorConfig.from_paths(
        transform_profile=profile,
        camera_params=camera,
        start_xy=(0.5, 0.5),
        enforce_xy_limits=False,
    )

    assert config.start_xy == (0.5, 0.5)


def test_driver_error_does_not_commit_staged_row():
    recorder = EpisodeRecorder()
    row = {"pixels": np.zeros((224, 224, 3), dtype=np.uint8)}
    with pytest.raises(RuntimeError):
        recorder.send_and_commit(row, lambda: (_ for _ in ()).throw(RuntimeError("driver")))
    assert recorder.rows == []


def test_recorder_requires_zero_terminal_action():
    recorder = EpisodeRecorder()
    recorder.append({"action": np.array([0.005, 0.0], dtype=np.float32)})
    with pytest.raises(ValueError, match="zero"):
        recorder.finish(save=False)
