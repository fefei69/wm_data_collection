import numpy as np
import pytest

from scripts.collect_keyboard_xy import (
    CollectorConfig,
    EpisodeRecorder,
    FixedZMonitor,
    accepted_target,
    advance_tick_deadline,
    command_interval_is_valid,
    move_arm_home_then_zero,
)


def test_default_config_validates_with_limits_enforced():
    # Guards against the defaults drifting apart again (start_xy used to sit
    # outside x_bounds, making every default-constructed config self-reject).
    CollectorConfig().validate()


def test_config_rejects_nonpositive_fixed_z_warning_tolerance():
    with pytest.raises(ValueError, match="fixed_z_warning_tolerance"):
        CollectorConfig(fixed_z_warning_tolerance=0.0).validate()


def test_fixed_z_monitor_reports_deviation_and_recovery_transitions():
    monitor = FixedZMonitor(target_z=0.03, tolerance_m=0.002)

    assert monitor.update(0.031) is None
    warning = monitor.update(0.033)
    assert warning is not None
    assert "WARNING" in warning
    assert "error=+3.0 mm" in warning
    assert monitor.update(0.034) is None

    recovery = monitor.update(0.0315)
    assert recovery is not None
    assert "recovered" in recovery
    assert monitor.update(0.03) is None


def test_fixed_z_monitor_warns_for_nonfinite_measurement():
    monitor = FixedZMonitor(target_z=0.03)

    warning = monitor.update(float("nan"))

    assert warning is not None
    assert "WARNING" in warning
    assert "not finite" in warning


def test_shutdown_moves_home_then_zero_with_blocking_adapter_calls(capsys):
    class FakeArm:
        def __init__(self):
            self.calls = []

        def home(self, positions, goal_time=2.0):
            self.calls.append((np.asarray(positions).copy(), goal_time))

    arm = FakeArm()
    home = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0])

    move_arm_home_then_zero(arm, home)

    assert len(arm.calls) == 2
    np.testing.assert_allclose(arm.calls[0][0], home)
    np.testing.assert_allclose(arm.calls[1][0], np.zeros(7))
    assert arm.calls[0][1] == 2.0
    assert arm.calls[1][1] == 2.0
    output = capsys.readouterr().out
    assert output.index("home position") < output.index("zero position")


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


def test_tick_deadline_advances_normally_and_realigns_after_tolerance():
    deadline, missed = advance_tick_deadline(10.0, 10.019)
    assert deadline == pytest.approx(10.2)
    assert missed is False

    deadline, missed = advance_tick_deadline(10.0, 10.021)
    assert deadline == pytest.approx(10.221)
    assert missed is True


def test_command_interval_uses_the_same_twenty_ms_tolerance_as_qa():
    assert command_interval_is_valid(None, 1_000_000_000)
    assert command_interval_is_valid(1_000_000_000, 1_180_000_000)
    assert command_interval_is_valid(1_000_000_000, 1_220_000_000)
    assert not command_interval_is_valid(1_000_000_000, 1_179_999_999)
    assert not command_interval_is_valid(1_000_000_000, 1_220_000_001)
