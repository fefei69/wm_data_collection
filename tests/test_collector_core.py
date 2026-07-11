import numpy as np
import pytest

from scripts.collector_core import (
    ImageTransformProfile,
    action_from_held,
    build_episode_columns,
    validate_magnitudes,
)


def test_diagonal_is_normalized_to_selected_norm():
    action = action_from_held(
        {"up": True, "down": False, "left": True, "right": False},
        0.005,
    )
    np.testing.assert_allclose(action, [0.005 / np.sqrt(2)] * 2)
    assert np.linalg.norm(action) == pytest.approx(0.005)


def test_opposing_keys_cancel_and_no_keys_hold():
    assert np.array_equal(
        action_from_held(
            {"up": True, "down": True, "left": False, "right": False},
            0.005,
        ),
        [0.0, 0.0],
    )
    assert np.array_equal(action_from_held({}, 0.005), [0.0, 0.0])


def test_magnitude_validation_requires_three_ordered_levels():
    validate_magnitudes({1: 0.0025, 2: 0.005, 3: 0.010}, 0.010)
    with pytest.raises(ValueError):
        validate_magnitudes({1: 0.005, 2: 0.0025, 3: 0.010}, 0.010)
    with pytest.raises(ValueError):
        validate_magnitudes({1: 0.0025, 2: 0.005, 3: 0.011}, 0.010)


def test_letterbox_preserves_full_four_to_three_frame():
    profile = ImageTransformProfile.from_dict({
        "mode": "letterbox",
        "fill_rgb": [0, 0, 0],
        "size": 224,
    })
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, 0, :] = [255, 0, 0]
    output = profile.apply(image)
    assert output.shape == (224, 224, 3)
    assert output.dtype == np.uint8
    assert np.all(output[0] == 0)


def test_square_roi_requires_complete_explicit_roi():
    profile = ImageTransformProfile.from_dict({
        "mode": "square_roi",
        "x": 80,
        "y": 0,
        "size": 480,
    })
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert profile.apply(image).shape == (224, 224, 3)
    with pytest.raises(ValueError):
        ImageTransformProfile.from_dict({"mode": "square_roi", "x": 80})


def test_episode_columns_are_equal_length_and_terminal_action_is_zero():
    rows = [{
        "pixels": np.zeros((224, 224, 3), np.uint8),
        "proprio": np.zeros(4, np.float32),
        "action": np.array([0.005, 0.0], np.float32),
        "image_timestamp_ns": 1,
        "image_receipt_monotonic_ns": 2,
        "command_monotonic_ns": 3,
    }, {
        "pixels": np.zeros((224, 224, 3), np.uint8),
        "proprio": np.zeros(4, np.float32),
        "action": np.zeros(2, np.float32),
        "image_timestamp_ns": 4,
        "image_receipt_monotonic_ns": 5,
        "command_monotonic_ns": 6,
    }]
    columns = build_episode_columns(rows, episode_id=7)
    assert all(len(value) == 2 for value in columns.values())
    np.testing.assert_array_equal(columns["action"][-1], [0.0, 0.0])
    np.testing.assert_array_equal(columns["episode_idx"], [7, 7])
    np.testing.assert_array_equal(columns["step_idx"], [0, 1])
