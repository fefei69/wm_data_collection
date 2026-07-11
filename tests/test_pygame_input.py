import numpy as np

from scripts.pygame_input import HeldInput


MAGNITUDES = {1: 0.0025, 2: 0.005, 3: 0.010}


def test_held_input_supports_diagonal_and_speed_selection():
    input_state = HeldInput(MAGNITUDES)
    input_state.set_speed_level(3)
    action = input_state.update_held({"up": True, "left": True})
    np.testing.assert_allclose(action, [0.010 / np.sqrt(2)] * 2)


def test_focus_loss_disables_motion_until_arrows_are_released():
    input_state = HeldInput(MAGNITUDES)
    input_state.focus_lost()
    input_state.focus_regained()
    np.testing.assert_array_equal(
        input_state.update_held({"up": True}),
        [0.0, 0.0],
    )
    np.testing.assert_array_equal(input_state.update_held({}), [0.0, 0.0])
    np.testing.assert_allclose(input_state.update_held({"left": True}), [0.0, 0.005])


def test_opposing_held_arrows_cancel():
    input_state = HeldInput(MAGNITUDES)
    np.testing.assert_array_equal(
        input_state.update_held({"up": True, "down": True}),
        [0.0, 0.0],
    )
