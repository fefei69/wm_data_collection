from types import SimpleNamespace

import numpy as np

from scripts.pygame_input import HeldInput, PygameInputBackend


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


def test_render_does_not_pace_the_control_loop():
    class Surface:
        def fill(self, *_args):
            pass

        def blit(self, *_args):
            pass

    class Clock:
        calls = 0

        def tick(self, _fps):
            self.calls += 1

    backend = PygameInputBackend(HeldInput(MAGNITUDES))
    clock = Clock()
    backend._pygame = SimpleNamespace(
        image=SimpleNamespace(frombuffer=lambda *_args: object()),
        transform=SimpleNamespace(scale=lambda image, _size: image),
        Surface=lambda *_args: Surface(),
        SRCALPHA=1,
        display=SimpleNamespace(flip=lambda: None),
    )
    backend._screen = Surface()
    backend._font = SimpleNamespace(render=lambda *_args: object())
    backend._clock = clock

    backend.render(np.zeros((224, 224, 3), dtype=np.uint8), "idle")

    assert clock.calls == 0
