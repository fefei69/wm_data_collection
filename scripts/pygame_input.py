"""Pure held-key and focus-safety state for the local Pygame collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

try:  # package import for tests; direct-file import for ``uv run scripts/...``
    from .collector_core import action_from_held, validate_magnitudes
except ImportError:  # pragma: no cover - exercised by direct script execution
    from collector_core import action_from_held, validate_magnitudes


ARROW_NAMES = ("up", "down", "left", "right")


@dataclass
class HeldInput:
    magnitudes: Mapping[int, float]
    speed_level: int = 2
    motion_enabled: bool = True
    _focus_present: bool = True
    _awaiting_release: bool = False
    _held: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.magnitudes = {int(level): float(value) for level, value in self.magnitudes.items()}
        validate_magnitudes(self.magnitudes, max(self.magnitudes.values()))
        self.set_speed_level(self.speed_level)

    def set_speed_level(self, level: int) -> None:
        level = int(level)
        if level not in self.magnitudes:
            raise ValueError(f"unknown speed level: {level}")
        self.speed_level = level

    def focus_lost(self) -> None:
        self._focus_present = False
        self.motion_enabled = False
        self._awaiting_release = True
        self._held.clear()

    def focus_regained(self) -> None:
        self._focus_present = True
        self.motion_enabled = False
        self._awaiting_release = True
        self._held.clear()

    def update_held(self, held: Mapping[str, bool]) -> np.ndarray:
        current = {name: bool(held.get(name, False)) for name in ARROW_NAMES}
        self._held = current
        if not self._focus_present:
            return np.zeros(2, dtype=np.float32)
        if self._awaiting_release:
            if any(current.values()):
                return np.zeros(2, dtype=np.float32)
            self._awaiting_release = False
            self.motion_enabled = True
        if not self.motion_enabled:
            return np.zeros(2, dtype=np.float32)
        return action_from_held(current, self.magnitudes[self.speed_level])


@dataclass(frozen=True)
class PygameEvents:
    quit_requested: bool = False
    save_toggle: bool = False
    discard_requested: bool = False
    reset_requested: bool = False
    focus_lost: bool = False
    focus_regained: bool = False


class PygameInputBackend:
    """Focused local preview and key-state backend with lazy Pygame import."""

    def __init__(self, held_input: HeldInput, size: int = 224, ui_fps: int = 30) -> None:
        self.held_input = held_input
        self.size = int(size)
        self.ui_fps = int(ui_fps)
        self._pygame: Any = None
        self._screen: Any = None
        self._font: Any = None

    def start(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Pygame is required on the local display host") from exc
        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((self.size * 2, self.size * 2))
        pygame.display.set_caption("Push-box keyboard collector")
        self._font = pygame.font.Font(None, 24)

    def poll(self) -> PygameEvents:
        if self._pygame is None:
            raise RuntimeError("PygameInputBackend.start() must be called first")
        pygame = self._pygame
        quit_requested = False
        save_toggle = False
        discard_requested = False
        reset_requested = False
        focus_lost = False
        focus_regained = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit_requested = True
                elif event.key == pygame.K_SPACE:
                    save_toggle = True
                elif event.key == pygame.K_d:
                    discard_requested = True
                elif event.key == pygame.K_r:
                    reset_requested = True
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.held_input.set_speed_level(int(event.key - pygame.K_0))
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.held_input.focus_lost()
                focus_lost = True
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.held_input.focus_regained()
                focus_regained = True
        if quit_requested:
            return PygameEvents(quit_requested=True, focus_lost=focus_lost, focus_regained=focus_regained)
        if discard_requested:
            return PygameEvents(discard_requested=True, focus_lost=focus_lost, focus_regained=focus_regained)
        if save_toggle:
            return PygameEvents(save_toggle=True, focus_lost=focus_lost, focus_regained=focus_regained)
        return PygameEvents(
            reset_requested=reset_requested,
            focus_lost=focus_lost,
            focus_regained=focus_regained,
        )

    def action(self) -> np.ndarray:
        pygame = self._pygame
        if pygame is None:
            raise RuntimeError("PygameInputBackend.start() must be called first")
        keys = pygame.key.get_pressed()
        return self.held_input.update_held({
            "up": keys[pygame.K_UP],
            "down": keys[pygame.K_DOWN],
            "left": keys[pygame.K_LEFT],
            "right": keys[pygame.K_RIGHT],
        })

    def render(self, rgb: np.ndarray, status: str) -> None:
        pygame = self._pygame
        if pygame is None or self._screen is None or self._font is None:
            raise RuntimeError("PygameInputBackend.start() must be called first")
        image = pygame.image.frombuffer(
            np.ascontiguousarray(rgb).tobytes(),
            (self.size, self.size),
            "RGB",
        )
        image = pygame.transform.scale(image, (self.size * 2, self.size * 2))
        self._screen.blit(image, (0, 0))
        overlay = pygame.Surface((self.size * 2, 30), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        overlay.blit(self._font.render(status, True, (255, 255, 255)), (8, 5))
        self._screen.blit(overlay, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._font = None
