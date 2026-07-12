# Collector Cadence and Dataset QA Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee saved episodes satisfy the 5 Hz cadence contract and make offline QA reject undefined alignment and malformed episode bookkeeping safely.

**Architecture:** Keep Pygame preview rendering at 30 Hz but remove its blocking frame limiter from `render()`, pacing the main loop independently at short sleeps. Centralize tick-deadline and actual-command-interval decisions in pure helpers, then harden the standalone HDF5 checker before it indexes episode rows.

**Tech Stack:** Python 3.12, NumPy, Pygame, h5py, pytest, OpenCV (headless in tests)

## Global Constraints

- Camera delivery remains approximately 58-60 Hz; recording consumes one newest, fresh, previously unused frame per 5 Hz tick.
- Preview remains capped at 30 Hz and may display the same latest frame repeatedly.
- Valid command cadence is exactly 0.20 +/- 0.02 seconds.
- A timing violation discards the active episode; no catch-up ticks are emitted.
- Existing untracked `tmp/` content is not modified or committed.

---

### Task 1: Decouple preview pacing and enforce collection cadence

**Files:**
- Modify: `tests/test_pygame_input.py`
- Modify: `tests/test_collect_keyboard_xy.py`
- Modify: `scripts/pygame_input.py`
- Modify: `scripts/collect_keyboard_xy.py`

**Interfaces:**
- Produces: `advance_tick_deadline(deadline_s: float, now_s: float) -> tuple[float, bool]`
- Produces: `command_interval_is_valid(previous_ns: int | None, current_ns: int) -> bool`
- Changes: `PygameInputBackend.render()` renders without sleeping.

- [ ] **Step 1: Write a failing nonblocking-render test**

Add a lightweight fake Pygame surface/display to `tests/test_pygame_input.py`, call `PygameInputBackend.render()`, and assert a fake clock's `tick()` was not called. The current implementation calls it once.

```python
def test_render_does_not_pace_the_control_loop():
    class Surface:
        def fill(self, *_args): pass
        def blit(self, *_args): pass

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
```

- [ ] **Step 2: Run the render test and verify RED**

Run:

```bash
uv run --no-project --with pytest --with numpy python -m pytest -q tests/test_pygame_input.py::test_render_does_not_pace_the_control_loop
```

Expected: FAIL because `clock.calls == 1`.

- [ ] **Step 3: Remove frame limiting from `render()`**

Delete `self._clock.tick(self.ui_fps)` from `PygameInputBackend.render()`. Keep `ui_fps` as the main loop's preview cadence setting; remove `_clock` construction/reset if it has no remaining use.

- [ ] **Step 4: Run the render test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Write failing deadline and command-interval tests**

Import `advance_tick_deadline` and `command_interval_is_valid` in `tests/test_collect_keyboard_xy.py`, then add:

```python
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
```

- [ ] **Step 6: Run the cadence tests and verify RED**

Run:

```bash
uv run --no-project --with pytest --with numpy --with opencv-python-headless python -m pytest -q tests/test_collect_keyboard_xy.py
```

Expected: collection error because the two helpers do not exist.

- [ ] **Step 7: Implement the pure cadence helpers**

In `scripts/collect_keyboard_xy.py`, replace the 100 ms overrun threshold with the QA tolerance and add:

```python
TICK_TOLERANCE_S = 0.02
UI_SLEEP_MAX_S = 0.005


def advance_tick_deadline(deadline_s: float, now_s: float) -> tuple[float, bool]:
    late = float(now_s) - float(deadline_s)
    if late > TICK_TOLERANCE_S:
        return float(now_s) + TICK_S, True
    return float(deadline_s) + TICK_S, False


def command_interval_is_valid(previous_ns: int | None, current_ns: int) -> bool:
    if previous_ns is None:
        return True
    interval_s = (int(current_ns) - int(previous_ns)) / 1e9
    return abs(interval_s - TICK_S) <= TICK_TOLERANCE_S
```

- [ ] **Step 8: Integrate independent preview scheduling and actual-command validation**

Initialize `next_render = time.monotonic()` and `previous_command_ns = None`. Reset the command timestamp in both `discard_episode()` and episode start. Use `advance_tick_deadline()` when a tick is due; discard recording on `missed=True`. Immediately before building/sending a recording row, capture `command_ns`; if it is invalid relative to `previous_command_ns`, discard without committing or sending that row. Otherwise store it in the row and update `previous_command_ns` only after a successful send.

After event/tick processing, render only when `now >= next_render`, then set `next_render = now + 1 / pygame_input.ui_fps`. Sleep at most `UI_SLEEP_MAX_S` toward the earlier of `next_tick` and `next_render`; this keeps event polling responsive without a CPU spin.

- [ ] **Step 9: Run collector and Pygame tests and verify GREEN**

Run:

```bash
uv run --no-project --with pytest --with numpy --with opencv-python-headless python -m pytest -q tests/test_collect_keyboard_xy.py tests/test_pygame_input.py
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 1**

```bash
git add scripts/collect_keyboard_xy.py scripts/pygame_input.py tests/test_collect_keyboard_xy.py tests/test_pygame_input.py
git commit -m "fix: enforce dataset-safe collection cadence"
```

---

### Task 2: Reject undefined alignment when motion was commanded

**Files:**
- Create: `tests/test_check_dataset.py`
- Modify: `tests/check_dataset.py`

**Interfaces:**
- Consumes: `check_episode(...) -> CheckResult`
- Changes: undefined correlation plus nonzero commanded action adds a failure.

- [ ] **Step 1: Write a failing undefined-alignment test**

Create a 50-row valid episode fixture with constant proprio X/Y, at least one nonzero action before the terminal zero row, valid state/timestamps, and nonconstant sampled pixels. Call `check_episode()` and assert:

```python
assert any("undefined" in message for message in result.failures)
```

- [ ] **Step 2: Run the alignment test and verify RED**

Run:

```bash
uv run --no-project --with pytest --with numpy --with h5py python -m pytest -q tests/test_check_dataset.py::test_commanded_motion_with_undefined_alignment_fails
```

Expected: FAIL because `result.failures` has no alignment failure.

- [ ] **Step 3: Implement undefined-correlation handling**

In `check_episode()`, calculate whether `action[:-1]` contains commanded motion above `CAP_EPSILON_M`. Replace the NaN-skipping conditional with:

```python
commanded_motion = bool(np.any(np.linalg.norm(action[:-1], axis=1) > CAP_EPSILON_M))
if np.isnan(correlation):
    if commanded_motion:
        result.fail(
            f"{prefix}: displacement/action correlation is undefined despite nonzero commands"
        )
elif correlation < MIN_ALIGNMENT_CORRELATION:
    hint = ""
    shifted = {s: alignment_correlation(proprio, action, shift=s) for s in (-1, 1)}
    best_shift = max(shifted, key=lambda s: np.nan_to_num(shifted[s], nan=-2.0))
    if np.nan_to_num(shifted[best_shift], nan=-2.0) > correlation + 0.1:
        hint = (
            f"; correlates better with action[t{best_shift:+d}]"
            f" (r={shifted[best_shift]:.2f}) — logging may be off by one tick"
        )
    result.warn(
        f"{prefix}: displacement/action correlation r={correlation:.2f} is below"
        f" {MIN_ALIGNMENT_CORRELATION}{hint}"
    )
```

- [ ] **Step 4: Run the alignment test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tests/check_dataset.py tests/test_check_dataset.py
git commit -m "fix: reject undefined commanded-motion alignment"
```

---

### Task 3: Validate episode bookkeeping before indexing

**Files:**
- Modify: `tests/test_check_dataset.py`
- Modify: `tests/check_dataset.py`

**Interfaces:**
- Consumes: `check_schema(f: h5py.File) -> tuple[CheckResult, int]`
- Changes: malformed bookkeeping becomes `CheckResult.failures`, never an indexing exception.

- [ ] **Step 1: Write failing malformed-bookkeeping tests**

Add an HDF5 fixture helper that creates zero-row datasets for every `EXPECTED_COLUMNS` entry. Test `ep_len=[0]`, `ep_offset=[0]` through `check_file()` and assert it returns a failure containing `positive`. Add a second test with two-dimensional bookkeeping arrays and assert `check_schema()` returns a failure containing `one-dimensional`.

- [ ] **Step 2: Run the bookkeeping tests and verify RED**

Run:

```bash
uv run --no-project --with pytest --with numpy --with h5py python -m pytest -q tests/test_check_dataset.py
```

Expected: the zero-length case raises `IndexError`, and the dimensionality assertion fails.

- [ ] **Step 3: Implement strict bookkeeping schema checks**

Before converting bookkeeping values, require both datasets to exist, have `ndim == 1`, and have integer dtypes using `np.issubdtype(dataset.dtype, np.integer)`. Return the accumulated schema failures before episode arithmetic when those structural checks fail. After conversion, fail if any `ep_len <= 0` or `ep_offset < 0`; retain equal-shape, row-total, and running-offset validation.

- [ ] **Step 4: Run the checker tests and verify GREEN**

Run the Step 2 command. Expected: all checker tests pass without exceptions.

- [ ] **Step 5: Run the complete hardware-independent suite**

Run:

```bash
uv run --no-project --with pytest --with numpy --with h5py --with opencv-python-headless python -m pytest -q tests
```

Expected: all tests pass.

- [ ] **Step 6: Run the checker against the pilot capture**

Run:

```bash
uv run --no-project --with h5py --with numpy python tests/check_dataset.py data/pushbox_keyboard_20260711_172610.h5
```

Expected: exit 1 with the already-known cadence failures; no traceback. This historical file is not rewritten.

- [ ] **Step 7: Commit Task 3**

```bash
git add tests/check_dataset.py tests/test_check_dataset.py
git commit -m "fix: validate dataset episode bookkeeping"
```

---

### Task 4: Final verification

**Files:**
- Verify only; no planned source changes.

**Interfaces:**
- Verifies all interfaces produced by Tasks 1-3.

- [ ] **Step 1: Check formatting and repository state**

Run:

```bash
git diff --check origin/main...HEAD
git status --short
```

Expected: no whitespace errors; only the user's pre-existing untracked `tmp/` remains.

- [ ] **Step 2: Re-run the complete hardware-independent suite**

Run the Task 3 Step 5 command. Expected: all tests pass.

- [ ] **Step 3: Review committed diff**

Run:

```bash
git diff --stat origin/main...HEAD
git log --oneline --decorate origin/main..HEAD
```

Expected: three focused implementation commits plus the approved design and implementation-plan commits, with no `tmp/` files included.
