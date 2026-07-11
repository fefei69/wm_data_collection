# ROS 2 Pygame Keyboard Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the follower-only keyboard collector described by the approved specification, using ROS 2 Jazzy images, a focused Pygame held-key interface, 5 Hz `frameskip=1` data, and episode-aware HDF5 output.

**Architecture:** Keep all action math, image-profile validation, timestamp pairing, and episode-row construction in dependency-light modules that can be tested without hardware. Keep ROS 2, Pygame, Trossen, and stable-worldmodel imports lazy inside the executable collector so unit tests run on the development machine. The collector samples a latest-image ROS buffer at 5 Hz, sends fixed-Z/orientation Cartesian targets, and writes accepted command rows through `stable_worldmodel.HDF5Writer`.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Pygame, ROS 2 Jazzy (`rclpy`, `sensor_msgs`, `cv_bridge`), Trossen Arm 1.9.x, stable-worldmodel 0.1.1, HDF5Writer, imageio-ffmpeg.

## Global Constraints

- ROS topic: `/camera/camera/color/image_raw`, `sensor_msgs/msg/Image`, `rgb8`, 640×480, expected 30 fps.
- ROS subscription: best effort, volatile, keep-last depth 1; retain only the newest frame.
- Dataset/control rate: 5 Hz (`TICK_S = 0.2`); LeWM training `frameskip=1`.
- Action magnitudes: 0.0025, 0.005, and 0.010 m/step; `norm(action) <= 0.010` after diagonal normalization and workspace clipping.
- Fixed orientation: angle-axis `[0, π/2, 0]`; only base X/Y translation changes.
- The process requires a focused local Pygame window; terminal/SSH motion input is unsupported.
- Focus loss must clear motion, issue a zero hold, discard an active episode, and require released arrows after focus returns.
- The image transform is selected by a commissioned profile file; never stretch 4:3 to 1:1 or mix profiles in one dataset.
- Native 30 fps archival is a separate rosbag2 process; the collector’s MP4 contains selected 5 Hz model frames.
- No detector, AprilTag, online box pose, gravity-compensation mode, or automatic box reset.
- Existing `scripts/collect_data.py` remains unchanged as the legacy direct-RealSense prototype.
- Do not commit unrelated existing worktree changes or deleted files.

---

### Task 1: Add dependency-light action and image-profile core

**Files:**
- Create: `scripts/collector_core.py`
- Create: `tests/test_collector_core.py`

**Interfaces:**
- `action_from_held(held: Mapping[str, bool], magnitude_m: float) -> np.ndarray`: returns a normalized 2D base-frame delta for `up`, `down`, `left`, and `right` held states.
- `validate_magnitudes(magnitudes: Mapping[int, float], cap_m: float) -> None`: raises `ValueError` for missing levels, non-positive/non-increasing values, or values above the cap.
- `ImageTransformProfile.from_dict(data: Mapping[str, Any]) -> ImageTransformProfile`: accepts `square_roi` or `letterbox` profile data and rejects missing/invalid parameters.
- `ImageTransformProfile.apply(rgb: np.ndarray) -> np.ndarray`: returns `(224, 224, 3)` `uint8` without geometric stretching.
- `build_episode_columns(rows: Sequence[Mapping[str, Any]], episode_id: int) -> dict[str, np.ndarray]`: creates equal-length HDF5Writer columns and validates row shapes.

- [ ] **Step 1: Write failing action tests**

```python
def test_diagonal_is_normalized_to_selected_norm():
    action = action_from_held(
        {"up": True, "down": False, "left": True, "right": False},
        0.005,
    )
    np.testing.assert_allclose(action, [0.005 / np.sqrt(2)] * 2)
    assert np.linalg.norm(action) == pytest.approx(0.005)

def test_opposing_keys_cancel_and_no_keys_hold():
    assert np.array_equal(
        action_from_held({"up": True, "down": True, "left": False, "right": False}, 0.005),
        [0.0, 0.0],
    )
    assert np.array_equal(action_from_held({}, 0.005), [0.0, 0.0])
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-import failure**

Run: `python -m pytest -q tests/test_collector_core.py -k 'diagonal or opposing'`

Expected: FAIL because `scripts.collector_core` does not yet exist.

- [ ] **Step 3: Implement the minimal action helpers**

Implement `action_from_held` by computing `[up-down, left-right]`, returning zero for norm zero, and otherwise scaling the unit direction by `magnitude_m`. Implement magnitude validation with sorted integer levels and a finite cap check.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest -q tests/test_collector_core.py -k 'diagonal or opposing'`

Expected: the two action tests PASS.

- [ ] **Step 5: Add failing image-profile and episode-schema tests**

```python
def test_letterbox_preserves_full_4_to_3_frame():
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
```

- [ ] **Step 6: Run the new tests and verify they fail for the missing profile/schema implementation**

Run: `python -m pytest -q tests/test_collector_core.py -k 'letterbox or episode_columns'`

Expected: FAIL because the profile and row-builder implementations are not present.

- [ ] **Step 7: Implement profile and row-builder behavior**

Implement `letterbox` with aspect-preserving resize and fixed padding, `square_roi` with explicit `x`, `y`, and `size`, strict RGB/uint8/shape checks, and row stacking for `pixels`, `action`, `proprio`, `state`, episode/step indexes, and three timestamp columns. Reject mixed row lengths and non-final nonzero terminal padding.

- [ ] **Step 8: Run all core tests**

Run: `python -m pytest -q tests/test_collector_core.py`

Expected: all core tests PASS.

---

### Task 2: Add ROS latest-frame buffer and Pygame input state

**Files:**
- Create: `scripts/ros_camera.py`
- Create: `scripts/pygame_input.py`
- Modify: `tests/test_collector_core.py`
- Create: `tests/test_ros_camera.py`
- Create: `tests/test_pygame_input.py`

**Interfaces:**
- `LatestImageStore.update(rgb, source_timestamp_ns, receipt_monotonic_ns) -> int`: atomically replaces the latest frame and increments a sequence.
- `LatestImageStore.snapshot() -> ImageSnapshot | None`: returns a copy/reference plus sequence and timestamps.
- `LatestImageStore.accept(snapshot, previous_sequence, now_monotonic_ns, max_age_s) -> bool`: enforces new-frame and freshness rules.
- `HeldInput.direction_and_magnitude(held_keys: Mapping[str, bool], speed_level: int) -> np.ndarray`: delegates to core action math.
- `HeldInput.focus_lost() -> None`, `HeldInput.focus_regained() -> None`, `HeldInput.motion_enabled -> bool`: implements focus safety.

- [ ] **Step 1: Write failing buffer and input tests**

Test that a second image replaces the first, stale/reused snapshots are rejected, focus loss disables motion, and focus return does not re-enable motion until arrows are released.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q tests/test_ros_camera.py tests/test_pygame_input.py`

Expected: FAIL because the modules do not yet exist.

- [ ] **Step 3: Implement the pure buffer/input state**

Use `threading.Lock` for the image slot. Store `np.asarray(rgb).copy()`, the ROS source stamp, host receipt time, and sequence. `accept` must reject sequence reuse and receipt age greater than `max_age_s`. Pygame event translation remains in the executable collector; the helper only owns state transitions and action calculation.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest -q tests/test_ros_camera.py tests/test_pygame_input.py`

Expected: all buffer/input tests PASS.

- [ ] **Step 5: Add the ROS 2 subscriber adapter without importing ROS at test collection time**

Implement `RosImageSubscriber` with lazy imports inside `start()`: create a Jazzy node, subscribe to `sensor_msgs/msg/Image` using a depth-1 best-effort/volatile `QoSProfile`, convert via `CvBridge.imgmsg_to_cv2(message, desired_encoding="rgb8")`, validate encoding/dimensions, and update `LatestImageStore`. Run the executor in a daemon thread and expose `stop()` that destroys the node and calls `rclpy.shutdown()` only when this process initialized ROS.

- [ ] **Step 6: Add Pygame held-key event translation**

Implement edge-triggered lifecycle events for `q`, `d`, `SPACE`, `r`, and speed levels `1`/`2`/`3`; use `pygame.key.get_pressed()` for motion; handle `WINDOWFOCUSLOST`, `WINDOWFOCUSGAINED`, and `QUIT`; and expose a 224×224 preview renderer with status text. Motion must be disabled on focus loss and require released arrows after focus regain.

- [ ] **Step 7: Run import and unit verification**

Run: `python -m pytest -q tests/test_collector_core.py tests/test_ros_camera.py tests/test_pygame_input.py` and `python -m py_compile scripts/collector_core.py scripts/ros_camera.py scripts/pygame_input.py`.

Expected: unit tests PASS and all pure modules compile without requiring ROS or a display.

---

### Task 3: Implement the hardware collector and episode lifecycle

**Files:**
- Create: `scripts/collect_keyboard_xy.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock` only if `uv lock` can resolve without altering unrelated dependencies
- Create: `tests/test_collect_keyboard_xy.py`

**Interfaces:**
- `CollectorConfig`: validates follower IP, fixed pose, workspace bounds, action magnitudes, ROS profile, image-transform profile, camera parameter dump path, and timing.
- `FollowerArmAdapter`: wraps the pinned Trossen calls `set_all_modes`, `set_all_positions`, `get_cartesian_positions`, `get_cartesian_velocities`, and `set_cartesian_positions` with explicit Cartesian interpolation, goal time, blocking, and trajectory-check arguments.
- `EpisodeRecorder.start()`, `EpisodeRecorder.append(row)`, `EpisodeRecorder.finish(save: bool)`: buffers equal-length rows, writes through `swm.data.HDF5Writer`, and writes selected 5 Hz model-view MP4 frames.
- `main(argv: Sequence[str]) -> int`: starts ROS/Pygame only after config validation, performs commissioned startup, runs the 5 Hz loop, and follows safe shutdown.

- [ ] **Step 1: Write failing pure collector tests**

Test config rejection for an absent transform/camera-parameter file, accepted-target clipping, final zero-action row, and rejection of a driver command that raises before committing the staged row.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_collect_keyboard_xy.py`

Expected: FAIL because the collector module and adapters do not yet exist.

- [ ] **Step 3: Implement config and arm adapter**

Use lazy imports for `trossen_arm`, `stable_worldmodel`, `pygame`, `rclpy`, and `cv2`. Validate all finite pose/bound values, require the exact three magnitudes and 10 mm cap, require the two commissioning files, configure the follower only, set position mode, execute slow checked startup moves through safe Z, then use explicit nonblocking Cartesian calls at `goal_time=0.2` for the 5 Hz loop. Never copy Z/orientation from measured pose.

- [ ] **Step 4: Run pure config/adapter tests and verify GREEN**

Run: `python -m pytest -q tests/test_collect_keyboard_xy.py -k 'config or target or commit'`

Expected: all pure collector tests PASS.

- [ ] **Step 5: Implement episode recorder and alignment**

At each tick, snapshot a new fresh ROS image, apply the commissioned transform, read proprio, compute the held-key action, stage the row, timestamp immediately before the arm call, send the exact accepted target, and append only after success. On graceful stop, append the final observation with zero action before writing. Use `HDF5Writer` with `mode="append"`; never use the legacy bare HDF5 layout.

- [ ] **Step 6: Implement Pygame/ROS main loop and safety paths**

Pump the Pygame window at `UI_FPS`, run ROS executor in its thread, schedule commands at 5 Hz using a monotonic deadline, discard on stale/reused frames or focus loss, and handle `SPACE`, `d`, `r`, `q`, window close, driver errors, and ROS shutdown exactly as the specification states. Keep the Pygame shutdown confirmation window open until Enter before executing the commissioned safe retreat.

- [ ] **Step 7: Run static verification and hardware-free tests**

Run: `python -m pytest -q`, `python -m py_compile scripts/*.py`, and `git diff --check`.

Expected: all tests PASS, Python compilation succeeds, and only intended files differ.

---

### Task 4: Synchronize documentation and perform robot-host commissioning checks

**Files:**
- Modify: `README.md`
- Modify: `dataset_spec.md`
- Modify: `docs/keyboard_xy_collection_spec.md`
- Modify: `docs/teleoperator_guidelines.md`
- Modify: `docs/hardware-api-reference.md`

- [ ] **Step 1: Update docs to name the implemented Pygame collector and speed levels**

Document the new script path, local-display requirement, normalized diagonal mapping, 2.5/5/10 mm levels, 10 mm action cap, focus-loss behavior, and CEM projection boundary. Keep the legacy `collect_data.py` direct-RealSense description explicitly separate.

- [ ] **Step 2: Add the robot-host preflight commands**

Run on the Jazzy host:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic info -v /camera/camera/color/image_raw
ros2 topic hz /camera/camera/color/image_raw
ros2 node list
ros2 param dump "$(ros2 node list | rg 'camera|realsense' | head -n 1)" > camera_params.yaml
```

Expected: `sensor_msgs/msg/Image`, `rgb8`, 640×480, approximately 30 Hz, sensor-data-compatible QoS, and a saved parameter dump with supported auto controls disabled.

- [ ] **Step 3: Commission the arm and input without contact**

Run the collector with the tool raised and no box. Verify startup path, fixed orientation, all eight normalized directions, all three magnitudes, opposing-key cancellation, focus-loss stop, XY bounds, and safe retreat.

- [ ] **Step 4: Commission recording and loader alignment**

Record one no-contact and one supervised-contact episode while rosbag2 runs separately. Verify timestamp freshness, causal action alignment, equal HDF5 column lengths, `ep_len`/`ep_offset`, `frameskip=1` loader shape, and 5 Hz model-view MP4.

- [ ] **Step 5: Final verification**

Run: `python -m pytest -q`, `python -m py_compile scripts/collector_core.py scripts/ros_camera.py scripts/pygame_input.py scripts/collect_keyboard_xy.py`, and `git diff --check`.

Expected: all automated checks pass; hardware commissioning results are recorded in the session notes; no live CEM action is executed before offline validation.
