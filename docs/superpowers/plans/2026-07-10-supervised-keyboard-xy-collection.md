# Supervised Keyboard XY Dataset Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a supervised, PushT-like data collector in which a human teleoperates the follower arm with bounded table-frame XY keyboard actions while the system enforces a fixed tool pose, records aligned episodes, detects manual-reset conditions, and reports coverage deficits.

**Architecture:** Keep `scripts/collect_data.py` unchanged as the manual setup-validation prototype. Add a new `pushbox_collect` package and a thin `scripts/collect_keyboard_xy.py` entry point. A single `FollowerRobot` adapter owns the hardware driver; the 5 Hz supervisor obtains an observation, asks the safety guardian to approve one XY action, records the observation and exact approved action, then sends that action.

**Tech Stack:** Python 3.12, NumPy, OpenCV/AprilTag detector adapter, RealSense, Trossen Arm Driver 1.9.x, h5py or a verified stable-worldmodel HDF5 writer, `unittest` with fake camera/robot implementations.

## Global Constraints

- Collection is supervised. The operator manually re-randomizes a box that reaches the visible-workspace boundary; the collector never searches for, retrieves, or repositions a lost box.
- In a recorded episode, the only controllable action is `action[t] = [delta_x_table, delta_y_table]` in metres, `float32`, with `norm(action[t]) <= 0.025`.
- Every recorded EE target is `T_base_tool(x_table, y_table, z_fixed, R_fixed)`. The pusher's Z, roll, pitch, yaw, and gripper state are fixed during an episode. Box XY and box yaw remain free.
- Record `pixels[t]`, `proprio[t]`, and `state[t]` before issuing `action[t]`; record the guardian-approved command at issue time, never a later measured leader displacement.
- No module other than `pushbox_collect.robot.FollowerRobot` may import or call `trossen_arm`.
- A physical E-stop and a supervised operator are mandatory. Keyboard input is not an E-stop and must never be treated as one.
- No recorded episode may contain an out-of-view box, stale/missing scene estimate, operator intervention, unhandled exception, timing overrun, or safety rejection. Those runs are audit artifacts and are discarded from training data.
- A terminal cannot reliably report a key-up event. V1 keyboard control is therefore non-latching: each received arrow/WASD event authorizes exactly one bounded 5 Hz tick. OS auto-repeat may generate more one-tick requests, but silence always means zero action.
- V1 has no hand-guided or gravity-compensation mode. The pusher is rigidly mounted in place of the gripper (not gripper-held), so the gripper is never load-bearing, and the EE is repositioned only by fixed-pose Cartesian position commands. Any XY-constrained or full-DOF hand guiding requires a separately commissioned high-rate Cartesian impedance/admittance controller and is explicitly out of V1 scope.
- The current macOS/Apple Silicon lock cannot install the pinned `pyrealsense2` wheel. Implement and commission on a supported robot host with a tested RealSense installation before connecting a live arm.
- Pin and validate the exact Trossen Python package/firmware pair on the robot host before enabling Cartesian motion. The public API is actively developed; do not rely on a `main`-branch signature.
- Preserve the user's existing `.gitignore` modification. Plan commits must stage only files named in their task.

## Locked Design Decisions

### Input and data modes

Movement keys are single-byte and non-latching, each authorizing exactly one bounded 5 Hz tick: `w` requests +table-Y, `s` requests -table-Y, `a` requests -table-X, and `d` requests +table-X. The guardian applies the global L2 cap. V1 has no diagonal keys and no arrow keys: arrow keys emit multi-byte escape sequences whose first byte collides with Escape, so V1 uses only single-byte keys and covers oblique approaches by staircasing X and Y over successive ticks. `Space` toggles only between `IDLE/PARKED_CLEAR` and `ARMED`/`RECORDING` after all prechecks succeed. `Backspace` discards the current episode. `x`/Escape requests immediate hold and enters `ABORTED`. `q` performs graceful shutdown only from `IDLE/PARKED_CLEAR`. There is no gravity-reset key.

The initial `keyboard_step_m` is 0.005 m, deliberately below the dataset's 0.025 m action cap for first bring-up. At 5 Hz this caps hand speed at 0.025 m/s — one-fifth of the action cap — which is slow for contact pushes, so commissioning is expected to raise it toward the cap after a no-contact test. Any change requires that test and a config version update.

### EE repositioning between episodes

V1 has no hand-guided reset. Because the pusher is rigidly mounted and the operator re-randomizes only the box by hand, the follower's start pose is re-randomized between episodes by a fixed-pose Cartesian position command (`FollowerRobot.move_to_start`) to a sampled XY inside the certified polygon, and parked clear by `FollowerRobot.park`. The arm is never back-driven and never enters external-effort mode.

The sole V1 way to guarantee XY-only motion is `RECORDING` or `ARMED` position control through fixed-pose Cartesian targets. A future hand-guided `RESET_XY_GUIDE` mode is a separate project only after a 1 kHz Cartesian compliance controller, force limits, watchdog, and hardware validation exist.

### Scene policy

A fixed table fiducial establishes the table frame; a box fiducial yields `(box_x, box_y, box_yaw)` and confidence. A scene is usable only when both tags are fresh, the box polygon lies within the calibrated visible polygon with its required margin, and the fixed tool pose is reachable. The box may be briefly occluded by the tool only if the tag tracker remains fresh and confident; a frame-border clip is always an abort in V1.

### Contact and coverage policy

The core world model is trained from pixels and actions; contact is not a required training label. Store an audit-only contact confidence derived from geometry, EE external effort, and observed box motion. Coverage reports must separately count free-space, approach, contact, and release steps; box XY/yaw bins; EE XY bins; contacted face/corner; push direction; action magnitude; observed translation; and observed rotation. The operator uses the report to choose the next motion. The collector does not invent scripted autonomous pushes in V1.

### State machine

```text
PRECHECK -> IDLE/PARKED_CLEAR -> ARMED -> RECORDING -> COMPLETE -> IDLE/PARKED_CLEAR
                                   |          |
                                   |          +-> ABORTED --------> IDLE/PARKED_CLEAR
                                   |          +-> RESET_REQUIRED -> IDLE/PARKED_CLEAR
                                   |
                                   +-> HARD_FAULT -> HOLD_OR_IDLE
```

- `PRECHECK`: validate config schema, calibration IDs, target hardware/API contract, camera settings, safe XY polygon, and storage destination.
- `IDLE/PARKED_CLEAR`: follower holds a hardware-validated parked pose with the pusher clear of the table and human work area; no recording. Between episodes the follower re-randomizes its start pose here via a fixed-pose Cartesian position command.
- `ARMED`: both tags are stable and the current box/tool geometry is safe; first valid movement request starts an episode.
- `RECORDING`: 5 Hz observation/approval/record/command tick.
- `COMPLETE`: a valid operator-requested end; commit the buffered episode only if it meets all acceptance checks.
- `ABORTED`: hold immediately, keep an audit record, and discard the buffered training episode.
- `RESET_REQUIRED`: box is out of view, too near a boundary, or manually moved; hold, discard, prompt manual reset, then require stable tag reacquisition and explicit re-arm.
- `HARD_FAULT`: driver/camera/control failure. Execute only the hardware-validated hold/idle sequence; never automatically home.

## Planned File Structure

```text
pushbox_collect/
  __init__.py                 # Package marker and public version.
  config.py                   # Validated TOML configuration and calibration IDs.
  types.py                    # Immutable typed data exchanged between modules.
  geometry.py                 # Table/base transforms, L2 projection, polygons, swept-path checks.
  keyboard.py                 # Non-latching terminal events and key-to-one-tick requests.
  robot.py                    # The sole Trossen driver owner and Cartesian follower adapter.
  camera.py                   # Sole RealSense owner: locked exposure/WB/focus/gain, 224x224 crop, pixels.
  vision.py                   # Table/box tag estimates, freshness, visibility margin checks.
  safety.py                   # Action approval/rejection, watchdog, and safe hold requests.
  coverage.py                 # Observed-event bins and operator-facing deficit summary.
  storage.py                  # Buffered episode validation, writer adapter, audit JSONL sidecar.
  supervisor.py               # 5 Hz state machine and aligned observation/action sequencing.
  qa.py                       # Offline dataset validation and report generation.
config/
  .gitignore
  pushbox_collection.example.toml
scripts/
  collect_keyboard_xy.py
  calibrate_table.py
  qa_dataset.py
tests/
  test_config.py
  test_geometry.py
  test_keyboard.py
  test_robot.py
  test_camera.py
  test_vision.py
  test_safety.py
  test_coverage.py
  test_storage.py
  test_cli.py
  test_supervisor.py
```

`config/pushbox_collection.local.toml` is an operator-owned copy of the example file and is ignored by `config/.gitignore`. It contains IPs and calibrated transforms; the example contains no real credentials or rig-specific numbers.

## Interfaces Shared by All Tasks

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass(frozen=True)
class ToolPose:
    translation_base_m: np.ndarray  # shape (3,), float64
    angle_axis_base_rad: np.ndarray  # shape (3,), float64

@dataclass(frozen=True)
class XYAction:
    delta_table_m: np.ndarray  # shape (2,), float32
    source: str                # "keyboard" or "zero"

@dataclass(frozen=True)
class SceneEstimate:
    timestamp_s: float
    box_xyyaw_table: np.ndarray  # shape (3,), float64
    table_tag_visible: bool
    box_tag_visible: bool
    confidence: float
    age_s: float
    fully_visible: bool

@dataclass(frozen=True)
class RobotSnapshot:
    timestamp_s: float
    tool_pose_base: ToolPose
    tool_velocity_base: np.ndarray  # shape (6,), float64
    external_effort_base: np.ndarray  # shape (6,), float64

@dataclass(frozen=True)
class CameraFrame:
    timestamp_s: float
    pixels: np.ndarray  # shape (224, 224, 3), uint8

class RunState(str, Enum):
    PRECHECK = "precheck"
    IDLE = "idle"
    ARMED = "armed"
    RECORDING = "recording"
    COMPLETE = "complete"
    ABORTED = "aborted"
    RESET_REQUIRED = "reset_required"
    HARD_FAULT = "hard_fault"
```

### Build and test conventions

- Split `pyproject.toml` so hardware-only wheels are optional: keep `numpy`, `h5py`, and TOML libraries in the core/dev set, and move `pyrealsense2` and `trossen-arm` into a `hardware` optional group. The hardware-free `unittest` suite then installs and runs on the macOS dev host (where the pinned `pyrealsense2` wheel will not install); the robot host installs `.[hardware]`.
- Tests use `unittest` (`python -m unittest`). The code shown in each task is illustrative; write real tests as `unittest.TestCase` methods using `tempfile`/`setUp` for temp paths, not pytest fixtures such as `tmp_path`.
- Reading TOML uses the stdlib `tomllib`; writing it (e.g. `calibrate_table.py`) needs `tomli-w`, which belongs in the core/dev dependency set.

---

### Task 1: Establish an importable package, config contract, and hardware-free test harness

**Files:**
- Create: `pushbox_collect/__init__.py`
- Create: `pushbox_collect/types.py`
- Create: `pushbox_collect/config.py`
- Create: `config/pushbox_collection.example.toml`
- Create: `config/.gitignore`
- Create: `tests/test_config.py`
- Modify: `pyproject.toml` (split hardware-only deps into an optional `hardware` group; see Build and test conventions)

**Interfaces:**
- Produces `CollectionConfig.load(path: Path) -> CollectionConfig`.
- Produces the shared types above for all later tasks.

- [ ] **Step 1: Write the failing configuration tests**

```python
def test_config_rejects_a_keyboard_step_larger_than_action_cap(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[control]\ntick_s = 0.2\nmax_action_norm_m = 0.025\nkeyboard_step_m = 0.03\n")
    with self.assertRaisesRegex(ValueError, "keyboard_step_m"):
        CollectionConfig.load(path)

def test_config_requires_fixed_pose_and_visibility_margin(tmp_path):
    path = tmp_path / "missing.toml"
    path.write_text("[control]\ntick_s = 0.2\nmax_action_norm_m = 0.025\nkeyboard_step_m = 0.005\n")
    with self.assertRaisesRegex(ValueError, "fixed_tool_pose"):
        CollectionConfig.load(path)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_config -v`

Expected: FAIL because `pushbox_collect.config` does not exist.

- [ ] **Step 3: Implement immutable config validation**

Implement `CollectionConfig` with these required TOML fields: `control.tick_s`, `control.max_action_norm_m`, `control.keyboard_step_m`, `tool.fixed_translation_table_m`, `tool.fixed_angle_axis_table_rad`, `workspace.safe_polygon_table_m`, `workspace.box_visibility_margin_m`, `vision.max_age_s`, `vision.min_confidence`, `storage.dataset_path`, and `storage.audit_path`. Reject nonpositive values, a tick other than `0.2`, a step larger than the action cap, and any safe polygon with fewer than three points.

Use this example shape:

```toml
[control]
tick_s = 0.2
max_action_norm_m = 0.025
keyboard_step_m = 0.005

[tool]
fixed_translation_table_m = [0.0, 0.0, 0.0]
fixed_angle_axis_table_rad = [0.0, 0.0, 0.0]

[workspace]
safe_polygon_table_m = [[-0.10, -0.10], [0.10, -0.10], [0.10, 0.10], [-0.10, 0.10]]
box_visibility_margin_m = 0.03

[vision]
max_age_s = 0.10
min_confidence = 0.90

[storage]
dataset_path = "data/pushbox.h5"
audit_path = "data/pushbox_audit.jsonl"
```

The numerical polygon and pose in the example are sentinel schema values, not a usable rig calibration. `collect_keyboard_xy.py` must refuse to connect to hardware while `config/pushbox_collection.local.toml` still contains those values.

- [ ] **Step 4: Make local calibration files untracked without touching the dirty root ignore file**

Create `config/.gitignore` with exactly this entry:

```gitignore
/pushbox_collection.local.toml
```

- [ ] **Step 5: Run the tests to verify success**

Run: `python -m unittest tests.test_config -v`

Expected: PASS with both validation tests green.

- [ ] **Step 6: Commit only this task's files**

```bash
git add pushbox_collect/__init__.py pushbox_collect/types.py pushbox_collect/config.py \
  config/pushbox_collection.example.toml config/.gitignore tests/test_config.py
git commit -m "feat: add collection configuration contract"
```

### Task 2: Implement table geometry and the planar action guardian

**Files:**
- Create: `pushbox_collect/geometry.py`
- Create: `pushbox_collect/safety.py`
- Create: `tests/test_geometry.py`
- Create: `tests/test_safety.py`

**Interfaces:**
- Consumes `CollectionConfig`, `ToolPose`, `XYAction`, `SceneEstimate`, and `RobotSnapshot`.
- Produces `project_l2(delta, max_norm)`, `table_delta_to_base(delta)`, and `SafetyGuardian.approve(...) -> XYAction` or `SafetyRejection`.

- [ ] **Step 1: Write the failing geometry and guardian tests**

```python
def test_l2_projection_preserves_direction_and_caps_diagonal():
    actual = project_l2(np.array([0.025, 0.025]), 0.025)
    np.testing.assert_allclose(actual, np.array([0.025, 0.025]) / np.sqrt(2), atol=1e-7)
    self.assertAlmostEqual(float(np.linalg.norm(actual)), 0.025)

def test_guard_rejects_stale_or_partially_visible_box():
    result = guardian.approve(request, snapshot, stale_scene, now_s=10.0)
    self.assertEqual(result.reason, "scene_stale")
    result = guardian.approve(request, snapshot, clipped_scene, now_s=10.0)
    self.assertEqual(result.reason, "box_not_fully_visible")

def test_guard_holds_fixed_z_and_orientation_when_approving_xy():
    approved = guardian.approve(request, snapshot, good_scene, now_s=10.0)
    target = guardian.target_tool_pose(snapshot.tool_pose_base, approved)
    self.assertEqual(target.translation_base_m[2], snapshot.tool_pose_base.translation_base_m[2])
    np.testing.assert_allclose(target.angle_axis_base_rad, config.fixed_tool_angle_axis_base_rad)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_geometry tests.test_safety -v`

Expected: FAIL because geometry and safety modules do not exist.

- [ ] **Step 3: Implement the guard as a pure, hardware-free component**

`project_l2` must return the original vector when its L2 norm is already within the cap and otherwise return `delta * max_norm / norm`. `SafetyGuardian.approve` must reject: non-finite inputs, unsafe state, stale or low-confidence tags, a box outside the visibility margin, a target outside the certified XY polygon, an invalid table/base transform, an action whose sweep crosses a no-go polygon, and any action requested outside `ARMED` or `RECORDING`.

The target pose must be computed from the fixed table tool pose plus the approved XY delta. It must not copy leader joints or derive an action from later EE measurements.

- [ ] **Step 4: Run the tests to verify success**

Run: `python -m unittest tests.test_geometry tests.test_safety -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pushbox_collect/geometry.py pushbox_collect/safety.py tests/test_geometry.py tests/test_safety.py
git commit -m "feat: guard table-frame XY actions"
```

### Task 3: Implement non-latching keyboard requests and operator commands

**Files:**
- Create: `pushbox_collect/keyboard.py`
- Create: `tests/test_keyboard.py`

**Interfaces:**
- Produces `KeyboardEvent(kind: str, delta_table_m: np.ndarray | None)`.
- Produces `KeyboardMapper.event_for_character(char: str) -> KeyboardEvent`.

- [ ] **Step 1: Write the failing input tests**

```python
def test_w_requests_one_positive_y_tick():
    event = mapper.event_for_character("w")
    self.assertEqual(event.kind, "move")
    np.testing.assert_allclose(event.delta_table_m, [0.0, 0.005])

def test_d_requests_one_positive_x_tick():
    event = mapper.event_for_character("d")
    self.assertEqual(event.kind, "move")
    np.testing.assert_allclose(event.delta_table_m, [0.005, 0.0])

def test_backspace_discards_and_is_not_a_motion_command():
    self.assertEqual(mapper.event_for_character("\x7f").kind, "discard")

def test_no_character_means_no_motion_request():
    self.assertEqual(mapper.event_for_character(""), KeyboardEvent(kind="none", delta_table_m=None))

def test_escape_is_an_abort_not_a_latched_motion_command():
    self.assertEqual(mapper.event_for_character("\x1b").kind, "abort")
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_keyboard -v`

Expected: FAIL because `KeyboardMapper` does not exist.

- [ ] **Step 3: Implement one-event/one-tick behavior**

Map `w`, `a`, `s`, `d` to individual +Y/-X/-Y/+X movement events. Map `Space` to arm/end, `Backspace` (`\x7f`) to discard, `x` and Escape (`\x1b`) to abort, and `q` to shutdown request. There is no gravity-reset key and no arrow-key or diagonal handling in V1 — all keys are single-byte, so `event_for_character(char)` is sufficient. Do not retain a motion vector after an event has been consumed; a supervisor tick with no movement event must receive `XYAction([0, 0], "zero")`.

Document in the module docstring that terminal auto-repeat is convenience only, never a deadman guarantee, and that arrow keys are intentionally unsupported to avoid the Escape/escape-sequence collision.

- [ ] **Step 4: Run the tests to verify success**

Run: `python -m unittest tests.test_keyboard -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pushbox_collect/keyboard.py tests/test_keyboard.py
git commit -m "feat: add non-latching keyboard control"
```

### Task 4: Add a fake follower and a sole-owner Cartesian follower adapter

**Files:**
- Create: `pushbox_collect/robot.py`
- Create: `tests/test_robot.py`

**Interfaces:**
- Produces `FollowerRobot.snapshot() -> RobotSnapshot`.
- Produces `FollowerRobot.command_xy(approved: XYAction) -> ToolPose`.
- Produces `FollowerRobot.hold() -> None`, `park() -> None`, and `move_to_start(xy_table_m: np.ndarray) -> ToolPose`.
- `FollowerRobot` is the sole module allowed to import `trossen_arm`.

- [ ] **Step 1: Write the fake-driver contract tests**

```python
def test_command_xy_sends_a_six_dof_cartesian_target_with_fixed_pose():
    robot = FollowerRobot(fake_driver, config, transforms)
    robot.command_xy(XYAction(np.array([0.005, 0.0], np.float32), "keyboard"))
    goal, interpolation, goal_time, blocking = fake_driver.cartesian_calls[-1]
    self.assertEqual(len(goal), 6)
    self.assertEqual(interpolation, "cartesian")
    self.assertEqual(goal_time, 0.2)
    self.assertFalse(blocking)

def test_no_leader_interface_is_constructed():
    source = Path("pushbox_collect/robot.py").read_text()
    self.assertNotIn("wxai_v0_leader", source)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_robot -v`

Expected: FAIL because `FollowerRobot` does not exist.

- [ ] **Step 3: Implement the adapter against a narrow driver protocol**

Define a testable `FollowerDriverProtocol` containing only `get_cartesian_positions`, `get_cartesian_velocities`, `get_cartesian_external_efforts`, `set_all_modes`, `set_cartesian_positions`, and `get_num_joints`. Inject it into `FollowerRobot`; do not construct a driver in business logic. V1 never commands external effort, so the protocol excludes `set_all_external_efforts`; `park`/`move_to_start` use position mode with `set_cartesian_positions` to a fixed-pose target.

For actual collection, configure only `wxai_v0_follower`; park the leader and do not connect it. `command_xy` must transform the guardian-approved table-frame action into a six-element base-frame target with fixed Z and angle-axis orientation, use Cartesian interpolation, a 0.2 s goal time, and nonblocking execution. The exact Python binding signature, trajectory-check setting, and safe mode transition must be confirmed by the hardware commissioning gate before this live call is enabled.

- [ ] **Step 4: Add a hardware-free API contract command**

Run: `uv run python -c "import trossen_arm; print(trossen_arm.__version__)"`

Expected: prints the pinned driver version on the Linux robot host. Do not run this command on the current macOS environment until its dependency installation is repaired.

- [ ] **Step 5: Run the fake-driver tests**

Run: `python -m unittest tests.test_robot -v`

Expected: PASS with no network, camera, or arm interaction.

- [ ] **Step 6: Commit**

```bash
git add pushbox_collect/robot.py tests/test_robot.py
git commit -m "feat: add Cartesian follower adapter"
```

### Task 5: Add the camera adapter and Cartesian EE parking/re-randomization

**Files:**
- Create: `pushbox_collect/camera.py`
- Create: `tests/test_camera.py`
- Modify: `pushbox_collect/robot.py`
- Modify: `pushbox_collect/config.py`

**Interfaces:**
- Produces `CameraSource.frame() -> CameraFrame` (locked-settings RGB plus capture timestamp) and `CameraSource.lock_settings()`.
- Produces `FollowerRobot.park() -> None` and `FollowerRobot.move_to_start(xy_table_m) -> ToolPose`.
- `CameraSource` is the sole owner of the RealSense pipeline.

- [ ] **Step 1: Write the failing camera and parking tests**

```python
def test_camera_locks_exposure_white_balance_focus_and_gain():
    cam = CameraSource(fake_rs, config)
    cam.lock_settings()
    self.assertFalse(fake_rs.auto_exposure)
    self.assertFalse(fake_rs.auto_white_balance)

def test_frame_is_center_cropped_and_resized_to_the_spec_shape():
    frame = CameraSource(fake_rs, config).frame()
    self.assertEqual(frame.pixels.shape, (224, 224, 3))
    self.assertEqual(frame.pixels.dtype, np.uint8)

def test_move_to_start_sends_a_fixed_pose_cartesian_target_not_effort():
    robot = FollowerRobot(fake_driver, config, transforms)
    robot.move_to_start(np.array([0.02, -0.03], np.float64))
    goal, interpolation, goal_time, blocking = fake_driver.cartesian_calls[-1]
    self.assertEqual(len(goal), 6)
    self.assertEqual(fake_driver.mode, "position")
    self.assertEqual(fake_driver.external_effort_calls, [])
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_camera -v`

Expected: FAIL because the camera adapter does not exist.

- [ ] **Step 3: Implement the camera adapter and Cartesian parking**

`CameraSource` owns the RealSense pipeline behind a narrow injectable protocol so tests use a fake. `lock_settings` must disable auto-exposure, auto-white-balance, autofocus, and auto-gain and set the fixed values from config (spec §1); `frame` returns a center-square-cropped 224x224 uint8 RGB image plus the camera timestamp, using the exact crop/resize the prototype already uses in `process_frame`. `PRECHECK` calls `lock_settings` and refuses to record if any auto-setting is still on.

`FollowerRobot.park` commands the configured clear parked pose; `move_to_start` accepts an XY inside the certified polygon and commands a fixed-Z/fixed-orientation Cartesian target in position mode. Neither uses external-effort mode. Add `camera.*` fields (device serial, resolution, and the locked exposure/WB/focus/gain values) to the config schema.

- [ ] **Step 4: Run the tests to verify success**

Run: `python -m unittest tests.test_camera -v`

Expected: PASS with no camera or arm interaction.

- [ ] **Step 5: Commit**

```bash
git add pushbox_collect/camera.py pushbox_collect/robot.py pushbox_collect/config.py tests/test_camera.py
git commit -m "feat: add locked-settings camera adapter and Cartesian EE parking"
```

### Task 6: Add fresh tag-based scene estimates and visibility gating

**Files:**
- Create: `pushbox_collect/vision.py`
- Create: `scripts/calibrate_table.py`
- Create: `tests/test_vision.py`
- Modify: `pushbox_collect/config.py`

**Interfaces:**
- Produces `SceneProvider.latest() -> SceneEstimate`.
- Produces `is_fully_visible(box_polygon_table, visible_polygon_table, margin_m) -> bool`.
- `scripts/calibrate_table.py` writes only a new local calibration file after an explicit operator confirmation.

- [ ] **Step 1: Write the failing scene-gate tests**

```python
def test_scene_requires_both_table_and_box_tags():
    scene = provider.from_detections(table=None, box=box_detection, now_s=1.0)
    self.assertFalse(scene.table_tag_visible)
    self.assertFalse(scene.fully_visible)

def test_box_near_the_frame_margin_is_not_fully_visible():
    scene = make_scene(box_xyyaw=[0.095, 0.0, 0.0], fully_visible=False)
    self.assertFalse(scene.fully_visible)

def test_scene_age_is_measured_from_the_camera_timestamp():
    scene = make_scene(timestamp_s=9.8)
    self.assertAlmostEqual(scene.age_s, 0.2)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_vision -v`

Expected: FAIL because the scene provider does not exist.

- [ ] **Step 3: Implement detector-independent vision plumbing**

Define a `TagDetectorProtocol` that returns table-tag and box-tag detections with camera timestamps and confidence. Convert both poses into the table frame, apply the calibrated table-to-camera visibility polygon, and calculate a confidence/age-bearing `SceneEstimate`. Use an actual AprilTag detector only behind this protocol so every safety case remains testable with recorded detections.

`calibrate_table.py` must show the measured reprojection and table/base residuals, require the operator to type the exact calibration ID to confirm saving, and refuse to overwrite an existing local calibration without a timestamped backup.

- [ ] **Step 4: Run the tests to verify success**

Run: `python -m unittest tests.test_vision -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pushbox_collect/vision.py scripts/calibrate_table.py tests/test_vision.py pushbox_collect/config.py
git commit -m "feat: gate collection on fresh visible tags"
```

### Task 7: Build the 5 Hz supervisor and aligned recording state machine

**Files:**
- Create: `pushbox_collect/supervisor.py`
- Create: `tests/test_supervisor.py`

**Interfaces:**
- Produces `CollectionSupervisor.tick(now_s: float) -> RunState`.
- Consumes a `CameraSource`, `SceneProvider`, `FollowerRobot`, `KeyboardMapper`, `SafetyGuardian`, `EpisodeBuffer`, and `CoverageTracker`.

- [ ] **Step 1: Write the failing state/alignment tests**

```python
def test_recording_tick_observes_records_then_commands():
    supervisor.state = RunState.RECORDING
    supervisor.tick(now_s=1.0)
    self.assertEqual(log.events, ["camera_snapshot", "robot_snapshot", "scene_snapshot", "buffer_append", "robot_command"])

def test_lost_visibility_holds_discards_and_requires_manual_reset():
    supervisor.state = RunState.RECORDING
    scene_provider.latest_return = clipped_scene
    state = supervisor.tick(now_s=1.0)
    self.assertEqual(state, RunState.RESET_REQUIRED)
    self.assertTrue(fake_robot.hold_called)
    self.assertTrue(buffer.discarded)

def test_tick_overrun_aborts_instead_of_silently_resynchronizing():
    state = supervisor.tick(now_s=1.41)  # previous deadline was 1.20
    self.assertEqual(state, RunState.ABORTED)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_supervisor -v`

Expected: FAIL because the supervisor does not exist.

- [ ] **Step 3: Implement the explicit state machine**

Schedule the supervisor at 5 Hz with a monotonic clock. In `RECORDING`, acquire a fresh camera frame, atomic robot snapshot, and scene estimate; reject the tick if their configured age/skew bounds are violated; append the observation with an initially zero action slot; map one keyboard event to a requested action; obtain guardian approval; replace the action slot with the exact approved action; then command the follower.

No command may be issued after a timing overrun. `x`/Escape, driver exception, stale scene, invalid transform, force alarm, or visibility failure must call `hold()`, discard the episode buffer, write an audit entry, and move to `ABORTED` or `RESET_REQUIRED` as appropriate. On `COMPLETE` and after any reset, re-randomize the follower start pose via `FollowerRobot.move_to_start` before re-arming; the supervisor never enters external-effort mode.

- [ ] **Step 4: Run the tests to verify success**

Run: `python -m unittest tests.test_supervisor -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pushbox_collect/supervisor.py tests/test_supervisor.py
git commit -m "feat: add supervised 5Hz collection state machine"
```

### Task 8: Add validated episode storage, audit records, and coverage feedback

**Files:**
- Create: `pushbox_collect/storage.py`
- Create: `pushbox_collect/coverage.py`
- Create: `pushbox_collect/qa.py`
- Create: `scripts/qa_dataset.py`
- Create: `tests/test_storage.py`
- Create: `tests/test_coverage.py`

**Interfaces:**
- Produces `EpisodeBuffer.append(observation, action)`, `EpisodeBuffer.validate()`, and `EpisodeStore.commit(buffer)`.
- Produces `CoverageTracker.update(observed_step)` and `CoverageTracker.summary() -> dict`.

- [ ] **Step 1: Write the failing storage and coverage tests**

```python
def test_valid_episode_contains_exactly_aligned_columns():
    buffer = populated_valid_buffer(steps=50)
    columns = buffer.validate()
    self.assertEqual(columns["pixels"].shape[0], columns["action"].shape[0])
    self.assertEqual(columns["action"].dtype, np.float32)
    self.assertLessEqual(np.linalg.norm(columns["action"], axis=1).max(), 0.025 + 1e-7)

def test_short_or_aborted_episode_cannot_be_committed():
    with self.assertRaisesRegex(ValueError, "minimum episode length"):
        EpisodeStore(fake_writer).commit(populated_valid_buffer(steps=49))
    with self.assertRaisesRegex(ValueError, "aborted"):
        EpisodeStore(fake_writer).commit(aborted_buffer)

def test_coverage_counts_camera_side_pushes_as_a_box_face_bin():
    tracker.update(observed_camera_side_push)
    self.assertEqual(tracker.summary()["face_bins"]["camera_side"], 1)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_storage tests.test_coverage -v`

Expected: FAIL because storage and coverage modules do not exist.

- [ ] **Step 3: Implement buffered, validated storage**

Store the core columns `pixels`, `action`, `proprio`, `state`, `episode_idx`, `step_idx`, and `timestamp` with the shapes/dtypes from `dataset_spec.md`. Add only calibration ID, reset/abort reason, contact confidence, coverage bins, camera timestamps, and diagnostics to a JSONL audit sidecar unless the verified downstream writer explicitly supports extra HDF5 columns.

`EpisodeStore` must use the downstream-compatible writer selected by a dedicated integration test against the exact stable-worldmodel/LeWM training environment. That test must prove correct episode boundaries and a loader smoke test before a live dataset is accepted. Do not reimplement an assumed `ep_len`/`ep_offset` layout from memory.

Coverage uses observed scene/robot data, not user intent. Its summary must report: free/approach/contact/release step counts; box XY/yaw occupancy; EE XY occupancy; push direction; contacted face/corner; action norm histogram; box translation; box rotation; and unfilled bins.

- [ ] **Step 4: Run the tests to verify success**

Run: `python -m unittest tests.test_storage tests.test_coverage -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pushbox_collect/storage.py pushbox_collect/coverage.py pushbox_collect/qa.py \
  scripts/qa_dataset.py tests/test_storage.py tests/test_coverage.py
git commit -m "feat: validate episodes and report collection coverage"
```

### Task 9: Wire the command-line entry point and operator workflow

**Files:**
- Create: `scripts/collect_keyboard_xy.py`
- Modify: `README.md`
- Modify: `dataset_spec.md`

**Interfaces:**
- CLI: `uv run python scripts/collect_keyboard_xy.py --config config/pushbox_collection.local.toml --output data/run.h5`.
- Exit status `0` only after clean shutdown from `IDLE`; nonzero after an unhandled hardware or configuration fault.

- [ ] **Step 1: Write a CLI parsing test**

```python
def test_cli_requires_an_explicit_local_config_and_output():
    with self.assertRaises(SystemExit) as error:
        parse_args([])
    self.assertEqual(error.exception.code, 2)
```

- [ ] **Step 2: Run the test to verify failure**

Run: `python -m unittest tests.test_cli -v`

Expected: FAIL because `parse_args` does not exist.

- [ ] **Step 3: Implement the safe entry point and docs**

The script must require an explicit local config and output path, call `PRECHECK` (including the camera settings-lock) before constructing an arm command loop, print the current run state and coverage deficit after each episode, and restore terminal settings in a `finally` block. It must not start an episode until table/box tags and fixed-pose workspace checks pass.

Update README with the V1 keyboard map (`w`/`a`/`s`/`d` move, `Space` arm/end, `Backspace` discard, `x`/Esc abort, `q` quit; no gravity key), physical E-stop requirement, manual box-reset/out-of-view behavior, non-latching input semantics, and the rigid-pusher/position-only repositioning policy (no hand-guided mode). Fix the post-hoc leader-delta action description in the `collect_data.py` prototype docstring, and tighten `dataset_spec.md` §3 to the exact sent table-frame action contract, labeling the old collector a prototype.

- [ ] **Step 4: Run the test to verify success**

Run: `python -m unittest tests.test_cli -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/collect_keyboard_xy.py README.md dataset_spec.md tests/test_cli.py
git commit -m "feat: add keyboard XY collection workflow"
```

### Task 10: Commission in ascending-risk stages and gate dataset acceptance

**Files:**
- Modify: `README.md`
- Modify: `dataset_spec.md`
- Add: `docs/commissioning/supervised_keyboard_xy.md`

**Interfaces:**
- Produces a signed-off local commissioning report containing calibration IDs, driver/firmware versions, workspace polygon, test results, and operator name/date.

- [ ] **Step 1: Add the commissioning checklist**

Document the following mandatory order:

1. Run all `unittest` tests without hardware.
2. Verify the target Linux environment imports `cv2`, `pyrealsense2`, and the pinned `trossen_arm` version.
3. With the tool raised and no box on the table, verify the pusher is rigidly mounted (not gripper-held), then test `hold`, `park`, `move_to_start` re-randomization inside the polygon, the camera settings-lock, and the physical E-stop. Confirm the arm never enters external-effort mode.
4. With the tool still raised, test zero, +X, -X, +Y, -Y, and diagonal Cartesian commands inside the certified polygon. Reject any pose with an IK/singularity or command-tracking failure.
5. Calibrate camera/table/base transforms; test tag freshness, box visibility margin, and manual-reset abort when the box reaches a boundary.
6. Lower to the fixed contact height; run contact-free keyboard trajectories only. Verify 5 Hz timestamp spacing, action/proprio alignment, and zero box motion.
7. Run one slow, supervised contact push from each box face, including the camera-facing face that moves the box toward the arm. Verify pose, video, audit data, and force/contact proxy before collecting longer episodes.
8. Collect capped pilot episodes, run `scripts/qa_dataset.py`, inspect coverage deficits, and accept only datasets meeting the documented timing, visibility, length, and loader checks.

- [ ] **Step 2: Add a machine-checkable QA command**

Run: `uv run python scripts/qa_dataset.py --dataset data/run.h5 --audit data/run_audit.jsonl`

Expected: reports timing, episode lengths, tag health, full-visibility rate, action-norm maximum, action/proprio alignment, and coverage bins; exits nonzero on any failed gate.

- [ ] **Step 3: Commit**

```bash
git add README.md dataset_spec.md docs/commissioning/supervised_keyboard_xy.md
git commit -m "docs: add supervised collector commissioning gates"
```

## Plan Self-Review

- **Spec coverage:** The plan covers keyboard-based Cartesian XY control, fixed Z/orientation, a locked-settings camera, table/box tracking, full-visibility aborts, manual box resets, free/contact coverage, exact-action recording, HDF5 validation, position-only EE re-randomization, and staged commissioning. It deliberately excludes autonomous object recovery and all hand-guided/gravity-compensation modes because neither is safe to infer from the current driver loop.
- **Safety boundary:** V1 has no hand-guided or external-effort mode; the arm is only ever in position control, and the rigidly-mounted pusher means the gripper is never load-bearing. The only V1 XY constraint is fixed-pose Cartesian position control, and repositioning between episodes is a position command, never back-driving.
- **Placeholder scan:** No task asks an implementer to guess a driver signature, a force threshold, a calibration, or a downstream HDF5 schema. Those are explicit test/commissioning gates before live collection.
- **Type consistency:** All task interfaces use `XYAction`, `SceneEstimate`, `RobotSnapshot`, `ToolPose`, and `RunState` defined near the beginning of this document.

## References Used for the Design

- [Trossen Arm Driver v1.9 API](https://docs.trossenrobotics.com/trossen_arm/v1.9/api/classtrossen__arm_1_1TrossenArmDriver.html): Cartesian position commands use a six-element translation/angle-axis target in the base frame.
- [Trossen configuration guidance](https://docs.trossenrobotics.com/trossen_arm/v1.9/getting_started/configuration.html): zero desired external effort invokes nominal model-based gravity/friction compensation; Cartesian motion may fault near singularities.
- [Trossen v1.9.3 gravity-compensation demo](https://github.com/TrossenRobotics/trossen_arm/blob/v1.9.3/demos/python/gravity_compensation.py): `set_all_modes(external_effort)` + `set_all_external_efforts([0]*7, 0.0, False)` back-drives all joints including the gripper (the scalar arg is goal_time, not a gripper effort). Evaluated for a reset mode and dropped from V1 because it cannot hold the gripper or enforce XY-only guiding.
- [Trossen demo index](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/demo_scripts.html): gravity compensation, Cartesian velocity, and Cartesian external-effort controls are distinct demos.
- [WorldPlanner](/Users/benwang/Desktop/vscode_workspace/wm_data_collection/papers/worldplanner.pdf) and [Global/Local WM](/Users/benwang/Desktop/vscode_workspace/wm_data_collection/papers/global_local_wm.pdf): real-robot high-entropy play and 5 Hz planar tool actions motivate supervised keyboard collection.
- [SkyJEPA](/Users/benwang/Desktop/vscode_workspace/wm_data_collection/papers/skyJEPA.pdf): clustered state-action coverage and transition richness motivate the coverage dashboard.
