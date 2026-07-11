# Minimal Keyboard XY Data Collector — Code Spec

## Goal

Add one new script, `scripts/collect_keyboard_xy.py`, for supervised push-box
data collection. The operator uses held arrow-key combinations in one focused
local Pygame preview/control window. The controller changes only the two
coordinates parallel to the table while it holds tool height and orientation
fixed.

The recording behavior should stay close to `scripts/collect_data.py`: one
RealSense color stream received from ROS 2 Jazzy, 5 Hz samples, episode
buffering, HDF5 output, a 5 Hz model-view MP4 per episode, and a separate
native-rate rosbag2 archive. The new collector never opens the camera through
`pyrealsense2`.

This is intentionally a single-script MVP. Do not add a detector, AprilTags,
autonomous pushing, automatic box reset, coverage tracking, a new Python
package, or gravity-compensation mode.

The collector is local-display only. SSH/headless and terminal escape-sequence
input are explicitly out of scope.

The arm reset is also planar: while idle, `r` returns the tool to the configured
collection start X/Y while preserving the same fixed Z and orientation. This
provides an XY-only reset without unlocking the arm's other degrees of freedom.

## 1. Coordinate convention

Trossen Cartesian poses are expressed in the **robot base frame**:

```text
pose = [base_x, base_y, base_z, angle_axis_x, angle_axis_y, angle_axis_z]
```

For the standard WXAI base convention:

| Base coordinate | Positive direction | Use here |
|---|---|---|
| `x` | forward | table-plane motion |
| `y` | left | table-plane motion |
| `z` | up | fixed height |

Therefore the commanded plane is **base X/Y**, not Y/Z. Tool orientation does
not rotate these translation coordinates.

Use the fixed angle-axis orientation:

```python
FIXED_ORIENTATION = np.array([0.0, np.pi / 2.0, 0.0])
```

This is a +90° right-hand rotation about base `+Y` and gives:

```text
tool +X -> base -Z  (down)
tool +Y -> base +Y  (left)
tool +Z -> base +X  (forward)
```

The complete Cartesian target on every command is consequently:

```python
[target_base_x, target_base_y, FIXED_BASE_Z, 0.0, np.pi / 2.0, 0.0]
```

### Height and the reported 5 cm offset

Height remains `base_z`, even though tool `+X` points down. A reported 5 cm
measurement along tool X is a **tool-frame offset**, not
`base_x = -0.05`. Before entering it anywhere, record which two physical points
the 5 cm was measured between and which direction is positive.

The preferred commanded frame is the pusher contact point. Configure the
driver's tool frame there and document that choice. Then:

```text
FIXED_BASE_Z = measured_table_base_z + desired_tip_height_above_table
```

If `t_flange_tool` is identity (the configured tool frame coincides with the
flange) and the pusher tip is 5 cm along tool `+X`, tool `+X` points downward,
so:

```text
flange_base_z = desired_tip_base_z + 0.05 m
```

This flange formula applies only if the measured 5 cm really is the
flange-to-tip distance in tool `+X`. The actual endpoints and sign must be
checked against the mounted tool and its Trossen `t_flange_tool` configuration
before contact. Never compensate for tool length by changing base-frame X.

## 2. Scope and file structure

Primary implementation files:

```text
scripts/collect_keyboard_xy.py
tests/test_collect_keyboard_xy.py
```

`stable-worldmodel==0.1.1` and `pygame` are declared in `pyproject.toml` and
pinned in `uv.lock`. The robot host must use that environment (plus its ROS 2
Jazzy installation); do not silently fall back to a different HDF5 layout.

Add `pygame` as the local preview/key-state dependency. Do not use a global
keyboard hook or read `/dev/input`; the focused Pygame window owns all motion
and lifecycle input.

Keep `scripts/collect_data.py` unchanged as the leader/follower prototype. The
new script connects only to the follower at `192.168.1.3`.

Reuse or copy the small, proven pieces from `collect_data.py`:

- H.264 writer setup
- episode buffering and HDF5 schema
- `SPACE` / `d` / `q` episode lifecycle

Replace the leader-arm setup, force feedback, joint mirroring, and reconstructed
leader action with the Cartesian keyboard controller below. Replace direct
RealSense acquisition and `process_frame` with the ROS 2 subscriber and fixed
image transform in §6. Replace the OpenCV/terminal input path with one Pygame
window that displays the exact model frame and tracks held keys.

## 3. Required configuration constants

Keep the MVP constants at the top of the script:

```python
FOLLOWER_IP = "192.168.1.3"
TICK_S = 0.2
ACTION_MAGNITUDES_M = {1: 0.0025, 2: 0.005, 3: 0.010}
DEFAULT_SPEED_LEVEL = 2
RESET_STEP_M = 0.005
ACTION_CAP_M = 0.010
UI_FPS = 30

ROS_IMAGE_TOPIC = "/camera/camera/color/image_raw"
ROS_IMAGE_ENCODING = "rgb8"
ROS_IMAGE_WIDTH = 640
ROS_IMAGE_HEIGHT = 480
ROS_IMAGE_FPS = 60
MAX_IMAGE_AGE_S = 0.20
MODEL_IMAGE_SIZE = 224
IMAGE_TRANSFORM_PROFILE_PATH = ...
CAMERA_PARAMETER_DUMP_PATH = ...

START_BASE_X_M = ...
START_BASE_Y_M = ...
FIXED_BASE_Z_M = ...
SAFE_CLEARANCE_BASE_Z_M = ...
X_MIN_M = ...
X_MAX_M = ...
Y_MIN_M = ...
Y_MAX_M = ...
NUM_TRAJECTORY_CHECK_SAMPLES = ...
STARTUP_GOAL_TIME_S = 2.0
HOLD_POSITION_TOLERANCE_M = ...
HOLD_VELOCITY_TOLERANCE_MPS = ...
ORIENTATION_TOLERANCE_RAD = ...
FIXED_ORIENTATION = np.array([0.0, np.pi / 2.0, 0.0])
```

The workspace/pose values must be measured on the robot. Explicitly choose and
benchmark the trajectory-check sample count for the pinned v1.9.3 driver; do
not inherit a version-dependent default. The script must refuse to start while
placeholders remain. Commission the 2.5, 5, and 10 mm levels in that order with
the tool raised and then without the box before allowing contact.

Validate configuration before connecting/moving: all values are finite;
`X_MIN <= START_X <= X_MAX`; `Y_MIN <= START_Y <= Y_MAX`; each min is below its
max; safe-clearance Z is above fixed Z by the commissioned margin;
the action magnitudes are exactly positive and increasing, their maximum equals
`ACTION_CAP_M == 0.010`, `DEFAULT_SPEED_LEVEL` selects the 5 mm entry,
`0 < RESET_STEP_M <= ACTION_CAP_M`; timing and all tolerances are positive; and
the trajectory-check count is a nonnegative integer.

The camera constants are a strict session contract. Both profile paths must
exist before recording. The program must refuse to record if the ROS topic type,
encoding, width, or height differs, or if the measured stream is not healthy
enough to provide a new image for every 5 Hz dataset tick.

A single fixed base Z assumes the tabletop is parallel to base X/Y. Check tip
clearance near every XY workspace corner. If it is not constant, level/shim the
setup before collection; plane-following control is out of scope for this MVP.

## 4. Keyboard behavior

| Key | Command |
|---|---|
| `Up` | `+base_x` (forward) |
| `Down` | `-base_x` (backward) |
| `Left` | `+base_y` (left) |
| `Right` | `-base_y` (right) |
| `1` | select 0.0025 m/step |
| `2` | select 0.005 m/step (default) |
| `3` | select 0.010 m/step |
| `r` | while idle, return to the fixed-pose collection start X/Y |
| `SPACE` | start recording / stop and save |
| `d` | stop and discard the current episode |
| `q` | save an active episode, then quit normally |
| Close window | same graceful path as `q` |

Use one focused Pygame window for both the exact model-view preview and input.
Pump Pygame events at `UI_FPS` and sample `pygame.key.get_pressed()` at each 5 Hz
control tick. Motion uses held state, not `KEYDOWN` repetition; pressing two
orthogonal arrows therefore produces one diagonal action vector.

At each control tick, compute:

```python
direction = np.array([
    int(up_held) - int(down_held),
    int(left_held) - int(right_held),
], dtype=np.float32)

if np.linalg.norm(direction) == 0:
    requested_action = np.zeros(2, dtype=np.float32)
else:
    requested_action = selected_magnitude * direction / np.linalg.norm(direction)
```

Consequences:

- Orthogonal pairs generate the four diagonals. A 5 mm diagonal has components
  about `[3.54, 3.54]` mm, so its norm remains 5 mm rather than `sqrt(2)` times
  faster than a cardinal action.
- Opposing arrows cancel their axis. If the full direction is zero, command and
  record `[0, 0]`.
- Keys `1`, `2`, and `3` persistently select the magnitude for later ticks and
  may be changed while recording. Display the active level and magnitude in the
  Pygame overlay. Record the exact resulting delta, not the speed-level label.
- Arrow control remains available while idle so the operator can set an
  episode's starting EE position.
- Key repeat settings do not affect motion; the sampled held state is the only
  motion input.

Lifecycle and speed selection use edge-triggered `KEYDOWN` events. Process at
most one lifecycle transition per event-pump pass, with priority
`window close/q > d > SPACE > r`; speed selection does not hide a lifecycle
request. Ignore repeated lifecycle `KEYDOWN` events until the matching `KEYUP`.

Focus loss is a safety event. Immediately clear all motion state, request a
zero-delta hold, discard an active episode, and disable motion. After focus
returns, require all arrows to be released before re-enabling idle positioning;
recording never resumes automatically.

- Ignore `r` during recording. Before using it while idle, the operator must
  make sure the planar path back to the start target is clear. Reset is not one
  large command: advance toward start by at most `RESET_STEP_M` per 5 Hz tick
  with the normal bounds and checks. Update the persistent target after each
  accepted step; finish only when start XY is within tolerance. Any arrow
  cancels the remaining reset.
- `SPACE` does not start recording during a reset. Its first press cancels the
  reset and waits for the in-flight step to reach hold tolerance; a later press
  starts the episode. `q` also cancels reset before its shutdown path.

The mapping is defined in the robot base frame. If the camera view makes a key
feel reversed, change only a documented sign mapping after a raised-tool test;
do not change the coordinate semantics stored in the dataset.

## 5. Cartesian command logic

Do not move directly from joint home to a low contact pose. With the table
clear, execute a commissioned startup path:

1. Move to the normal joint-space home pose.
2. Use a slow, blocking, trajectory-checked move to the collection X/Y and
   fixed orientation at `SAFE_CLEARANCE_BASE_Z_M`.
3. Descend in base Z at the same X/Y and orientation to `FIXED_BASE_Z_M`.
4. Verify measured translation, velocity, and orientation are within the
   configured tolerances.

The final startup target is:

```python
[START_BASE_X_M, START_BASE_Y_M, FIXED_BASE_Z_M,
 0.0, np.pi / 2.0, 0.0]
```

Both Cartesian startup segments must use explicit interpolation space,
`STARTUP_GOAL_TIME_S`, blocking mode, and the configured trajectory-check
count. The physical paths must be commissioned with no box; IK feasibility
alone is not collision checking. Enable keyboard input only after startup
succeeds and the measured pose is
within tolerance. Then initialize the persistent target `(target_x, target_y)`
to `(START_BASE_X_M, START_BASE_Y_M)`. This prevents the first arrow from
turning into a large move from the joint-home Cartesian pose.

For every 5 Hz tick:

1. Verify the Pygame window is focused, then read held-arrow state and the
   persistent speed level.
2. Convert it to the normalized `(dx, dy)` from §4; no held direction is zero.
3. Enforce `norm([dx, dy]) <= ACTION_CAP_M`.
4. Form the candidate XY target.
5. Clamp or reject it at the configured X/Y bounds.
6. Compute `accepted_action = accepted_target_xy - previous_target_xy`.
7. Build the full six-value pose with fixed Z and fixed orientation.
8. Send that pose in Cartesian interpolation space with `goal_time=TICK_S`,
   `blocking=False`, and the explicitly configured trajectory-check count.
9. Call the driver and commit the staged observation/action row only if the command is
   accepted without an exception. Store `accepted_action`, not the raw key and
   not later measured motion.

Every command must explicitly contain `FIXED_BASE_Z_M` and
`FIXED_ORIENTATION`. Never copy Z or orientation from a drifting measured pose.
No command may be sent outside the configured XY bounds.

Treat each row as tentative until the non-blocking driver call returns. If the
driver rejects the command, raises, or later reports a Cartesian fault, stop
issuing commands and discard the active training episode; never retain a
nonzero action labeled as sent when acceptance is unknown.

For the pinned v1.9.3 API, `set_cartesian_positions` also requires
`trossen_arm.InterpolationSpace.cartesian`. Do not rely on its default blocking
or trajectory-check behavior. Non-blocking control is required so keyboard and
camera polling continue between commands.

## 6. ROS 2 image input

The robot host runs ROS 2 Jazzy. Before starting the collector, source:

```bash
source /opt/ros/jazzy/setup.bash
```

The image interface is:

| Property | Required value |
|---|---|
| Topic | `/camera/camera/color/image_raw` |
| Type | `sensor_msgs/msg/Image` |
| Encoding | `rgb8` |
| Source size | 640×480 |
| Expected source rate | 60 fps |
| QoS | best effort, volatile, keep last, depth 1 |

Use `rclpy` and `sensor_msgs` from the sourced ROS installation; do not
attempt to install `rclpy` from PyPI (`rclpy` needs `pyyaml`, declared in
`pyproject.toml`). Do not use `cv_bridge`: the Jazzy binary is built against
NumPy 1.x and crashes under this project's NumPy >= 2 pin. Decode the `rgb8`
payload with NumPy only (`ros_camera.decode_rgb8`, a bounded stride-aware
reshape). The callback must do only bounded work: validate the message,
decode/copy it, attach timestamps, and atomically replace a single
latest-frame slot. It must never queue an unbounded image backlog.

Keep these values together for each callback:

```text
rgb_image
image_timestamp_ns      # message.header.stamp
image_receipt_monotonic_ns # time.monotonic_ns() in the callback
frame_sequence          # local monotone callback counter
```

At each 5 Hz dataset tick, atomically snapshot the newest slot. A recording
frame is valid only when its sequence is newer than the previous recorded
sequence and its receipt age is at most `MAX_IMAGE_AGE_S`. If either check
fails, command a zero-delta hold, stop and discard the active episode, and
return to idle. Do not issue the staged motion command and do not reuse a frame.

On startup, wait for the topic and validate its type, `rgb8` encoding, 640×480
shape, strictly increasing source stamps, and approximately 60 Hz delivery.
The collector runs this as a startup stream-health gate (see
`scripts/check_camera_health.py`); `--skip-camera-check` bypasses it when the
operator accepts the risk of frequent freshness-based episode discards.
`/camera/camera/color/camera_info` is optional provenance and is not needed by
this no-detector collector.

The image message does not report whether exposure, white balance, gain, or
focus is locked. Configure supported controls on the external RealSense ROS
publisher, disable automatic modes, and save `ros2 param dump
<publisher_node>` to `CAMERA_PARAMETER_DUMP_PATH` for the session. Fixed-focus
or unsupported controls must be recorded as not applicable rather than guessed.

The 4:3-to-square transform is a commissioning gate, not a decision in this
revision. Before the first saved pilot episode, compare (a) a fixed square ROI
that contains the complete useful workspace and (b) aspect-preserving
letterboxing. Select one, record its ROI/scale/interpolation/padding parameters,
and make the collector refuse to record until that named profile is configured.
Never stretch 640×480 directly to 224×224 and never mix preprocessing profiles
within a dataset. Record the original 640×480 topic in a separate rosbag2
process so this choice can be revisited without recollecting robot interaction.

The collector should spin the ROS executor independently of the 5 Hz control
tick, for example in a background thread. Shutdown must stop the executor,
destroy the node, and call `rclpy.shutdown()` after recording resources close.

## 7. Recording alignment

Preserve the world-model timing contract. “Commit” is an in-memory bookkeeping
operation after the driver accepts the command; observation capture still
occurs before command issue:

```text
at tick t:
    capture pixels[t]
    read proprio[t]
    stage action[t]
    queue that exact action[t]
    if accepted: commit the staged row
    if rejected: discard the active episode

pixels[t + 1] shows the result of action[t]
```

Do not reconstruct actions after the episode from measured EE positions, as
the current prototype does. The action is the exact accepted Cartesian target
delta at command time.

When `SPACE` requests episode stop, use the next tick as a final observation:
record and issue a zero-delta command (the same Cartesian target), then close
the episode. This keeps the outcome of the preceding action inside the episode.
An active `q` uses this same graceful close and then proceeds to shutdown. It is
the only intended episode-lifecycle difference from `collect_data.py`.

The final observation remains on the regular 5 Hz schedule; it represents the
actual 0.2 s outcome even if the arm has not fully converged. After closing the
episode, keep commanding the same target until measured position/velocity are
within hold tolerances or a timeout occurs. If hold is not confirmed, do not
start an automatic retreat.

## 8. Dataset output

Keep the current output layout and types. Buffer an episode as in
`collect_data.py`, then write it with the `stable_worldmodel.HDF5Writer`
contract in `dataset_spec.md` so episode boundaries are preserved:

| Column | Shape per step | Content |
|---|---|---|
| `pixels` | `(224, 224, 3) uint8` | RGB, fixed commissioned square transform from §6 |
| `action` | `(2,) float32` | accepted `[delta_base_x, delta_base_y]` in metres |
| `proprio` | `(4,) float32` | `[base_x, base_y, base_vx, base_vy]` |
| `state` | `(6,) float32` | first two values are EE XY; box values are `NaN` |
| `episode_idx` | scalar `int64` | episode number |
| `step_idx` | scalar `int64` | step inside episode |
| `image_timestamp_ns` | scalar `int64` | ROS image source stamp in nanoseconds |
| `image_receipt_monotonic_ns` | scalar `int64` | host receipt time paired with that image |
| `command_monotonic_ns` | scalar `int64` | host time immediately before issuing the action |

At a dataset tick, use the image and image timestamps from the same atomic ROS
slot; do not assign the current time to an older cached image. Record
`command_monotonic_ns` immediately before calling the arm driver. If a recording
tick has no fresh frame, hold the current target, stop and discard that episode,
and return to idle; never skip a tick and later resume with a timing gap. Camera
availability does not block arrow positioning while idle because those moves
are not logged.

Use a separate rosbag2 process as the native-rate 640×480/60 fps archive. The
collector writes only the selected 5 Hz model frames to its per-episode preview
MP4 and shows that exact 224×224 view live. There is no object detector or
box-pose estimator. The RGB topic is only the world model's observation source
and the operator's preview.

## 9. Manual collection workflow

1. Put the box and pusher fully inside the exact 224×224 model-view preview.
2. Start the script; it homes the arm and moves to the fixed-pose collection
   start target.
3. Use arrows while idle to choose the EE start position; select the initial
   speed with `1`, `2`, or `3`.
4. Press `SPACE` and teleoperate both contact-free motion and pushes, including
   simultaneous-arrow diagonals and all three speed levels.
5. Press `SPACE` to save or `d` to discard.
6. If the box approaches or leaves the useful view, stop the episode and reset
   it manually while recording is off.
7. Use `r` while idle if the arm should return to its known planar start pose.
8. Repeat with varied box position/yaw, approach side, all eight action
   directions, all three magnitudes, push duration, and deliberate no-contact
   motion.

The program does not decide whether contact occurred and does not measure
coverage. The operator supplies that diversity. Prefer keeping the whole box
visible; a brief small crop at an image edge is not automatically invalid, but
long or severe occlusion should be discarded because the model cannot observe
the full state.

The expected primitive support is zero plus eight directions at three
magnitudes (25 action vectors). This supports interpolation within the 10 mm
action disk but does not make the training distribution truly continuous.
Continuous CEM candidates must be norm-projected to that disk and validated
offline before live execution.

## 10. Safety and shutdown

- A supervised operator and reachable physical E-stop are required.
- Run the first coordinate/orientation test with the tool raised and no box.
- Verify each arrow's physical direction one key at a time.
- Verify the pusher-tip height separately before enabling contact.
- Use position mode only. Do not add external-effort/gravity mode to this MVP.
- On normal `q` or a window-close event, gracefully close any active episode,
  stop motion, and close video/HDF5/ROS resources. Keep the Pygame window open
  with a clear-table prompt and wait for an edge-triggered `Enter`. Only after
  that explicit confirmation run a commissioned safe retreat: first raise in
  base +Z with fixed orientation, then use the existing joint-home and sleep
  sequence. Close Pygame afterward. Never sweep directly from a low contact
  pose through an uncleared table.
- On a driver error or unexpected exception, stop issuing commands, close
  recording and Pygame resources, and leave the arm holding its last position;
  do not launch an automatic recovery trajectory.

## 11. Acceptance checks

Hardware-free tests must verify:

- held arrows map to the four expected base-frame cardinal directions;
- every orthogonal arrow pair produces the expected normalized diagonal at
  each of the three magnitudes;
- opposing arrows cancel their axis and no held arrow produces `[0, 0]`;
- speed keys `1`, `2`, and `3` persistently select 2.5, 5, and 10 mm while the
  stored action remains the exact Cartesian delta;
- Pygame key repeat settings do not change held-state motion;
- focus loss commands a zero hold, discards an active episode, disables motion,
  and requires all arrows released after focus returns;
- lifecycle keys are edge-triggered and respect the specified priority;
- repeated commands change only base X/Y;
- base Z and all three orientation values remain invariant;
- `r` is ignored while recording and returns to start X/Y while idle without
  changing fixed Z/orientation;
- workspace clamping changes the recorded action to the delta actually sent;
- row transaction order is `observe -> stage -> command -> commit`;
- rejected commands never commit their staged rows, and discard an active
  episode;
- ROS messages with the wrong type/profile are rejected before arm motion is enabled;
- recording is disabled when the commissioned image-transform profile or camera
  parameter dump is absent;
- best-effort depth-1 reception retains only the newest ROS image;
- while recording, both image timestamps belong to their exact image, command
  time is logged separately, and a reused or older-than-`MAX_IMAGE_AGE_S` image prevents a
  movement command and discards the episode;
- recording is disabled until one named 640×480-to-224×224 transform profile
  is commissioned, and that profile is identical for every row;
- HDF5 columns have the expected shapes and dtypes;
- Pygame and file resources close after a simulated exception.

Live commissioning, in order:

1. Inspect the pinned Python signature of `set_cartesian_positions` on the robot
   host.
2. Confirm the ROS topic is `sensor_msgs/msg/Image`, `rgb8`, 640×480, near
   60 Hz (`uv run scripts/check_camera_health.py` must report HEALTHY on a
   quiet machine), and visually inspect the exact 224×224 model transform.
3. With the tool raised, test fixed orientation, cardinals, normalized
   diagonals, opposing-key cancellation, all three speed levels, and focus-loss
   motion disablement.
4. Benchmark the chosen trajectory-check count while maintaining 5 Hz.
5. Test all XY software bounds with no box.
6. Set the pusher-tip height and verify clearance at all workspace corners.
7. Commission the clear-table vertical retreat and shutdown path.
8. Record a short no-contact episode and check timing/action alignment.
9. Record one supervised contact episode and inspect the HDF5 and MP4.

Implementation is complete only after the hardware-free tests pass. Contact
collection begins only after all nine commissioning checks pass.
