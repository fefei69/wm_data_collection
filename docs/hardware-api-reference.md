# Hardware & API Reference

Consolidated facts for the executing agent. **Verified** items come from the
existing prototype scripts in `scripts/` and are safe to rely on.
**Commissioning-gated** items must be confirmed on the Linux robot host before
any live call — do not guess them, and do not enable the live path on the macOS
dev host.

## Rig

- **Follower arm** — Trossen WXAI V0, follower end effector, IP `192.168.1.3`.
  This is the only arm used for V1 collection.
- **Leader arm** — WXAI V0, IP `192.168.1.5` in `collect_data.py` (some reference
  scripts say `.2`). **Unused in V1** — park it, do not connect it.
- **Camera** — one fixed Intel RealSense, color stream 640×480 @ 30 fps (`bgr8`),
  mounted overhead / high-oblique so the whole arena, the box, and the tool are
  always visible.
- IPs, gains, and calibrated transforms belong in the gitignored
  `config/pushbox_collection.local.toml` — never in code or the example config.

## Arm degrees of freedom

The `_all_` API operates on **7 entries = 6 arm joints + 1 gripper**.
`HOME_POSITIONS = [0.0, π/2, π/2, 0.0, 0.0, 0.0, 0.0]` (the last element is the
gripper). `get_num_joints()` returns this count.

## Trossen driver API — verified from the prototype scripts

```python
import trossen_arm
driver = trossen_arm.TrossenArmDriver()
driver.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_follower,  # follower for collection
    "192.168.1.3",
    False,
)

# Modes (GLOBAL — one mode for every joint; no verified per-joint mode setter):
driver.set_all_modes(trossen_arm.Mode.position)         # or Mode.external_effort

# Commands — trailing scalars are (goal_time, blocking), NOT a gripper value:
driver.set_all_positions(positions7, goal_time, blocking)            # + optional velocities7
driver.set_all_external_efforts(efforts7, goal_time, blocking)

# Reads:
driver.get_all_positions(); driver.get_all_velocities(); driver.get_all_external_efforts()
driver.get_cartesian_positions()   # tool pose; [0],[1] read as x,y in the prototype
driver.get_cartesian_velocities()
driver.get_num_joints()
```

Notes:
- `set_all_modes` is **global**. Do not assume you can hold the gripper in
  position mode while the arm is in another mode — no per-joint mode setter is
  demonstrated by any script.
- In the prototype the table frame **equals** the arm base frame. V1 adds a real
  table↔base calibration (from `calibrate_table.py`, stored in the local config),
  and all recorded targets are expressed in the table frame.

## Gravity compensation — reference only, DROPPED from V1

The Trossen manual-teaching pattern is:

```python
driver.set_all_modes(trossen_arm.Mode.external_effort)
driver.set_all_external_efforts([0, 0, 0, 0, 0, 0, 0], 0.0, False)
input("Press Enter to end gravity compensation...")
```

This back-drives **all 7 DOF including the gripper** (the `0.0` is `goal_time`,
not a gripper hold). It cannot keep the gripper holding a tool and cannot enforce
XY-only guiding, so **V1 has no gravity / hand-guided mode**. This is documented
here only to explain the decision; do not add such a mode.

## Cartesian POSITION commands — needed by V1, commissioning-gated

No existing script commands a Cartesian *position*. `FollowerRobot.command_xy`,
`move_to_start`, and `park` all need `driver.set_cartesian_positions(...)`.

- Per the Trossen v1.9 API, a Cartesian target is a **6-vector
  `[x, y, z, ax, ay, az]`** — translation + angle-axis, in the base frame.
- The **exact Python signature is UNVERIFIED**: argument order, any interpolation
  flag, `goal_time`, `blocking`, and the trajectory-check setting. Confirm it on
  the robot host (plan Task 4 Step 4 and Task 10) before enabling the live call.
  The fake driver in tests defines this contract; make the real adapter match
  whatever the pinned binding actually exposes.
- Cartesian motion can **fault near singularities** — reject/handle IK and
  command-tracking failures.

## RealSense — verified from the prototype

```python
import pyrealsense2 as rs
pipeline = rs.pipeline(); config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
```

Frame processing (reuse `process_frame` in `collect_data.py` exactly):
640×480 BGR → center **square crop** → resize to **224×224** → BGR→RGB `uint8`.

**Spec §1 requires locking exposure, white balance, focus, and gain.** The
prototype leaves auto-* on (a known gap). V1's `CameraSource.lock_settings()`
must turn every auto-* **off** and set fixed values, and `PRECHECK` must refuse
to record while any auto-setting is still on.

## Dataset constants (spec + prototype)

- Tick **0.2 s (5 Hz)**. Action cap **|Δ| ≤ 0.025 m/step**, `float32`, table
  frame. Image **224×224×3 `uint8`**.
- Episode length target **120–150 steps**; **hard floor 50**.
- HDF5 columns: `pixels`, `action` (T,2), `proprio` (T,4), `state` (T,6),
  `episode_idx`, `step_idx`, `timestamp`. Use the stable-worldmodel `HDF5Writer`
  (dataset_spec §7) — **do not** reinvent the `ep_len`/`ep_offset` layout; prove
  it with a loader smoke test before accepting a dataset.

## Version pins

- Python ≥ 3.12. `trossen-arm >= 1.9, < 1.10`.
- **Pin the exact `trossen_arm` package + firmware pair on the robot host** before
  any live Cartesian motion — the public API is actively developed; do not rely on
  a `main`-branch signature.
