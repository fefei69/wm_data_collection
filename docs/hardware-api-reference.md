# Hardware API Reference — Minimal Keyboard Collector

This file records only the hardware facts needed to implement and commission
`scripts/collect_keyboard_xy.py`.

## Rig

- Follower: Trossen WXAI V0 with follower end effector, `192.168.1.3`.
- Leader: unused by the keyboard collector.
- Camera source: ROS 2 Jazzy topic `/camera/camera/color/image_raw`,
  `sensor_msgs/msg/Image`, `rgb8`, 640×480 at 60 fps (raised from 30 on
  2026-07-11 so per-tick freshness survives multi-frame delivery gaps).
- Dataset/control rate: 5 Hz (`0.2 s` per action).

## Existing verified calls

The current prototype demonstrates:

```python
driver = trossen_arm.TrossenArmDriver()
driver.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_follower,
    "192.168.1.3",
    False,
)

driver.set_all_modes(trossen_arm.Mode.position)
driver.set_all_positions(positions7, goal_time, blocking)
driver.get_cartesian_positions()
driver.get_cartesian_velocities()
driver.get_num_joints()
```

The normal joint-space home pose in `collect_data.py` is:

```python
[0.0, pi / 2, pi / 2, 0.0, 0.0, 0.0, 0.0]
```

## Cartesian pose convention

For Trossen Arm v1.9, Cartesian targets are six values:

```text
[translation_x, translation_y, translation_z,
 angle_axis_x, angle_axis_y, angle_axis_z]
```

The pose is measured in the **robot base frame**. The official Cartesian demo
identifies the WXAI base directions as:

```text
+X forward     -X backward
+Y left        -Y right
+Z up          -Z down
```

The table plane is therefore base X/Y and fixed height is base Z. Tool
orientation never changes the meaning of the first three values.

The fixed orientation requested for collection is:

```python
[0.0, pi / 2, 0.0]
```

It maps tool `+X` down, tool `+Y` left, and tool `+Z` forward.

Primary references:

- [Trossen Arm v1.9.3 driver header](https://github.com/TrossenRobotics/trossen_arm/blob/v1.9.3/include/libtrossen_arm/trossen_arm.hpp#L223-L264)
- [Trossen Arm v1.9.3 Cartesian output type](https://github.com/TrossenRobotics/trossen_arm/blob/v1.9.3/include/libtrossen_arm/trossen_arm_type.hpp#L265-L295)
- [Trossen Arm v1.9.3 Python Cartesian demo](https://github.com/TrossenRobotics/trossen_arm/blob/v1.9.3/demos/python/cartesian_position.py#L64-L93)

## Tool offset and height

If the reported 5 cm is the flange-to-pusher-tip offset, it belongs in the
flange-to-tool transform, not in base X. Prefer configuring the tool frame at the pusher tip with Trossen's
`t_flange_tool`. Then the Cartesian Z target is simply the measured table Z plus
the requested tip height above it.

If `t_flange_tool` is identity (so the configured tool frame coincides with the
flange) and the tip is 5 cm along tool `+X`,
the fixed flange Z is 5 cm above the desired tip Z because tool `+X` points
down. Verify the actual mount and configured tool frame before using that
formula.

Reference: [Trossen end-effector configuration](https://docs.trossenrobotics.com/trossen_arm/v1.9/getting_started/configuration.html).

## Cartesian call contract and robot-host checks

`uv.lock` currently pins `trossen-arm==1.9.3`; use that lock on the robot host or
pin the project dependency exactly before relying on this contract. In v1.9.3,
the Cartesian position call requires an interpolation-space argument and has
the following logical parameters:

```text
set_cartesian_positions(
    target6,
    InterpolationSpace.cartesian,
    goal_time,
    blocking,
    optional_feedforward_velocity,
    optional_feedforward_acceleration,
    num_trajectory_check_samples,
)
```

The 5 Hz loop must explicitly use `goal_time=0.2` and `blocking=False`; otherwise
the default blocking call can stall camera and keyboard polling. The startup
move is instead slow, blocking, and checked. Inspect the installed binding and
confirm:

- the exact Python argument names/order and representation of optional values;
- the chosen explicit trajectory-check sample count fits the 0.2 s tick budget;
- what happens after an IK failure or communication timeout;
- the current `t_flange_tool` transform and therefore where the configured tool
  frame lies; identity means it coincides with the flange.

Do not guess these from a newer branch of the library. Test first with no box
and the tool raised.

## ROS 2 Jazzy image contract

The keyboard collector does not import `pyrealsense2` and does not configure
the camera. A separate RealSense ROS node publishes:

| Property | Required value |
|---|---|
| Topic | `/camera/camera/color/image_raw` |
| Type | `sensor_msgs/msg/Image` |
| Encoding | `rgb8` |
| Width / height | 640 / 480 |
| Expected rate | 60 fps |

Source ROS before launching the collector:

```bash
source /opt/ros/jazzy/setup.bash
```

Use `rclpy` and `sensor_msgs` from the Jazzy installation. The
subscription requests best-effort reliability, volatile durability, keep-last
history, and depth 1. This is compatible with the normal sensor-data publisher
profile and prevents queued old frames. The application also retains only one
atomic latest-frame slot.

Do **not** use `cv_bridge`: Jazzy's `cv_bridge` extension is compiled against
NumPy 1.x and crashes under this project's NumPy >= 2 pin (verified on the
robot host 2026-07-11). An `rgb8` payload needs no color conversion, so
`scripts/ros_camera.py::decode_rgb8` decodes it with NumPy only — a bounded
reshape honoring `message.step` — and `rclpy` additionally needs `pyyaml`,
which is declared in `pyproject.toml`.

Pair every image with its ROS `header.stamp`, host `time.monotonic_ns()` receipt
time, and a local sequence counter. Record a separate monotonic timestamp
immediately before each arm command. A 5 Hz recording tick may use an image only
once and only while its receipt age is at most 0.20 s (`IMAGE_MAX_AGE_S`).
Wrong encoding/shape, a reused frame, or a stale frame invalidates an active
episode.

### Stream health gate (measured 2026-07-11)

Delivery hiccups of 100–370 ms occur even on a healthy rig, and desktop load
(a browser, etc.) multiplies them ~80×. For a 150-tick episode to survive the
freshness rule with p >= 0.9, the newest frame may be older than the bound for
at most ~0.05% of wall time. `scripts/check_camera_health.py` measures exactly
this and must report HEALTHY before a session; the collector runs the same gate
at startup (skip with `--skip-camera-check`). Collect on a quiet machine.
Kernel tuning that helps and currently resets at reboot: `net.core.rmem_max`
/ `rmem_default` / `wmem_max` / `wmem_default` = 64 MB, and
`/sys/module/usbcore/parameters/usbfs_memory_mb` = 256.

The required source profile is already appropriate. Changing the SDK profile
is unnecessary unless a tested native square profile preserves at least the
same useful field of view at 60 fps. The 4:3-to-square model transform remains
a commissioning choice between a fixed square workspace ROI and
aspect-preserving letterboxing. Record and freeze the selected parameters
before collecting; never stretch 4:3 directly to 224×224 or mix transforms.

The image topic cannot prove that exposure, white balance, gain, or focus is
locked. Configure every supported control on the external RealSense publisher,
disable its automatic modes, and save a ROS parameter dump with each collection
session. Record fixed-focus or unsupported controls as not applicable. Discover
the actual publisher node with `ros2 node list`, then archive its parameters
with `ros2 param dump <publisher_node>`; parameter names vary by driver version,
so do not guess them in the collector.

The new script needs no depth topic, detector, tags, camera calibration, or
object pose. `/camera/camera/color/camera_info` may be archived as provenance
but is not a runtime dependency.

## Gravity compensation

The keyboard MVP uses position mode only. Trossen's external-effort mode
back-drives the whole arm and does not enforce an XY-only hand-guided constraint,
so it is out of scope.
