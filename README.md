# wm_data_collection

Data collection for the LeWM **push-box pilot**. The current prototype uses two
Trossen WXAI arms in leader/follower teleoperation. The keyboard collector uses only
the follower, with arrow-key Cartesian X/Y motion, fixed height/orientation, and
one Intel RealSense color stream delivered by ROS 2 Jazzy. Both approaches
record 5 Hz HDF5 episodes; only the legacy prototype opens the camera directly.

The full dataset contract (rig, timing, action space, schema, QA checklist,
training criteria) lives in [dataset_spec.md](dataset_spec.md).

## Hardware

| Component | Details |
|---|---|
| Leader arm | Trossen WXAI V0, `192.168.1.5` (prototype only) |
| Follower arm | Trossen WXAI V0, `192.168.1.3` (keyboard collector) |
| Camera (keyboard collector) | ROS 2 topic `/camera/camera/color/image_raw`; `sensor_msgs/msg/Image`, `rgb8`, 640×480 @ 60 fps |
| Camera (legacy prototype) | Direct `pyrealsense2` color stream, 640×480 @ 30 fps |

IPs, gains, and dataset constants are defined at the top of each script.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
source /opt/ros/jazzy/setup.bash  # required by the keyboard collector
uv sync
```

`rclpy`, `sensor_msgs`, and `cv_bridge` come from ROS 2 Jazzy, not PyPI. The
existing prototype scripts still use `pyrealsense2` directly.

## Repository layout

```
├── README.md
├── dataset_spec.md            # dataset contract: rig, timing, schema, QA
├── pyproject.toml / uv.lock   # uv-managed environment
└── scripts/
    ├── collect_data.py        # current leader/follower prototype
    ├── collect_keyboard_xy.py # ROS 2 + Pygame follower-only collector
    ├── collector_core.py      # dependency-light action/image/schema helpers
    ├── ros_camera.py          # latest-frame ROS 2 subscriber
    ├── pygame_input.py        # focused held-key input and safety state
    ├── teleoperation.py       # bare teleop (no recording)
    └── record_realsense.py    # standalone camera recorder
```

| Script | Purpose |
|---|---|
| [collect_data.py](scripts/collect_data.py) | Current leader/follower prototype — teleop + camera + HDF5 episodes + video |
| [collect_keyboard_xy.py](scripts/collect_keyboard_xy.py) | Follower-only ROS 2 image + Pygame arrow-key XY collector |
| [teleoperation.py](scripts/teleoperation.py) | Bare leader/follower teleop with force feedback (no recording); press `q` to stop |
| [record_realsense.py](scripts/record_realsense.py) | Standalone color+depth mp4 recorder, for checking the camera |

The implementation spec for the follower-only collector is
[docs/keyboard_xy_collection_spec.md](docs/keyboard_xy_collection_spec.md).
Its focused local Pygame window tracks held arrows, including normalized
diagonals, with `1`/`2`/`3` selecting 2.5/5/10 mm action magnitudes.
The short runbook for the person operating it is
[docs/teleoperator_guidelines.md](docs/teleoperator_guidelines.md).

To validate and summarize every HDF5 dataset below `data/` and fully decode
its matching preview videos below `output_videos/`, run:

```bash
bash scripts/check_dataset.sh
```

The report prints per-file, per-episode, and combined timing, image latency,
action coverage, arm tracking, pixel continuity, brightness, and video/HDF5
frame-count statistics. Its final summary shows the collected time, percentage
of the four-hour goal, and remaining time. Pass `--goal-hours N` to change that
goal, explicit HDF5 paths for a targeted report, or `--strict` when warnings
should produce a failing exit status.

To merge every dataset below `data/` into one training file per collection
date, first inspect the plan and then run the merge:

```bash
bash scripts/merge_datasets.sh --dry-run
bash scripts/merge_datasets.sh
```

For example, all files containing `20260713` in their name are merged into
`merged_data/pushbox_keyboard_20260713.h5`. The merger validates compatible
schemas, reindexes episodes, rebuilds episode bookkeeping, records source-file
provenance, and never modifies the source files. Existing merged files are not
replaced unless `--overwrite` is passed.

The collector requires a commissioned image-transform JSON and camera-parameter
dump, plus robot-specific pose/bound values:

```bash
uv run scripts/collect_keyboard_xy.py \
  --transform-profile profiles/pushbox_letterbox.json \
  --camera-params camera_params.yaml \
  --fixed-z 0.20 --safe-z 0.30 \
  --start-x 0.30 --start-y 0.0 \
  --x-min 0.15 --x-max 0.45 --y-min -0.25 --y-max 0.25 \
  --trajectory-check-samples 10
```

## Collecting data with the current prototype

Run in a real terminal (keyboard handling needs a TTY):

```bash
uv run scripts/collect_data.py            # writes pushbox_prototype.h5
uv run scripts/collect_data.py my_run.h5  # or choose a path
```

| Key | Action |
|---|---|
| `SPACE` | Start recording an episode / stop and **save** it |
| `d` | **Discard** the episode being recorded (also deletes its video) |
| `q` | Quit (saves an in-progress episode); arms go home → sleep |

If a display is attached, a live preview shows the exact 224×224 view the
model will see, with a REC/idle overlay. The HDF5 file opens in append mode,
so collection can resume across sessions and episode numbering continues.

### Outputs

```
my_run.h5                 # all episodes, appended
my_run_videos/ep_000.mp4  # raw 640x480@30 color stream, one file per episode
                          # (H.264 + yuv420p — plays directly in VS Code)
```

HDF5 columns emitted by the current prototype (see its limitations below):

| Column | Shape / dtype | Content |
|---|---|---|
| `pixels` | `(224, 224, 3) uint8` | RGB, center-cropped from 640×480 |
| `action` | `(2,) float32` | Post-hoc leader EE `[Δx, Δy]`, component-clipped to ±0.025 m |
| `proprio` | `(4,) float32` | `[ee_x, ee_y, ee_vx, ee_vy]` from the follower arm |
| `state` | `(6,) float32` | `[ee_x, ee_y, box_x, box_y, cos_yaw, sin_yaw]` — box fields NaN for now |
| `episode_idx` / `step_idx` | `int64` | Episode counter / step within episode |
| `timestamp` | `float64` | Unix time at capture (QA) |

Quick inspection of a collected file:

```python
import h5py
with h5py.File("pushbox_prototype.h5") as f:
    for k, v in f.items():
        print(k, v.shape, v.dtype)
```

## Prototype limitations

This is a setup-validation prototype; before the real collection run
(per the spec) the following still need to be addressed:

- **Camera auto-exposure / white balance are ON** — must be locked for the
  dataset (world models are brittle to photometric drift).
- It requires the leader arm and mirrors joint targets instead of accepting
  bounded Cartesian keyboard commands.
- It reconstructs `action` from later leader poses instead of recording the
  exact Cartesian delta sent at each tick.
- Box pose fields of `state` are `NaN` by design for the no-vision MVP.
