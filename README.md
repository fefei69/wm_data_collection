# wm_data_collection

Data collection for the LeWM **push-box pilot**: two Trossen WXAI arms in
leader/follower teleoperation plus one fixed Intel RealSense color camera,
recording 5 Hz episodes to HDF5 for world-model training. No ROS — plain
Python talking to the arm drivers and the camera directly.

The full dataset contract (rig, timing, action space, schema, QA checklist,
training criteria) lives in [dataset_spec.md](dataset_spec.md).

## Hardware

| Component | Details |
|---|---|
| Leader arm | Trossen WXAI V0, `192.168.1.5` |
| Follower arm | Trossen WXAI V0, `192.168.1.3` |
| Camera | Intel RealSense (color stream, 640×480 @ 30 fps) |

IPs, gains, and dataset constants are defined at the top of each script.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Repository layout

```
├── README.md
├── dataset_spec.md            # dataset contract: rig, timing, schema, QA
├── pyproject.toml / uv.lock   # uv-managed environment
└── scripts/
    ├── collect_data.py        # main collector
    ├── teleoperation.py       # bare teleop (no recording)
    └── record_realsense.py    # standalone camera recorder
```

| Script | Purpose |
|---|---|
| [collect_data.py](scripts/collect_data.py) | **Main collector** — teleop + camera + HDF5 episodes + per-episode video |
| [teleoperation.py](scripts/teleoperation.py) | Bare leader/follower teleop with force feedback (no recording); press `q` to stop |
| [record_realsense.py](scripts/record_realsense.py) | Standalone color+depth mp4 recorder, for checking the camera |

## Collecting data

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

HDF5 columns per step (see spec §4 for the full contract):

| Column | Shape / dtype | Content |
|---|---|---|
| `pixels` | `(224, 224, 3) uint8` | RGB, center-cropped from 640×480 |
| `action` | `(2,) float32` | Commanded EE `[Δx, Δy]` in meters, clipped to ±0.025 |
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
- **No AprilTag tracking yet** — the box pose fields of `state` are NaN
  (handled safely by the training pipeline), and the "table frame" is
  currently the arm base frame.
