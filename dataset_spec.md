# Dataset Spec — Trossen AI Arm, Push-Box Pilot (1 hour)

Target: a 1-hour pilot dataset to validate the full LeWM pipeline (collection → HDF5 →
`train.py` → probing / planning smoke test) before committing to the full 8–15 h collection.
Everything below is pinned to what this repo's code actually consumes
(`HDF5Dataset`, `train.py`, `eval.py` with `stable-worldmodel==0.1.1`).

Reference points: LeWM paper App. D/E (224×224, episodes 92–200 steps);
WorldPlanner & FoG-MBRL papers (real-arm pushing world models trained from scratch on
4 h of 5 Hz play with Cartesian velocity actions).

---

## 1. Rig

| Item | Requirement |
|---|---|
| Camera | One fixed Intel RealSense, rigidly mounted overhead or high-oblique. The collector receives ROS 2 Jazzy `sensor_msgs/msg/Image` messages from `/camera/camera/color/image_raw`; it never opens the camera with `pyrealsense2`. The camera supplies pixels only; no online detector is required. |
| ROS image profile | `rgb8`, 640×480, 30 fps. Subscribe best-effort/volatile with keep-last depth 1 and retain only the newest message. Preserve the full source frame in the raw archive. |
| Camera settings | **Lock exposure, white balance, focus, gain.** No auto-anything — world models are brittle to photometric drift (stable-worldmodel paper Tab. 2). |
| Lighting | Fixed, diffuse. No windows/daylight variation if possible. |
| Work area | Flat, plain matte table with high contrast to the box. A physical border is optional; when the box approaches or leaves the camera/reachable area, stop recording and reset it manually. |
| Box | Rigid, ~8–15 cm, light enough to push, and visually distinct. No marker or online box tracking is required. |
| EE tool | Rigid cylindrical pusher (PushT-style stick). Define the commanded frame explicitly (prefer the pusher contact point), hold its base-frame Z at a measured contact height above the table, and use angle-axis `[0, π/2, 0]`, so tool +X points down, +Y left, and +Z forward. |

## 2. Timing / frequency

- **Decision rate: 5 Hz** (0.2 s per step). This is the dataset's step rate; the arm's
  internal Cartesian controller runs at its native rate underneath.
- **Training frame skip: 1.** Every stored 5 Hz observation/action step is used; one model
  transition is 0.2 s. Temporal subsampling can be evaluated later without recollecting.
- Rationale: the raw rate matches WorldPlanner / FoG real-robot pushing while preserving
  contact onset and release for the first LeWM baseline.
- **Alignment contract:** at tick *t*, capture `pixels[t]`, `proprio[t]`, `state[t]`,
  **then** issue `action[t]`. So `pixels[t+1]` shows the outcome of `action[t]`.
- The image callback stores the ROS `header.stamp` and host monotonic receipt time with the
  image. At a dataset tick, consume the newest message only if it is new since the previous
  tick and no more than 0.10 s old. Otherwise hold position and discard the active episode.
- Ticks must be uniform to ±10% (see QA §8).

## 3. Action space — 2D Cartesian delta

- `action[t] = [Δx, Δy]` **float32, meters**, in the robot base X/Y frame: the EE displacement
  commanded for this 0.2 s tick (equivalently velocity × 0.2 s).
- **Collected magnitudes:** 0.0025, 0.005, and 0.010 m/step, selected by keys
  `1`, `2`, and `3`. **Cap: `norm([Δx, Δy]) ≤ 0.010 m/step`** (= 5 cm/s).
  Normalize simultaneous-arrow diagonals so their total norm equals the selected
  magnitude, then project to the workspace before sending. **Record the bounded
  command actually sent** (not the raw teleop input, not the achieved motion).
- Base Z, orientation, and gripper: constant, handled by the controller, **not** in the action vector.
- Trossen translations remain base-frame coordinates after rotating the tool. For the standard
  WXAI convention, base +X is forward, +Y is left, and +Z is up; planar movement is X/Y and
  height is Z.
- No joint space (CEM samples Gaussians — unsafe + wastes model capacity on kinematics),
  no quaternions (not normalizable/sampleable; latent WMs capture rotation poorly).
- Units don't need pre-normalization — the pipeline z-scores per column
  (`utils.get_column_normalizer` for training, `StandardScaler` in `eval.py`).
  Just be **consistent**.
- `train.py` auto-sets `action_encoder.input_dim = frameskip × action_dim` (= 1×2 = 2),
  so a 2D action needs **zero model-config changes**.

## 4. Observation columns

| Column | Shape/dtype | Content |
|---|---|---|
| `pixels` | `(224, 224, 3) uint8` | RGB derived from the 640×480 ROS image. The 4:3-to-square transform is deliberately not selected in this revision: during commissioning, compare a fixed square ROI that contains the entire useful workspace against aspect-preserving letterboxing, choose one, record its exact parameters, and freeze it before the first saved pilot episode. Never stretch 4:3 directly to 1:1 or mix transforms within a dataset. |
| `proprio` | `(4,) float32` | `[ee_x, ee_y, ee_vx, ee_vy]` — EE base-frame X/Y position (m) and velocity (m/s). |
| `state` | `(6,) float32` | `[ee_x, ee_y, NaN, NaN, NaN, NaN]`. Keep the compatible shape; box pose is intentionally not measured online and is not used by the training loss. |
| `episode_idx` | `() int64` | Episode counter, monotone across the file. |
| `step_idx` | `() int64` | 0-based step within the episode. |
| `image_timestamp_ns` | `() int64` | ROS `Image.header.stamp` converted to nanoseconds. |
| `image_receipt_monotonic_ns` | `() int64` | Host monotonic time when that exact ROS message was received; used for freshness QA. |
| `command_monotonic_ns` | `() int64` | Host monotonic time immediately before issuing `action[t]`; used for the 5 Hz cadence and image-to-command latency QA. |

The box fields of `state` are always `NaN` in this MVP. The LeWM loss consumes
`pixels` + `action`; do not add a detector solely to populate auxiliary state.

## 5. Episode structure

- **Length: 120–150 steps (24–30 s).** Hard floor 50 steps (PushT's min is 49; eval needs
  ≥ `goal_offset` 25 + margin), soft ceiling ~250.
- **1 hour of robot time = 18,000 steps ≈ 125–150 episodes.** With ~10–15 s manual
  re-randomization between episodes, budget ~1.5 h wall clock.
- Between episodes: manually vary **box position over the useful work area, box yaw,
  and the EE start pose**. No resets *within* an episode.
- Training windows span `num_steps × frameskip = 4 × 1 = 4` steps, so every episode ≥ 50
  steps yields plenty of samples; episode boundaries are handled by the loader
  (NaN-padded actions are zeroed in `train.py:25`).

## 6. Collection policy (what the play should look like)

Arrow-key teleop doing **unstructured, high-entropy play** — no task
success required, no demonstration quality bar (LeWM §3.1 allows "pseudo-expert or
exploratory" data; both real-robot reference papers used pure play):

- Repeat: approach the box from a random direction → push through contact 5–20 cm →
  release / reposition → repeat. Vary direction and push duration.
- Cover **all four faces and the corners** of the box (corner pushes produce rotation —
  you need rotational dynamics in the data).
- Cover all eight commanded directions and all three speed levels throughout
  both contact and contact-free motion; do not let normal-speed cardinal moves
  dominate the dataset.
- **~10–20% contact-free EE motion** (the model must learn that no contact ⇒ box stays).
- Coverage goals for the hour: box visits all work-area quadrants; box yaw covers the full
  circle; EE approaches from all sides; typical 5 s contact window moves the box ≥ 3–5 cm
  or rotates it ≥ 10°.
- Keep hands/faces out of frame; if you must intervene mid-episode, end the episode.

The local Pygame input tracks held-key state, so simultaneous arrows produce the
eight compass directions. Diagonals are normalized rather than moving `sqrt(2)`
faster. Together with zero and the three magnitudes, this gives 25 deliberately
covered primitive actions. This is still a finite training set: continuous CEM
relies on interpolation between those actions, must project every candidate to
`norm(action) <= 0.010`, and requires offline held-out validation before any
robot execution.

## 7. File format & writer

One file: `$STABLEWM_HOME/datasets/pushbox_pilot_train.h5`
(`STABLEWM_HOME=/home/cpw/workspace/le-wm/stable-wm`).

Use the library's writer — it manages `ep_len`/`ep_offset` and locks the schema from the
first episode:

```python
import numpy as np
import stable_worldmodel as swm

with swm.data.HDF5Writer(
    "/home/cpw/workspace/le-wm/stable-wm/datasets/pushbox_pilot_train.h5",
    mode="append",          # safe to resume across collection sessions
) as w:
    # once per finished episode, with T = episode length:
    w.write_episode({
        "pixels":      ep_pixels,                      # (T,224,224,3) uint8
        "action":      ep_actions.astype(np.float32),  # (T,2)
        "proprio":     ep_proprio.astype(np.float32),  # (T,4)
        "state":       ep_state.astype(np.float32),    # (T,6)
        "episode_idx": np.full(T, ep_id, np.int64),    # (T,)
        "step_idx":    np.arange(T, dtype=np.int64),   # (T,)
        "image_timestamp_ns": ep_image_ts.astype(np.int64),  # (T,)
        "image_receipt_monotonic_ns": ep_receipt_ts.astype(np.int64), # (T,)
        "command_monotonic_ns": ep_command_ts.astype(np.int64), # (T,)
    })
```

Buffer each episode in RAM (150 steps × 150 KB ≈ 23 MB) and write on episode end.
Expected file size: ~2.7 GB uncompressed (fine for the pilot; the writer doesn't compress).
**Also record the raw ROS image topic with rosbag2 at 640×480/30 fps.** The
collector may write a separate 5 Hz per-episode model-view MP4 for quick review,
but that preview is not the native-rate archive. The bag lets you regenerate the
HDF5 after the image-transform commissioning comparison without re-collecting.

## 8. QA checklist (run before training)

1. **Alignment:** `proprio[t+1, :2] − proprio[t, :2] ≈ action[t]` (correlation > 0.9 on
   contact-free segments). If it correlates with `action[t±1]` instead, your logging is
   off by one tick.
2. **Timing:** `diff(command_monotonic_ns)` is 0.20 ± 0.02 s; source stamps are
   strictly increasing; and `command_monotonic_ns - image_receipt_monotonic_ns`
   is between 0 and 0.10 s for every selected image.
3. **State contract:** all four unmeasured box-state values are `NaN`; EE X/Y remain finite.
4. **Visual sanity:** random 20 frames — box+EE are usefully visible, exposure constant, no motion
   blur that hides the box edge.
5. **Manual coverage review:** sample episode videos and confirm varied box positions/yaws,
   approach sides, push directions, contact durations, and deliberate no-contact motion.
6. **Loader smoke test:**
   `HDF5Dataset('pushbox_pilot_train', frameskip=1, num_steps=4, keys_to_load=['pixels', 'action'])`
   loads, and one `DataLoader` batch has `pixels (B,4,3,224,224)`-ish shapes post-transform.

## 9. Training & pilot success criteria

Create `config/train/data/pushbox.yaml` (copy of `pusht.yaml` with
`name: pushbox_pilot_train.h5`, `frameskip: 1`, and
`keys_to_load: [pixels, action]`), then:

```bash
python train.py data=pushbox trainer.max_epochs=100
```

Note on epochs: the paper's 10 epochs on 2.3M frames ≈ 160k optimizer steps; your pilot has
~16k windows, so 10 epochs ≈ 1.1k steps — far too few. ~100 epochs ≈ 11k steps is a
reasonable pilot budget (batch 128 may also need lowering if windows < 128×steps needed).

The pilot **passes** if:
1. `pred_loss` and `sigreg_loss` both decrease smoothly and plateau (cf. paper Fig. 18);
   sigreg drops sharply early.
2. Short held-out video rollouts are visually coherent enough to preserve the box and
   predict the qualitative result of contact versus no contact. A quantitative box-pose
   probe requires labels collected separately and is not part of this MVP.
3. Optional offline CEM smoke test: project candidates into the 0.010 m action
   disk and check held-out cardinal, diagonal, and intermediate actions. Do not
   execute interpolated actions on the arm until prediction checks and separate
   no-contact commissioning pass.

Real-robot closed-loop eval can't reuse `eval.py`'s dataset-driven protocol (`_set_state`
teleports don't exist in reality) — plan a small driver script that resets the box manually,
takes a goal image, and runs the policy with a step budget. That's a follow-up deliverable.
