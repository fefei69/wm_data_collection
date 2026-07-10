# Dataset Spec — Trossen AI Arm, Push-Box Pilot (1 hour)

Target: a 1-hour pilot dataset to validate the full LeWM pipeline (collection → HDF5 →
`train.py` → probing / planning smoke test) before committing to the full 8–15 h collection.
Everything below is pinned to what this repo's code actually consumes
(`HDF5Dataset`, `train.py`, `eval.py` with `stable-worldmodel==0.1.1`).

Reference points: LeWM paper App. D/E (frameskip 5, 224×224, episodes 92–200 steps);
WorldPlanner & FoG-MBRL papers (real-arm pushing world models trained from scratch on
4 h of 5 Hz play with Cartesian velocity actions).

---

## 1. Rig

| Item | Requirement |
|---|---|
| Camera | One fixed RGB camera, rigidly mounted, overhead or high-oblique so the **entire arena, the box, and the EE tool are always visible** (avoid arm-body occlusions). |
| Camera settings | **Lock exposure, white balance, focus, gain.** No auto-anything — world models are brittle to photometric drift (stable-worldmodel paper Tab. 2). |
| Lighting | Fixed, diffuse. No windows/daylight variation if possible. |
| Arena | Bounded planar workspace ~40×40 cm (walls or firm border) so the box cannot leave reach; plain matte surface with high contrast to the box. |
| Box | Rigid, ~8–15 cm, light enough to push, visually distinct color. **AprilTag (36h11) on top face** + one static tag on the table for the world frame. |
| EE tool | Rigid cylindrical pusher (PushT-style stick) mounted in place of / held by the gripper. **Fixed height** (z ≈ box mid-height), **fixed orientation** (pointing down), gripper state constant. |

## 2. Timing / frequency

- **Decision rate: 5 Hz** (0.2 s per step). This is the dataset's step rate; the arm's
  internal Cartesian controller runs at its native rate underneath.
- Rationale: matches WorldPlanner / FoG real-robot pushing; with the repo's `frameskip: 5`,
  one model action-block = 1 s of motion and the eval `goal_offset` of 25 steps = 5 s of
  pushing — enough for meaningful box displacement.
- **Alignment contract:** at tick *t*, capture `pixels[t]`, `proprio[t]`, `state[t]`,
  **then** issue `action[t]`. So `pixels[t+1]` shows the outcome of `action[t]`.
- Log a `timestamp` column; ticks must be uniform to ±10% (see QA §8).

## 3. Action space — 2D Cartesian delta

- `action[t] = [Δx, Δy]` **float32, meters**, in the fixed table frame: the EE displacement
  commanded for this 0.2 s tick (equivalently velocity × 0.2 s).
- **Cap: |Δ| ≤ 0.025 m/step** (= 12.5 cm/s). Clip before sending; **record the clipped
  command actually sent** (not the raw teleop input, not the achieved motion).
- z, orientation, gripper: constant, handled by the controller, **not** in the action vector.
- No joint space (CEM samples Gaussians — unsafe + wastes model capacity on kinematics),
  no quaternions (not normalizable/sampleable; latent WMs capture rotation poorly).
- Units don't need pre-normalization — the pipeline z-scores per column
  (`utils.get_column_normalizer` for training, `StandardScaler` in `eval.py`).
  Just be **consistent**.
- `train.py` auto-sets `action_encoder.input_dim = frameskip × action_dim` (= 5×2 = 10),
  so a 2D action needs **zero model-config changes**.

## 4. Observation columns

| Column | Shape/dtype | Content |
|---|---|---|
| `pixels` | `(224, 224, 3) uint8` | RGB. Capture ≥ 640×480, center square-crop, resize to 224×224 (record crop/resize params and never change them mid-dataset). |
| `proprio` | `(4,) float32` | `[ee_x, ee_y, ee_vx, ee_vy]` — EE position (m) and velocity (m/s) in the table frame (mirrors PushT's layout). |
| `state` | `(6,) float32` | `[ee_x, ee_y, box_x, box_y, cos(box_yaw), sin(box_yaw)]` from AprilTag tracking. Used for normalization stats, latent probing, and analysis — **not** by the training loss. |
| `episode_idx` | `() int64` | Episode counter, monotone across the file. |
| `step_idx` | `() int64` | 0-based step within the episode. |
| `timestamp` | `() float64` | Unix time at image capture (QA only; extra columns are allowed and ignored by configs). |

**Missed tag detections:** write `NaN` into the box fields of `state`. Both `train.py`
(`utils.py:29`) and `eval.py` drop NaN rows before fitting normalizers, and the LeWM loss
only consumes `pixels` + `action`, so NaNs are end-to-end safe. Optionally interpolate gaps
≤ 3 frames; drop any episode with > 5% missing detections.

## 5. Episode structure

- **Length: 120–150 steps (24–30 s).** Hard floor 50 steps (PushT's min is 49; eval needs
  ≥ `goal_offset` 25 + margin), soft ceiling ~250.
- **1 hour of robot time = 18,000 steps ≈ 125–150 episodes.** With ~10–15 s manual
  re-randomization between episodes, budget ~1.5 h wall clock.
- Between episodes: re-randomize **box position over the whole arena, box yaw uniformly,
  and the EE start pose**. No resets *within* an episode.
- Training windows span `num_steps × frameskip = 4 × 5 = 20` steps, so every episode ≥ 50
  steps yields plenty of samples; episode boundaries are handled by the loader
  (NaN-padded actions are zeroed in `train.py:25`).

## 6. Collection policy (what the play should look like)

Teleop (or scripted random pusher) doing **unstructured, high-entropy play** — no task
success required, no demonstration quality bar (LeWM §3.1 allows "pseudo-expert or
exploratory" data; both real-robot reference papers used pure play):

- Repeat: approach the box from a random direction → push through contact 5–20 cm →
  release / reposition → repeat. Vary speed within the cap.
- Cover **all four faces and the corners** of the box (corner pushes produce rotation —
  you need rotational dynamics in the data).
- **~10–20% contact-free EE motion** (the model must learn that no contact ⇒ box stays).
- Coverage goals for the hour: box visits all arena quadrants; box yaw covers the full
  circle; EE approaches from all sides; typical 5 s contact window moves the box ≥ 3–5 cm
  or rotates it ≥ 10°.
- Keep hands/faces out of frame; if you must intervene mid-episode, end the episode.

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
        "timestamp":   ep_ts.astype(np.float64),       # (T,)
    })
```

Buffer each episode in RAM (150 steps × 150 KB ≈ 23 MB) and write on episode end.
Expected file size: ~2.7 GB uncompressed (fine for the pilot; the writer doesn't compress).
**Also keep the raw camera stream** (video/rosbag at native resolution) so you can regenerate
the h5 with different crops/rates without re-collecting.

## 8. QA checklist (run before training)

1. **Alignment:** `proprio[t+1, :2] − proprio[t, :2] ≈ action[t]` (correlation > 0.9 on
   contact-free segments). If it correlates with `action[t±1]` instead, your logging is
   off by one tick.
2. **Timing:** `diff(timestamp)` = 0.20 ± 0.02 s, no gaps > 0.3 s inside episodes.
3. **Tag health:** box-pose detection rate > 95%; no episode > 5% NaN.
4. **Visual sanity:** random 20 frames — box+EE fully visible, exposure constant, no motion
   blur that hides the box edge.
5. **Coverage:** heat-map `state[:, 2:4]` (box position) — should fill the arena, not one blob;
   histogram of `atan2(sin,cos)` yaw — roughly uniform.
6. **Loader smoke test:**
   `HDF5Dataset('pushbox_pilot_train', frameskip=5, num_steps=4, keys_to_load=[...])`
   loads, and one `DataLoader` batch has `pixels (B,4,3,224,224)`-ish shapes post-transform.

## 9. Training & pilot success criteria

Create `config/train/data/pushbox.yaml` (copy of `pusht.yaml` with
`name: pushbox_pilot_train.h5`), then:

```bash
python train.py data=pushbox trainer.max_epochs=100
```

Note on epochs: the paper's 10 epochs on 2.3M frames ≈ 160k optimizer steps; your pilot has
~16k windows, so 10 epochs ≈ 1.1k steps — far too few. ~100 epochs ≈ 11k steps is a
reasonable pilot budget (batch 128 may also need lowering if windows < 128×steps needed).

The pilot **passes** if:
1. `pred_loss` and `sigreg_loss` both decrease smoothly and plateau (cf. paper Fig. 18);
   sigreg drops sharply early.
2. **Linear probe** of box `(x, y)` (and cos/sin yaw) from frozen latents reaches high
   correlation (paper achieves r ≈ 0.99 position / 0.90 angle on sim PushT with full data —
   expect worse at 1 h, but position r ≳ 0.8 says the latent sees the box).
3. Optional smoke test of planning: load the checkpoint via
   `load_pretrained('<run>/weights_epoch_N.pt')`, build `WorldModelPolicy` + `CEMSolver`,
   feed a live camera frame + a goal image from the dataset, and check the planned 2D
   deltas are sane (bounded, pointed toward the box) before ever executing on the arm.

Real-robot closed-loop eval can't reuse `eval.py`'s dataset-driven protocol (`_set_state`
teleports don't exist in reality) — plan a small driver script that resets the box manually,
takes a goal image, and runs the policy with a step budget. That's a follow-up deliverable.
