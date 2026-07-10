# Agent Execution Guide — Supervised Keyboard XY Collection

This is the onboarding brief for an agent implementing
[superpowers/plans/2026-07-10-supervised-keyboard-xy-collection.md](superpowers/plans/2026-07-10-supervised-keyboard-xy-collection.md).
Read it fully before writing code, then keep the plan open as the authoritative
spec. Hardware/API facts you will need live in
[hardware-api-reference.md](hardware-api-reference.md).

## Your task

Build a supervised, PushT-like data collector: a human drives a Trossen follower
arm with bounded table-frame XY keyboard actions at 5 Hz, the system holds a
fixed tool pose (fixed Z, orientation, gripper), records aligned
`pixels`/`proprio`/`state`/`action` episodes to HDF5, detects manual-reset
conditions, and reports coverage. The dataset must satisfy the contract in
[../dataset_spec.md](../dataset_spec.md).

## Execution model

The plan is **9 tasks**, done **in order** (later tasks import earlier modules).
Use the superpowers sub-skill `superpowers:subagent-driven-development`
(recommended) or `superpowers:executing-plans` to work task-by-task. Every task
is strict TDD:

1. Write the failing tests from the task.
2. Run them, confirm they fail for the stated reason.
3. Implement until they pass.
4. Commit **only** that task's named files.

Task checkboxes (`- [ ]`) track progress; tick them as you go.

## Repository orientation

### What already exists — do NOT rewrite

- `scripts/collect_data.py` — the leader/follower teleop prototype. It **stays**
  as a setup-validation prototype. Task 9 fixes only its action-recording
  docstring; nothing else touches it.
- `scripts/teleoperation.py`, `scripts/record_realsense.py` — reference scripts
  for driver and camera usage. Read them to learn the real API shapes; do not
  import or depend on them.
- `dataset_spec.md`, `README.md` — the dataset contract and repo overview.

### What you build

A new `pushbox_collect/` package, `tests/`, `config/`, and three `scripts/`
entry points. See the plan's **Planned File Structure**. Note that **Task 5 is a
camera adapter + Cartesian EE parking task** — an earlier draft placed a
gravity-reset module there; it was removed (see Decision log below).

### What to leave alone

- The prototype scripts (except the one docstring fix in Task 9).
- The working-tree `.gitignore` (it ignores `papers/`, `*.pdf`, `ref_video/`).
  Never `git add -A`; stage only the files each task names.

## Environment setup

- Python **3.12**, managed with `uv`.
- **The pinned `pyrealsense2` / `trossen-arm` wheels do not install on
  macOS/Apple Silicon.** Task 1 splits `pyproject.toml` so those two live in an
  optional `hardware` group:
  - Dev laptop (macOS): `uv sync` — core deps only; runs the hardware-free tests.
  - Robot host (Linux): `uv sync --extra hardware` — adds camera + arm drivers.
- Run tests with `python -m unittest tests.<name> -v`, or the whole suite with
  `python -m unittest discover -s tests`.
- **Every test is hardware-free by design** — cameras and arms are injected
  fakes. Do NOT import `trossen_arm`, open the RealSense pipeline, or move the arm
  on the dev host. Those are gated to the robot host by the Task 10 commissioning
  checklist.

## Decision log — do not reopen

Locked with the operator on 2026-07-10:

1. **Control interface: non-latching keyboard, tuned step.** Keep the keyboard.
   `keyboard_step_m` starts at 0.005 m (≈2.5 cm/s at 5 Hz) and is raised toward
   the 0.025 m action cap only after a no-contact commissioning test. (Rejected
   alternatives: decoupled command/record rate; leader-driven Cartesian teleop.)
2. **No gravity-compensation / hand-guided mode in V1.** The `RESET_GRAVITY`
   state was removed. The Trossen gravity-comp pattern back-drives *all* joints
   including the gripper and cannot enforce XY-only guiding, so it is unsafe and
   unsuitable here. Instead the EE is re-randomized between episodes by
   fixed-pose Cartesian **position** commands (`FollowerRobot.move_to_start`).
   The arm is only ever in position mode.
3. **The pusher is rigidly mounted in place of the gripper** (not gripper-held).
   This is a hard V1 requirement — it is what makes "no effort mode" safe, since
   the gripper is never load-bearing. Confirm the physical mount before
   commissioning.
4. **Record rate stays 5 Hz** (0.2 s tick); config rejects any other tick value.

## Keyboard map (V1)

Single-byte, non-latching — each key authorizes exactly one bounded 5 Hz tick:

| Key | Action |
|---|---|
| `w` / `a` / `s` / `d` | +table-Y / −table-X / −table-Y / +table-X |
| `Space` | arm / end episode |
| `Backspace` | discard the current episode |
| `x` or `Esc` | abort (immediate hold → `ABORTED`, keep audit record) |
| `q` | graceful shutdown (from `IDLE/PARKED_CLEAR` only) |

No arrow keys (their escape sequences collide with `Esc`), no diagonals
(staircase X and Y over ticks), no gravity key.

## Safety non-negotiables

- A physical **E-stop** and a supervised operator are mandatory. Keyboard input
  is **not** an E-stop and must never be treated as one.
- The arm **never** enters external-effort mode in V1 — only fixed-pose Cartesian
  position control.
- A recorded episode must never contain an out-of-view box, a stale/missing scene
  estimate, operator intervention, an unhandled exception, a timing overrun, or a
  safety rejection. Such runs are audit artifacts and are discarded from training
  data.
- Record `pixels[t]`/`proprio[t]`/`state[t]` **before** issuing `action[t]`, and
  record the **guardian-approved command at issue time** — never a later measured
  displacement.
- `pushbox_collect.robot.FollowerRobot` is the **only** module allowed to import
  `trossen_arm`.

## Git workflow

- The operator drives `git push`. Do not force-push or rewrite shared history.
- **One commit per task**, using the exact `git add <files>` list in that task —
  never `git add -A`, never stage the whole tree.
- Preserve the existing `.gitignore` change already in the tree.

## Definition of done

- Per task: its tests pass under `python -m unittest`, and only that task's files
  are committed.
- Overall (software): all hardware-free tests green; CLI (`collect_keyboard_xy.py`)
  parses and reaches `PRECHECK` without hardware.
- Live collection is **not** completed here — it happens on the robot host through
  the ascending-risk commissioning gates in Task 10.
