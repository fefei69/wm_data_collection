# docs/ — Supervised Keyboard XY Collection

Documentation for building the supervised keyboard XY data collector — the
successor to the `scripts/collect_data.py` leader/follower prototype. A human
teleoperates a Trossen follower arm with bounded table-frame XY keyboard actions
at 5 Hz; the system enforces a fixed tool pose, records aligned
pixels/proprio/action episodes to HDF5, and reports coverage deficits.

## If you are an agent executing the plan, read in order

1. **[agent-execution-guide.md](agent-execution-guide.md)** — orientation,
   environment setup, the locked design decisions, the git/TDD workflow, and the
   safety rules. **Start here.**
2. **[hardware-api-reference.md](hardware-api-reference.md)** — the verified
   Trossen driver / RealSense API facts, and the parts that are *commissioning-gated*
   (must be confirmed on the robot host, never guessed).
3. **[superpowers/plans/2026-07-10-supervised-keyboard-xy-collection.md](superpowers/plans/2026-07-10-supervised-keyboard-xy-collection.md)**
   — the authoritative, task-by-task implementation plan (9 tasks, strict TDD).

## Related (repo root)

- [../dataset_spec.md](../dataset_spec.md) — the dataset contract (rig, timing,
  action space, HDF5 schema, QA checklist) the collector must satisfy.
- [../README.md](../README.md) — repo overview and the existing prototype scripts.

## Status

Plan finalized 2026-07-10. Nothing under `pushbox_collect/` is implemented yet;
**Task 1** (config contract + hardware-free test harness + dependency split) is
the next step. Implementation, camera, and arm code are all still to be written.
