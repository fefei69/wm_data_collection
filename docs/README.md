# docs/ — Keyboard XY collection

The implementation target is deliberately small:

1. [keyboard_xy_collection_spec.md](keyboard_xy_collection_spec.md) — the
   complete code spec for a single-file, arrow-key Cartesian collector.
2. [hardware-api-reference.md](hardware-api-reference.md) — the small set of
   Trossen and ROS 2 Jazzy camera-interface facts to verify during hardware
   commissioning.
3. [teleoperator_guidelines.md](teleoperator_guidelines.md) — the short
   before/during/after runbook and dataset-quality dos and don'ts.

Related root documents:

- [../dataset_spec.md](../dataset_spec.md) — dataset schema and collection policy.
- [../README.md](../README.md) — current scripts and repository overview.

The existing `scripts/collect_data.py` remains the leader/follower prototype;
`scripts/collect_keyboard_xy.py` is the implemented follower-only ROS 2/Pygame
collector. Hardware-specific pose bounds and image profiles still require
commissioning before a robot run.
