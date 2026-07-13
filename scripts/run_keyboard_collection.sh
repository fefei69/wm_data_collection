#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root so the profile and parameter paths resolve.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d_%H%M%S)"
output="data/pushbox_keyboard_${timestamp}.h5"
video_dir="output_videos/${timestamp}"
check_seconds=5

cat <<EOF

Push-box keyboard collector
===========================

Motion (keep the Pygame preview focused):
  UP / DOWN     +X / -X in the robot base frame
  LEFT / RIGHT  +Y / -Y in the robot base frame
  Two arrows    normalized diagonal motion

Speed:
  1             2.5 mm per step
  2             5 mm per step (default)
  3             10 mm per step

Episode controls:
  SPACE         start recording; press again to stop and save
  d             stop and discard the active episode
  r             while idle, return to the configured start X/Y
  q             quit while idle (ignored while recording)

Timing:
  Elapsed recording time prints every 5 seconds.
  Aim to stop each episode between 24 and 30 seconds.

Safety:
  X/Y workspace limits are disabled for this launcher.
  Losing window focus stops motion and discards an active episode.
  On q/window-close while idle, the arm moves home, then to all-zero joints.
  Keep an emergency-stop hand ready; post-move tracking is not yet verified.

Dataset: $output
Videos:  $video_dir/

Starting with a ${check_seconds}-second camera health check...

EOF

# --disable-xy-limits: the X/Y bounds passed below are not yet commissioned on
# this table; drop the flag to enable clamping once they are verified. Keep the
# defaults in scripts/collect_keyboard_xy.py (CollectorConfig) in sync then.
uv run scripts/collect_keyboard_xy.py \
  --transform-profile config/transform-profile.json \
  --camera-params config/camera-params.yaml \
  --fixed-z 0.03 \
  --safe-z 0.15 \
  --start-x 0.14 \
  --start-y 0.0185 \
  --trajectory-check-samples 10 \
  --camera-check-seconds "$check_seconds" \
  --disable-xy-limits \
  --output "$output" \
  --video-dir "$video_dir" \
  "$@"
  # --skip-camera-check \
