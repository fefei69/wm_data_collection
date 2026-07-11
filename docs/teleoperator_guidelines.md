# Teleoperator Guidelines — Push-Box Collection

Use this checklist while collecting the 5 Hz push-box dataset. The goal is
diverse, accurately recorded interaction—not task success or polished
demonstrations.

## Before the session

- Keep the physical E-stop reachable and supervise the arm continuously.
- Clear the table before startup and confirm the commissioned safe path to the
  fixed-height collection pose.
- Source ROS 2 Jazzy and start the RealSense publisher.
- Close browsers and other heavy applications; desktop load starves the image
  subscriber and silently kills episodes (measured 2026-07-11).
- Confirm `/camera/camera/color/image_raw` is a healthy
  `sensor_msgs/msg/Image` stream: `rgb8`, 640×480, approximately 60 fps.

```bash
ros2 topic info -v /camera/camera/color/image_raw
uv run scripts/check_camera_health.py   # must print "verdict: HEALTHY"
```

  The collector repeats this gate at startup; `--skip-camera-check` bypasses
  it if you accept the risk of frequent freshness-based episode discards.

- Confirm supported exposure, white-balance, gain, and focus controls are fixed
  on the publisher; save `ros2 param dump <publisher_node>` with the session.
- In a separate terminal, start a session-level rosbag2 recording of
  `/camera/camera/color/image_raw` before the first episode. This is the 60 fps
  source archive; the collector's per-episode MP4 contains only selected 5 Hz
  model frames.
- Confirm the session's commissioned square image-transform profile, then
  inspect the collector's exact 224×224 model-view preview. The useful work
  area, box, and pusher must fit.
- Confirm exposure, white balance, focus, gain, camera mount, lighting, and
  image preprocessing are unchanged from earlier sessions.
- With the tool raised and no box, verify every cardinal/diagonal direction,
  all three speed levels, opposing-key cancellation, fixed orientation, XY
  bounds, focus-loss stop, and reset path.
- Check the pusher height across the usable XY workspace before allowing
  contact.

## Controls

| Key | Effect |
|---|---|
| Held arrow keys | One bounded Cartesian step in base X/Y; two orthogonal arrows make a normalized diagonal |
| `1` | Select 2.5 mm/step |
| `2` | Select 5 mm/step (default) |
| `3` | Select 10 mm/step |
| `SPACE` | Start recording, or gracefully finish and save an episode |
| `d` | Stop and discard the active episode |
| `r` | While idle, return toward the configured XY start |
| `q` | Gracefully finish, close resources, and begin supervised shutdown |

No held arrow means a recorded zero-delta hold. Opposing arrows cancel their
axis. Never use `r` while recording. Keep the Pygame preview window focused;
focus loss stops motion and discards an active episode.

## What good episodes contain

- Use episodes of roughly 24–30 seconds (120–150 samples).
- Start with varied box positions, box orientations, and pusher positions.
- Approach and contact every side of the box over the session.
- Include corner contacts and off-center pushes that rotate the box.
- Include pushes in both camera-near and camera-far directions.
- Use simultaneous orthogonal arrows to collect all four normalized diagonal
  actions as well as the four cardinals.
- Use all three speed levels during both contact and contact-free movement.
- Vary push duration and direction; include reversals and releases.
- Make approximately 10–20% of the motion deliberately contact-free.
- Include approach, first contact, sustained contact, separation, and periods
  where the box remains still.
- Prefer smooth, purposeful motion while still covering unusual interactions.

## Do

- Watch the exact model preview, not only the raw camera image.
- Check the displayed speed level before pressing an arrow, especially after
  using the 10 mm setting.
- Keep the entire box and pusher usefully visible whenever possible.
- Let each commanded move finish through the normal 5 Hz loop.
- End the episode before manually touching or resetting the box.
- Reset the box only while recording is off.
- Discard questionable episodes immediately; clean data is more valuable than
  maximizing the saved episode count.
- Write a short session note for lighting changes, camera movement, tool
  changes, faults, or unusual contacts.

## Don't

- Don't chase a task-success score or repeat one preferred pushing strategy.
- Don't let 5 mm cardinal actions dominate; deliberately cover every direction
  and magnitude.
- Don't collect only contact motion; the model must also learn no-contact
  dynamics.
- Don't let the box remain outside the useful view or reachable workspace.
- Don't place hands, tools, or unrelated objects in a recorded episode.
- Don't move the camera, change its profile, or change image preprocessing
  during a collection campaign.
- Don't continue after a stale-frame warning, ROS interruption, rejected arm
  command, unexpected collision, or arm pose-tracking fault.
- Don't save an episode containing manual intervention, a timing gap, or an
  uncertain command/action label.

## End or discard an episode when

- the box or pusher is substantially outside the model view;
- a person must touch the box or enter the image;
- the camera frame is stale, reused, corrupted, or the ROS stream stops;
- the Pygame control window loses focus during recording;
- the arm rejects a command, faults, or behaves differently from the command;
- an unexpected collision or unsafe configuration occurs; or
- the operator is unsure whether observations and actions remained aligned.

After a normal episode, review occasional MP4/HDF5 samples throughout the
session rather than waiting until the entire collection is finished. Across
the dataset, confirm broad box position/orientation coverage, all approach
sides, contact and no-contact behavior, rotations, reversals, and varied push
durations.
