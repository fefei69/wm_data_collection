"""Minimal single-file data collection for the push-box pilot (see dataset_spec.md).

Leader/follower teleoperation (Trossen WXAI) + one RealSense color camera,
recording 5 Hz episodes to HDF5. No ROS. Prototype to validate the physical
setup — box tracking (AprilTag) is not implemented yet, so the box fields of
`state` are NaN (safe end-to-end per the spec).

Usage (run in a real terminal):
    uv run collect_data.py [output.h5]

Keys (terminal, or the preview window if a display is attached):
    SPACE  start recording an episode / stop and save it
    d      discard the episode currently being recorded
    q      quit (saves an in-progress episode), arms go home -> sleep

Outputs:
    output.h5                    episodes appended, spec §4/§7 columns
    output_videos/ep_NNN.mp4     raw 640x480@30 color stream per episode,
                                 H.264 + yuv420p (plays in VS Code/browsers)

Prototype deviations from the spec, to fix before the real collection:
    - camera auto-exposure/white-balance left ON (lock them for the dataset)
    - table frame == arm base frame (no AprilTag world frame)
"""

import os
import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
import pyrealsense2 as rs

import trossen_arm

# Arms (same rig as teleoperation.py)
LEADER_IP = '192.168.1.5'
FOLLOWER_IP = '192.168.1.3'
FORCE_FEEDBACK_GAIN = 0.1
HOME_POSITIONS = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0])

# Dataset (dataset_spec.md §2-§4)
TICK = 0.2               # 5 Hz decision rate
ACTION_CAP = 0.025       # |delta| <= 0.025 m per step
IMG_SIZE = 224
MIN_EPISODE_STEPS = 50

# Camera
CAM_W, CAM_H, CAM_FPS = 640, 480, 30


def open_h264_writer(out_path, width, height, fps):
    """Pipe raw BGR frames to ffmpeg, encoding H.264 (plays in VS Code/browsers)."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)


def process_frame(color_bgr):
    """640x480 BGR -> center square crop -> 224x224 RGB uint8 (spec §4)."""
    h, w = color_bgr.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    crop = color_bgr[y0:y0 + s, x0:x0 + s]
    small = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def build_episode_arrays(buf, ep_id):
    """Turn per-tick buffers into the spec §4 columns for one episode.

    action[t] = clipped delta of the leader EE xy (the commanded target the
    follower tracks) between tick t and t+1; the last action is zero.
    """
    T = len(buf['pixels'])
    proprio = np.asarray(buf['proprio'], np.float32)
    leader_xy = np.asarray(buf['leader_xy'], np.float64)

    action = np.zeros((T, 2), np.float32)
    if T > 1:
        action[:-1] = np.diff(leader_xy, axis=0)
    np.clip(action, -ACTION_CAP, ACTION_CAP, out=action)

    # Box pose comes from AprilTag tracking, not implemented in the prototype.
    state = np.full((T, 6), np.nan, np.float32)
    state[:, 0:2] = proprio[:, 0:2]

    return {
        'pixels': np.asarray(buf['pixels'], np.uint8),
        'action': action,
        'proprio': proprio,
        'state': state,
        'episode_idx': np.full(T, ep_id, np.int64),
        'step_idx': np.arange(T, dtype=np.int64),
        'timestamp': np.asarray(buf['timestamp'], np.float64),
    }


def append_episode(h5, columns):
    """Append one episode's arrays to resizable datasets (created on first use)."""
    n_new = len(columns['action'])
    for name, arr in columns.items():
        if name not in h5:
            chunks = (1,) + arr.shape[1:] if arr.ndim > 1 else True
            h5.create_dataset(name, data=arr,
                              maxshape=(None,) + arr.shape[1:], chunks=chunks)
        else:
            d = h5[name]
            d.resize(d.shape[0] + n_new, axis=0)
            d[-n_new:] = arr
    h5.flush()


def next_episode_idx(h5):
    if 'episode_idx' in h5 and h5['episode_idx'].shape[0] > 0:
        return int(h5['episode_idx'][-1]) + 1
    return 0


def new_buffers():
    return {'pixels': [], 'proprio': [], 'leader_xy': [], 'timestamp': []}


def main():
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else 'pushbox_prototype.h5')
    video_dir = out_path.with_name(out_path.stem + '_videos')
    video_dir.mkdir(parents=True, exist_ok=True)

    # Camera first: fail fast before moving any arm.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
    profile = pipeline.start(config)
    device = profile.get_device()
    print(f"Camera: {device.get_info(rs.camera_info.name)} "
          f"(S/N {device.get_info(rs.camera_info.serial_number)})")

    h5 = h5py.File(out_path, 'a')
    ep_id = next_episode_idx(h5)
    print(f"Dataset: {out_path} (next episode: {ep_id})")

    print("Initializing the drivers...")
    driver_leader = trossen_arm.TrossenArmDriver()
    driver_follower = trossen_arm.TrossenArmDriver()

    print("Configuring the drivers...")
    driver_leader.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        LEADER_IP,
        False
    )
    
    driver_follower.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_follower,
        FOLLOWER_IP,
        False
    )

    print("Moving to home positions...")
    driver_leader.set_all_modes(trossen_arm.Mode.position)
    driver_leader.set_all_positions(HOME_POSITIONS, 2.0, True)
    driver_follower.set_all_modes(trossen_arm.Mode.position)
    driver_follower.set_all_positions(HOME_POSITIONS, 2.0, True)

    print("Starting teleoperation.")
    print("  SPACE: start / stop+save episode   d: discard episode   q: quit")
    time.sleep(1)
    driver_leader.set_all_modes(trossen_arm.Mode.external_effort)
    driver_follower.set_all_modes(trossen_arm.Mode.position)

    show_preview = bool(os.environ.get('DISPLAY'))
    if not show_preview:
        print("No display detected — running without preview.")

    recording = False
    buf = new_buffers()
    video_proc = None
    video_path = None
    latest_frame = None   # keep the rs frame alive so latest_color stays valid
    latest_color = None
    next_tick = time.monotonic()

    def start_episode():
        nonlocal recording, buf, video_proc, video_path
        video_path = video_dir / f'ep_{ep_id:03d}.mp4'
        video_proc = open_h264_writer(video_path, CAM_W, CAM_H, CAM_FPS)
        buf = new_buffers()
        recording = True
        print(f"\n[episode {ep_id}] recording...")

    def close_video():
        nonlocal video_proc
        if video_proc is not None:
            video_proc.stdin.close()
            video_proc.wait()
            video_proc = None

    def stop_episode(save):
        nonlocal recording, ep_id
        recording = False
        close_video()
        T = len(buf['pixels'])
        if not save or T == 0:
            if video_path is not None and video_path.exists():
                video_path.unlink()
            print(f"\n[episode {ep_id}] discarded ({T} steps)")
            return
        append_episode(h5, build_episode_arrays(buf, ep_id))
        note = '' if T >= MIN_EPISODE_STEPS else \
            f'  (WARNING: below the {MIN_EPISODE_STEPS}-step floor)'
        print(f"\n[episode {ep_id}] saved: {T} steps ({T * TICK:.1f}s), "
              f"video: {video_path}{note}")
        ep_id += 1

    stdin_fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)
    try:
        while True:
            # --- teleop feed (runs as fast as the drivers allow) ---
            driver_leader.set_all_external_efforts(
                -FORCE_FEEDBACK_GAIN * np.array(driver_follower.get_all_external_efforts()),
                0.0,
                False,
            )
            driver_follower.set_all_positions(
                driver_leader.get_all_positions(),
                0.0,
                False,
                driver_leader.get_all_velocities()
            )

            # --- camera: grab the newest frame without blocking ---
            frames = pipeline.poll_for_frames()
            if frames:
                cf = frames.get_color_frame()
                if cf:
                    latest_frame = cf
                    latest_color = np.asanyarray(cf.get_data())
                    if recording and video_proc is not None:
                        video_proc.stdin.write(latest_color.tobytes())

            # --- keyboard (terminal, plus preview window below) ---
            key = None
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()

            # --- 5 Hz tick: record observation, refresh preview ---
            now = time.monotonic()
            if now >= next_tick:
                next_tick += TICK
                if now >= next_tick:          # fell behind; resync
                    next_tick = now + TICK

                if recording and latest_color is not None:
                    ts = time.time()
                    pixels = process_frame(latest_color)
                    cart_f = driver_follower.get_cartesian_positions()
                    vel_f = driver_follower.get_cartesian_velocities()
                    cart_l = driver_leader.get_cartesian_positions()
                    buf['pixels'].append(pixels)
                    buf['proprio'].append(
                        [cart_f[0], cart_f[1], vel_f[0], vel_f[1]])
                    buf['leader_xy'].append([cart_l[0], cart_l[1]])
                    buf['timestamp'].append(ts)
                    if len(buf['pixels']) % 25 == 0 and not show_preview:
                        print(f"\r[episode {ep_id}] {len(buf['pixels'])} steps",
                              end='', flush=True)

                if show_preview and latest_color is not None:
                    view = cv2.cvtColor(process_frame(latest_color),
                                        cv2.COLOR_RGB2BGR)
                    view = cv2.resize(view, (448, 448),
                                      interpolation=cv2.INTER_NEAREST)
                    label = (f"REC ep {ep_id}  step {len(buf['pixels'])}"
                             if recording else "idle - SPACE to record")
                    cv2.putText(view, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255) if recording else (0, 255, 0), 2)
                    cv2.imshow('collect_data (model view) - q to quit', view)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord(' '), ord('q'), ord('d')):
                        key = chr(k).lower()

            # --- handle keys ---
            if key == 'q':
                if recording:
                    stop_episode(save=True)
                break
            elif key == ' ':
                if recording:
                    stop_episode(save=True)
                elif latest_color is None:
                    print("\nNo camera frame yet — try again.")
                else:
                    start_episode()
            elif key == 'd' and recording:
                stop_episode(save=False)
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_terminal_settings)
        close_video()
        h5.close()
        pipeline.stop()
        if show_preview:
            cv2.destroyAllWindows()

    print("Moving to home positions...")
    driver_leader.set_all_modes(trossen_arm.Mode.position)
    driver_leader.set_all_positions(HOME_POSITIONS, 2.0, True)
    driver_follower.set_all_modes(trossen_arm.Mode.position)
    driver_follower.set_all_positions(HOME_POSITIONS, 2.0, True)

    print("Moving to sleep positions...")
    driver_leader.set_all_positions(
        np.zeros(driver_leader.get_num_joints()), 2.0, True)
    driver_follower.set_all_positions(
        np.zeros(driver_follower.get_num_joints()), 2.0, True)


if __name__ == '__main__':
    main()
