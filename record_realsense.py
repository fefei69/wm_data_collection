"""Record color + depth video from a RealSense D435i.

Usage:
    uv run record_realsense.py [output.mp4] [duration_seconds]

Saves a side-by-side (color | depth colormap) mp4. Shows a live preview
window if a display is available. Press 'q' in the window or Ctrl+C to stop.
"""

import os
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30


def open_h264_writer(out_path, width, height, fps):
    """Pipe raw BGR frames to ffmpeg, encoding H.264 (plays in VSCode/browsers)."""
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
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "recording.mp4"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    device = profile.get_device()
    print(f"Recording from: {device.get_info(rs.camera_info.name)} "
          f"(S/N {device.get_info(rs.camera_info.serial_number)})")

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    writer = open_h264_writer(out_path, WIDTH * 2, HEIGHT, FPS)

    show_preview = bool(os.environ.get("DISPLAY"))
    if not show_preview:
        print("No display detected — recording headless (Ctrl+C to stop).")

    start = time.time()
    frame_count = 0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            combined = np.hstack((color, depth_vis))
            writer.stdin.write(combined.tobytes())
            frame_count += 1

            if show_preview:
                cv2.imshow("RealSense (color | depth) - press q to stop", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - start
            if duration and elapsed >= duration:
                break
            if frame_count % FPS == 0:
                print(f"\r{elapsed:.1f}s  {frame_count} frames", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        writer.stdin.close()
        writer.wait()
        pipeline.stop()
        if show_preview:
            cv2.destroyAllWindows()
        print(f"\nSaved {frame_count} frames "
              f"({frame_count / FPS:.1f}s) to {out_path}")


if __name__ == "__main__":
    main()
