"""
Camera viewer with manual recording support.

This script provides a camera viewer that can display multiple camera streams
and record them to video files with manual start/stop controls.

Features:
- Onscreen mode: Display camera feeds with optional recording
- Offscreen mode: No display, recording only when triggered
- Manual recording control with keyboard (R key to start/stop)
- Auto-recording triggered by ROS keyboard events (l/i/r/o keys from inference loop)

Usage Examples:

1. Basic onscreen viewing (with recording capability):
   python run_camera_viewer.py --camera-host localhost --camera-port 5555

2. Offscreen mode (no display, recording only):
   python run_camera_viewer.py --offscreen --camera-host localhost --camera-port 5555

3. Custom output directory:
   python run_camera_viewer.py --output-path ./my_recordings --camera-host localhost

Controls:
- R key: Start/Stop recording (manual, local keyboard)
- Q key: Quit application
- ROS keys (from inference loop): l=start recording, i/r/o=stop recording

Output Structure:
camera_output_20241211_143052/
├── rec_143205/
│   ├── ego_view_color_image.mp4
│   ├── head_left_color_image.mp4
│   └── head_right_color_image.mp4
└── rec_143410/
    ├── ego_view_color_image.mp4
    └── head_left_color_image.mp4
"""

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Optional

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sshkeyboard import listen_keyboard, stop_listening
import tyro

from gr00t_wbc.control.main.teleop.configs.configs import ComposedCameraClientConfig
from gr00t_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from gr00t_wbc.control.utils.img_viewer import ImageViewer
from gr00t_wbc.control.utils.keyboard_dispatcher import (
    KEYBOARD_LISTENER_TOPIC_NAME,
    KEYBOARD_RELIABLE_QOS,
)


@dataclass
class CameraViewerConfig(ComposedCameraClientConfig):
    """Config for running the camera viewer with recording support."""

    offscreen: bool = False
    """Run in offscreen mode (no display, manual recording with R key)."""

    output_path: Optional[str] = None
    """Output path for saving manual recordings. If None, uses 'camera_recordings'."""

    infer_video_path: Optional[str] = None
    """Output path for inference auto-recordings. If None, uses 'infer_video'."""

    codec: str = "mp4v"
    """Video codec to use for saving (e.g., 'mp4v', 'XVID')."""


ArgsConfig = CameraViewerConfig


def _get_camera_titles(image_data: dict[str, Any]) -> list[str]:
    """
    Detect all the individual camera streams from the image data.

    schema format:
    {
        "timestamps": {"ego_view": 123.45, "ego_view_left_mono": 123.46},
        "images": {"ego_view": np.ndarray, "ego_view_left_mono": np.ndarray}
    }

    Returns list of camera keys (e.g., ["ego_view", "ego_view_left_mono", "ego_view_right_mono"])
    """
    # Extract all camera keys from the images dictionary
    camera_titles = list(image_data.get("images", {}).keys())
    return camera_titles


def main(config: ArgsConfig):
    """Main function to run the camera viewer."""
    # Initialize ROS
    rclpy.init(args=None)
    node = rclpy.create_node("camera_viewer")

    # Use MultiThreadedExecutor so both camera node and keyboard subscriber get spun
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    # Subscribe to ROS keyboard events from the inference control loop
    # Create subscription directly on our node (which is managed by our executor)
    # instead of using KeyboardListenerSubscriber which attaches to the global executor
    from collections import deque
    _ros_key_queue = deque(maxlen=10)
    _ros_key_lock = threading.Lock()

    def _ros_key_callback(msg):
        with _ros_key_lock:
            _ros_key_queue.append(msg.data)
            print(f"[CameraViewer] Received ROS key '{msg.data}'")

    from std_msgs.msg import String as RosStringMsg
    node.create_subscription(RosStringMsg, KEYBOARD_LISTENER_TOPIC_NAME, _ros_key_callback, KEYBOARD_RELIABLE_QOS)

    def read_ros_key():
        with _ros_key_lock:
            if _ros_key_queue:
                return _ros_key_queue.popleft()
            return None

    image_sub = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    # pre-fetch a sample image to get the number of camera angles
    retry_count = 0
    while True:
        _sample_image = image_sub.read()
        if _sample_image:
            break
        time.sleep(0.1)
        if retry_count > 10:
            raise Exception("Failed to get sample image")

    camera_titles = _get_camera_titles(_sample_image)

    # Setup output directory
    if config.output_path is None:
        output_dir = Path("camera_recordings")
    else:
        output_dir = Path(config.output_path)

    # Recording state
    is_recording = False
    video_writers = {}
    frame_count = 0
    recording_start_time = None
    should_quit = False
    infer_video_dir = Path(config.infer_video_path) if config.infer_video_path else Path("./infer_video")

    if not infer_video_dir.exists():
        infer_video_dir.mkdir(parents=True, exist_ok=True)

    def _next_rec_index(parent_dir: Path) -> int:
        """Find the next available sequential recording index under parent_dir."""
        max_idx = -1
        print(parent_dir.iterdir())
        # if parent_dir.exists():
        #     for p in parent_dir.iterdir():
        #         suffix = p.name.split("_")[-1]
        #         if suffix.isdigit():
        #             max_idx = max(max_idx, int(suffix))
        # only files in parent_dir
        for p in parent_dir.iterdir():
            if p.is_file() and p.suffix == ".mp4":
                name_parts = p.stem.split("_")
                if name_parts[-1].isdigit():
                    max_idx = max(max_idx, int(name_parts[-1]))
        return max_idx + 1

    def start_recording(rec_output_dir: Path):
        nonlocal is_recording, video_writers, frame_count, recording_start_time
        rec_idx = _next_rec_index(rec_output_dir)

        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        video_writers.clear()

        for title in camera_titles:
            img = _sample_image["images"].get(title)
            if img is not None:
                height, width = img.shape[:2]
                video_path = rec_output_dir / f"{rec_idx}.mp4"
                writer = cv2.VideoWriter(
                    str(video_path), fourcc, config.fps, (width, height)
                )
                video_writers[title] = writer

        is_recording = True
        recording_start_time = time.time()
        frame_count = 0
        print(f"🔴 Recording started: {rec_output_dir}")

    def stop_recording():
        nonlocal is_recording, video_writers, frame_count, recording_start_time
        is_recording = False
        for title, writer in video_writers.items():
            writer.release()
        video_writers.clear()

        duration = time.time() - recording_start_time if recording_start_time else 0
        print(f"⏹️  Recording stopped - {duration:.1f}s, {frame_count} frames")

    def on_press(key):
        nonlocal should_quit

        if key == "r":
            if not is_recording:
                start_recording(output_dir)
            else:
                stop_recording()
        elif key == "q":
            should_quit = True
            stop_listening()

    # Setup keyboard listener in a separate thread
    keyboard_thread = threading.Thread(
        target=lambda: listen_keyboard(on_press=on_press), daemon=True
    )
    keyboard_thread.start()

    # Setup viewer for onscreen mode
    viewer = None
    if not config.offscreen:
        viewer = ImageViewer(
            title="Camera Viewer",
            figsize=(10, 8),
            num_images=len(camera_titles),
            image_titles=camera_titles,
        )

    # Print instructions
    mode = "Offscreen" if config.offscreen else "Onscreen"
    print(f"{mode} mode - Target FPS: {config.fps}")
    print(f"Manual recordings saved to: {output_dir}")
    print(f"Inference auto-recordings saved to: {infer_video_dir}")
    print("Controls: R key to start/stop recording, Q key to quit, Ctrl+C to exit")
    print("Auto-record: l=start, i/r/o=stop (via ROS keyboard topic)")

    # Create ROS rate controller
    rate = node.create_rate(config.fps)

    try:
        while rclpy.ok() and not should_quit:
            # Poll ROS keyboard events from inference control loop
            ros_key = read_ros_key()
            if ros_key == "l" and not is_recording:
                start_recording(infer_video_dir)
            elif ros_key in ("i", "r", "o") and is_recording:
                stop_recording()

            # Get images from all subscribers
            images = []
            image_data = image_sub.read()
            if image_data:
                for title in camera_titles:
                    img = image_data["images"].get(title)
                    images.append(img)

                    # Save frame if recording
                    if is_recording and img is not None and title in video_writers:
                        # Convert from RGB to BGR for OpenCV
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img
                        video_writers[title].write(img_bgr)

            # Display images if not offscreen
            if not config.offscreen and viewer and any(img is not None for img in images):
                status = "🔴 REC" if is_recording else "⏸️ Ready"
                viewer._fig.suptitle(f"Camera Viewer - {status}")
                viewer.show_multiple(images)

            # Progress feedback
            if is_recording:
                frame_count += 1
                if frame_count % 100 == 0:
                    duration = time.time() - recording_start_time
                    print(f"Recording: {frame_count} frames ({duration:.1f}s)")

            rate.sleep()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Cleanup
        try:
            stop_listening()
        except Exception:
            pass

        if video_writers:
            for title, writer in video_writers.items():
                writer.release()
            if is_recording:
                duration = time.time() - recording_start_time
                print(f"Final: {duration:.1f}s, {frame_count} frames")

        if viewer:
            viewer.close()

        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
