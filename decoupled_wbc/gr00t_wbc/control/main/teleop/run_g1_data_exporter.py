"""G1 Data Exporter for recording demonstrations in LeRobot format.

This script records robot demonstrations for training GR00T models. It subscribes
to robot state from the control loop and camera images from the camera server,
synchronizing them into episodes that can be used for imitation learning.

Architecture:
    Control Loop --[ROS2 /g1_state]--> This Script --[HDF5]--> Dataset
    Camera Server --[ZMQ]------------->

Recording States:
    IDLE -> RECORDING -> NEED_TO_SAVE -> IDLE

Keyboard/Button Controls:
    'c' or Button A: Start/stop recording
    'x' or Button B: Discard current episode

Usage:
    python run_g1_data_exporter.py \\
        --camera_host 192.168.123.164 \\
        --camera_port 5555 \\
        --data_collection_frequency 30 \\
        --hand_type aloha \\
        --root_output_dir ./real_robot_eval
"""

from collections import deque
from datetime import datetime
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
import tyro

from gr00t_wbc.control.main.constants import (
    DEFAULT_WRIST_POSE,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from gr00t_wbc.control.main.teleop.configs.configs import DataExporterConfig
from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
from gr00t_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from gr00t_wbc.control.utils.episode_state import EpisodeState
from gr00t_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber, ROSServiceClient
from gr00t_wbc.control.utils.telemetry import Telemetry
from gr00t_wbc.control.utils.text_to_speech import TextToSpeech
from gr00t_wbc.control.utils.img_viewer import ImageViewer
from gr00t_wbc.data.constants import BUCKET_BASE_PATH
from gr00t_wbc.data.exporter import (
    DataCollectionInfo,
    Gr00tDataExporter,
    _fix_directory_permissions,
)
from gr00t_wbc.data.utils import get_dataset_features, get_modality_config

import pdb


# =============================================================================
# Utility Functions
# =============================================================================


def _quat_wxyz_to_rpy(quat_wxyz):
    """Convert quaternion [w, x, y, z] to [roll, pitch, yaw]."""
    w, x, y, z = quat_wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


# =============================================================================
# Terminal Color Codes
# =============================================================================


class Colors:
    """ANSI color codes for terminal status display.

    Color Conventions:
        RED: IDLE state (stopped, ready to record)
        GREEN: RECORDING state (actively collecting data)
        YELLOW: SAVING state (processing/saving episode)
        CYAN: Moving to initial pose
        BLUE: Info messages
        MAGENTA: Episode numbers
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # State colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    # Background colors for emphasis
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_CYAN = "\033[46m"


# =============================================================================
# Timing Monitoring
# =============================================================================


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        """
        Exception raised when the time delta between two messages exceeds
        a threshold for a consecutive number of times
        """
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        """
        Monitor the time diff (between two messages) and optionally raise an exception
        if there is a consistent violations
        """
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in {self.reset_timeout_sec} seconds"
                f", time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


class Gr00tDataCollector:
    def __init__(
        self,
        node,
        camera_host: str,
        camera_port: int,
        state_topic_name: str,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency=30,
        state_act_msg_frequency=50,
        img_stream_viewer=False,
    ):

        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model
        self.img_stream_viewer = img_stream_viewer

        self.node = node

        thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        thread.start()
        time.sleep(0.5)

        self._episode_state = EpisodeState()
        self._keyboard_listener = KeyboardListenerSubscriber()
        self._state_subscriber = ROSMsgSubscriber(state_topic_name)
        # NOTE: We no longer publish to CONTROL_GOAL_TOPIC - TeleopPolicy handles initial pose
        self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)
        print(f"Image subscriber connected to {camera_host}:{camera_port}")
        self.rate = self.node.create_rate(self.frequency)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None

        self.state_polling_rate = 1 / state_act_msg_frequency
        self.last_state_poll_time = time.monotonic()

        self.telemetry = Telemetry(window_size=100)
        self.timing_threshold_monitor = TimingThresholdMonitor()
        self._last_timing_log = 0.0  # Rate limiter for timing logs
        self._last_status_reminder = 0.0  # Rate limiter for status reminders

        # Track whether we can delete previous episodes (set True after saving)
        self._can_delete_previous = False

        # Initialize image viewer if enabled
        self._image_viewer = None
        if self.img_stream_viewer:
            self._image_viewer = ImageViewer(
                title="Data Collection Camera Feed", num_images=1, image_titles=["Ego View"]
            )

        print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_status_banner(self, state: str, episode_num: int = None):
        """Print a large, colored status banner for the current recording state.

        Args:
            state: One of "idle", "recording", "need_to_save", or "moving"
            episode_num: Optional episode number to display
        """
        banner_width = 80

        if state == "idle":
            color = Colors.RED
            bg_color = Colors.BG_RED
            status_text = "⏸  IDLE - Ready to Record"
            if episode_num is not None:
                status_text += f" (Next: Episode {episode_num})"
        elif state == "recording":
            color = Colors.GREEN
            bg_color = Colors.BG_GREEN
            status_text = f"🔴 RECORDING - Episode {episode_num}"
        elif state == "need_to_save":
            color = Colors.YELLOW
            bg_color = Colors.BG_YELLOW
            status_text = f"💾 SAVING - Episode {episode_num}"
        elif state == "moving":
            color = Colors.CYAN
            bg_color = Colors.BG_CYAN
            status_text = "🤖 MOVING TO INITIAL POSE..."
        else:
            color = Colors.RESET
            bg_color = ""
            status_text = f"Unknown state: {state}"

        # Print banner
        print(f"\n{color}{Colors.BOLD}{'=' * banner_width}")
        print(f"{bg_color}{' ' * banner_width}{Colors.RESET}")
        print(f"{bg_color}{status_text.center(banner_width)}{Colors.RESET}")
        print(f"{bg_color}{' ' * banner_width}{Colors.RESET}")
        print(f"{color}{'=' * banner_width}{Colors.RESET}\n")

    # NOTE: Initial pose movement is now handled by TeleopPolicy
    # DataExporter simply waits for the state transition signal from control loop

    def _print_and_say(self, message: str, say: bool = True):
        """Helper to use TextToSpeech print_and_say or fallback to print."""
        # if self.text_to_speech is not None:
        #     self.text_to_speech.print_and_say(message, say)
        # else:
        #     print(message)

        print(message)

    def _check_keyboard_input(self):
        """Process all queued key presses - each button press is a discrete event."""
        # Debug: Print that we're checking (only if there's actually a message)
        with self._keyboard_listener._lock:
            queue_size = len(self._keyboard_listener._data_queue)
        if queue_size > 0:
            print(f"[DataExporter] _check_keyboard_input() called, queue size: {queue_size}")

        # Process all queued messages
        processed_count = 0
        max_per_iteration = 5  # Safety limit to prevent infinite loop

        while processed_count < max_per_iteration:
            key = self._keyboard_listener.read_msg()
            if key is None:
                break  # No more messages in queue

            processed_count += 1
            print(f"[DataExporter] Processing key from queue: '{key}'")

            if key == "c":
                current_state = self._episode_state.get_state()
                print(f"[DataExporter] Button 'c' pressed - current state: {current_state}")

                if current_state == self._episode_state.IDLE:
                    # IDLE -> RECORDING: TeleopPolicy moves arms to initial pose before sending this signal
                    self._can_delete_previous = False
                    self._episode_state.change_state()
                    self._print_status_banner("recording", self.current_episode_index)
                    self._print_and_say(f"Started recording episode {self.current_episode_index}")

                elif current_state == self._episode_state.RECORDING:
                    # RECORDING -> NEED_TO_SAVE: TeleopPolicy moves arms to initial pose before sending this signal
                    self._episode_state.change_state()
                    self._print_status_banner("need_to_save", self.current_episode_index)
                    self._print_and_say("Stopping recording, preparing to save")

                elif current_state == self._episode_state.NEED_TO_SAVE:
                    # NEED_TO_SAVE -> IDLE (shouldn't normally happen via button)
                    self._episode_state.change_state()
                    self._print_status_banner("idle", self.current_episode_index)
                    self._print_and_say("Saved episode and back to idle state")

            elif key == "x":
                print(f"[DataExporter] Button 'x' pressed - discarding episode")
                if self._episode_state.get_state() == self._episode_state.RECORDING:
                    # Discard: TeleopPolicy will move arms to initial pose, then discard
                    self.data_exporter.save_episode_as_discarded()
                    self._episode_state.reset_state()
                    self._print_status_banner("idle", self.current_episode_index)
                    self._print_and_say(
                        f"{Colors.RED}Episode {self.current_episode_index - 1} DISCARDED{Colors.RESET}"
                    )
            elif key == "d":
                print(f"[DataExporter] Button 'd' pressed - delete last saved episode")
                self._handle_delete_episode_request()
            elif key is not None:
                print(f"[DataExporter] Unknown key received: '{key}'")

    def _handle_delete_episode_request(self):
        """Handle episode deletion request from Y + Left Menu button press.

        Allows repeated deletion of episodes one-by-one (most recent first).
        Each press deletes the current most recent episode (total_episodes - 1).
        """
        current_state = self._episode_state.get_state()

        # Only allow deletion in IDLE state
        if current_state != self._episode_state.IDLE:
            print(
                f"[DataExporter] Cannot delete episode - current state is {current_state}, must be IDLE"
            )
            return

        if not self._can_delete_previous:
            print(f"[DataExporter] No saved episode to delete")
            return

        total_episodes = self.data_exporter.meta.total_episodes
        if total_episodes <= 0:
            print(f"[DataExporter] No episodes remaining to delete")
            self._can_delete_previous = False
            return

        # Delete the most recent remaining episode
        episode_to_delete = total_episodes - 1
        success = self.data_exporter.delete_episode(episode_to_delete)

        if success:
            remaining = self.data_exporter.meta.total_episodes
            self._print_status_banner("idle", self.current_episode_index)
            self._print_and_say(
                f"{Colors.RED}Episode {episode_to_delete} DELETED! ({remaining} episodes remaining){Colors.RESET}"
            )
            # Keep _can_delete_previous = True so next press can delete the next one
            if remaining <= 0:
                self._can_delete_previous = False
        else:
            self._print_and_say(
                f"{Colors.RED}Failed to delete episode {episode_to_delete}{Colors.RESET}"
            )

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self.latest_proprio_msg is None or self.latest_image_msg is None:
            # Waiting for initial messages - don't spam logs
            return False

        if self._episode_state.get_state() == self._episode_state.RECORDING:

            # Calculate max time delta between images and proprio
            max_time_delta = 0
            for _, image_time in self.latest_image_msg["timestamps"].items():
                time_delta = abs(image_time - self.latest_proprio_msg["timestamps"]["proprio"])
                max_time_delta = max(max_time_delta, time_delta)

            self.timing_threshold_monitor.log_time_delta(max_time_delta)
            if (self.timing_threshold_monitor.failure_count + 1) % 100 == 0:
                self._print_and_say("Image state delta too high, please discard data")

            # Compute torso RPY from quaternion and extract height
            torso_rpy = _quat_wxyz_to_rpy(self.latest_proprio_msg["torso_quat"])
            height = self.latest_proprio_msg["base_height_command"]
            navigate_cmd = self.latest_proprio_msg["navigate_command"]
            torso_orientation_rpy = self.latest_proprio_msg.get(
                "torso_orientation_rpy", [0.0, 0.0, 0.0]
            )

            # Extract joint group indices for selective recording
            q = self.latest_proprio_msg["q"]
            action_q = self.latest_proprio_msg["action"]
            rm = self.robot_model
            # Determine hand type (gripper for ALOHA, hand for three-finger)
            try:
                rm.get_joint_group_indices("left_gripper")
                hand_group = "gripper"
            except (KeyError, ValueError):
                hand_group = "hand"
            arm_gripper_indices = sorted(
                rm.get_joint_group_indices("left_arm")
                + rm.get_joint_group_indices("right_arm")
                + rm.get_joint_group_indices(f"left_{hand_group}")
                + rm.get_joint_group_indices(f"right_{hand_group}")
            )
            state_indices = sorted(
                rm.get_joint_group_indices("left_leg")
                + rm.get_joint_group_indices("right_leg")
                + rm.get_joint_group_indices("waist")
                + arm_gripper_indices
            )

            # state: left_leg + right_leg + waist + left_arm + right_arm + left_gripper + right_gripper + rpy + height
            state = np.concatenate(
                [
                    q[state_indices],
                    torso_rpy,
                    np.array([height]),
                ]
            ).astype(np.float64)

            # action: left_leg + right_leg + waist + left_arm + right_arm + left_gripper + right_gripper + rpy + height + vx + vy + vyaw + target_yaw
            action = np.concatenate(
                [
                    action_q[state_indices],
                    torso_rpy,
                    np.array([height]),
                    np.array(navigate_cmd),
                    np.array([torso_orientation_rpy[2]]),
                ]
            ).astype(np.float64)

            wrist_pose = np.asarray(
                self.latest_proprio_msg.get("action.eef", DEFAULT_WRIST_POSE),
                dtype=np.float64,
            )

            frame_data = {
                "observation.state": state,
                "action": action,
                "action.eef": wrist_pose,
                "observation.img_state_delta": (
                    np.array(
                        [max_time_delta],
                        dtype=np.float32,
                    )
                ),  # lerobot only supports adding numpy arrays
            }

            # Add images based on dataset features
            images = self.latest_image_msg["images"]
            for feature_name, feature_info in self.data_exporter.features.items():
                if feature_info.get("dtype") in ["image", "video"]:
                    # Extract image key from feature name (e.g., "observation.images.ego_view" -> "ego_view")
                    image_key = feature_name.split(".")[-1]

                    if image_key not in images:
                        raise ValueError(
                            f"Required image '{image_key}' for feature '{feature_name}' "
                            f"not found in image message. Available images: {list(images.keys())}"
                        )
                    frame_data[feature_name] = images[image_key]

            self.data_exporter.add_frame(frame_data)

        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            self.data_exporter.save_episode()
            self.timing_threshold_monitor.reset()
            self._episode_state.change_state()
            self._print_status_banner("idle", self.current_episode_index)
            self._print_and_say(
                f"{Colors.GREEN}✓ Episode {self.current_episode_index - 1} saved successfully!{Colors.RESET}"
            )
            # Enable deletion of previous episodes
            self._can_delete_previous = True

        return True

    def save_and_cleanup(self):
        try:
            self._print_and_say("saving episode done")
            # save on going episode if any
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
            self._print_and_say(f"Recording complete: {self.data_exporter.meta.root}", say=False)
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}")

        self.node.destroy_node()
        rclpy.shutdown()
        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        # Show initial status banner
        self._print_status_banner("idle", self.current_episode_index)
        print(f"{Colors.BLUE}Press Button A (or 'c') to start/stop recording{Colors.RESET}")
        print(f"{Colors.BLUE}Press Button B (or 'x') to discard episode{Colors.RESET}\n")

        try:
            print("rclpy.ok():", rclpy.ok())
            while rclpy.ok():
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    # 1. poll proprio msg
                    with self.telemetry.timer("poll_state"):
                        msg = self._state_subscriber.get_msg()
                        if msg is not None:
                            self.latest_proprio_msg = msg

                    # 2. poll image msg
                    with self.telemetry.timer("poll_image"):
                        msg = self._image_subscriber.read()
                        if msg is not None:
                            self.latest_image_msg = msg
                            # Display image in viewer if enabled
                            if self._image_viewer is not None:
                                if "video.ego_view" in msg:
                                    img = msg["video.ego_view"]
                                    self._image_viewer.show(img)
                                else:
                                    # Warn once if video.ego_view key is missing
                                    if not hasattr(self, "_img_key_warned"):
                                        self._img_key_warned = True
                                        print(
                                            f"Warning: 'video.ego_view' not in msg. Available keys: {list(msg.keys())}"
                                        )

                    # 3. check keyboard input
                    with self.telemetry.timer("check_keyboard"):
                        self._check_keyboard_input()

                    # 4. add frame
                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    end_time = time.monotonic()

                # Periodic status reminder (every 10 seconds)
                current_time = time.monotonic()
                if current_time - self._last_status_reminder > 10.0:
                    state = self._episode_state.get_state()
                    if state == self._episode_state.IDLE:
                        print(
                            f"{Colors.RED}[Status] ⏸  IDLE - Episode {self.current_episode_index} ready{Colors.RESET}"
                        )
                    elif state == self._episode_state.RECORDING:
                        print(
                            f"{Colors.GREEN}[Status] 🔴 RECORDING Episode {self.current_episode_index}{Colors.RESET}"
                        )
                    elif state == self._episode_state.NEED_TO_SAVE:
                        print(
                            f"{Colors.YELLOW}[Status] 💾 SAVING Episode {self.current_episode_index}...{Colors.RESET}"
                        )
                    self._last_status_reminder = current_time

                self.rate.sleep()

                # Log timing information only if significantly over budget (>100% miss)
                # Rate limited to once every 5 seconds to avoid log spam
                target_time = 1 / self.frequency
                current_time = time.monotonic()
                if (end_time - t_start) > (target_time * 2.0) and (
                    current_time - self._last_timing_log
                ) > 5.0:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed",
                        threshold=10.0,  # Only log very slow operations
                    )
                    self._last_timing_log = current_time

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            # The user will trigger a keyboard interrupt if there's something wrong,
            # so we flag the ongoing episode as discarded
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()


def main(config: DataExporterConfig):

    rclpy.init(args=None)
    node = rclpy.create_node("data_exporter")

    # Select robot model based on hand_type configuration
    robot_name = "g1_aloha" if config.hand_type == "aloha" else "g1"
    robot_type, robot_model = get_robot_type_and_model(
        robot=robot_name,
        high_elbow_pose=config.high_elbow_pose,
    )

    dataset_features = get_dataset_features(robot_model, config.add_stereo_camera)
    modality_config = get_modality_config(robot_model, config.add_stereo_camera)

    # text_to_speech = TextToSpeech() if config.text_to_speech else None
    text_to_speech = None

    # Only set DataCollectionInfo if we're creating a new dataset
    # When adding to existing dataset, DataCollectionInfo will be ignored
    if config.robot_id is not None:
        data_collection_info = DataCollectionInfo(
            teleoperator_username=config.teleoperator_username,
            support_operator_username=config.support_operator_username,
            robot_type=robot_type,
            robot_id=config.robot_id,
            lower_body_policy=config.lower_body_policy,
            wbc_model_path=config.wbc_model_path,
        )
    else:
        # Use default DataCollectionInfo when adding to existing dataset
        # This will be ignored if the dataset already exists
        data_collection_info = DataCollectionInfo()

    robot_config_client = ROSServiceClient(ROBOT_CONFIG_TOPIC)
    robot_config = robot_config_client.get_config()

    # Ensure output directory exists with proper permissions before creating exporter
    # This prevents permission errors when running in Docker with mounted volumes

    # First, ensure the base output directory exists
    base_output_path = Path(config.root_output_dir)
    if not base_output_path.exists():
        base_output_path.mkdir(parents=True, exist_ok=True)
        # If running as root in Docker, try to set ownership to host user
        # by looking at the parent directory (which should be mounted from host)
        if os.geteuid() == 0 and base_output_path.parent.exists():
            try:
                parent_stat = base_output_path.parent.stat()
                os.chown(base_output_path, parent_stat.st_uid, parent_stat.st_gid)
                print(
                    f"[Permissions] Set {base_output_path} ownership to uid={parent_stat.st_uid}, gid={parent_stat.st_gid}"
                )
            except Exception as e:
                print(f"[Permissions] Warning: Could not fix base directory ownership: {e}")

    # Now create the dataset-specific directory
    save_path = Path(f"{config.root_output_dir}/{config.dataset_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    # Fix permissions for the dataset directory, inheriting from the base directory
    _fix_directory_permissions(save_path, base_output_path)

    data_exporter = Gr00tDataExporter.create(
        save_root=str(save_path),
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        upload_bucket_path=BUCKET_BASE_PATH,
        data_collection_info=data_collection_info,
        script_config=robot_config,
    )

    data_collector = Gr00tDataCollector(
        node=node,
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=robot_model,
        state_topic_name=STATE_TOPIC_NAME,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        img_stream_viewer=config.img_stream_viewer,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(DataExporterConfig)

    # Ensure base output directory exists with proper permissions
    # This prevents "No such file or directory" and permission errors when saving episodes
    try:
        output_dir = Path(config.root_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test write permissions by creating a temporary file
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()

    except PermissionError as e:
        print(
            f"\n{Colors.RED}ERROR: Permission denied when accessing output directory: {config.root_output_dir}{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}The directory may be owned by root. Try one of these solutions:{Colors.RESET}"
        )
        print(f"  1. Fix permissions: sudo chown -R $USER:$USER {config.root_output_dir}")
        print(
            f"  2. Remove and recreate: sudo rm -rf {config.root_output_dir} && mkdir -p {config.root_output_dir}"
        )
        print(f"  3. Use a different directory: --root_output_dir ~/outputs")
        print(f"\nOriginal error: {e}")
        exit(1)
    except Exception as e:
        print(
            f"\n{Colors.RED}ERROR: Failed to create output directory: {config.root_output_dir}{Colors.RESET}"
        )
        print(f"Error: {e}")
        exit(1)

    # Default task prompt
    default_task_prompt = "pick the red cola and place it into the basket"
    print(f'Default task prompt: "{default_task_prompt}"')
    change_prompt = input("Change task prompt? (y/n) [n]: ").strip().lower()
    if change_prompt == "y":
        config.task_prompt = input("Enter new task prompt: ").strip().lower()
    else:
        config.task_prompt = default_task_prompt
        print(f'Using default task prompt: "{config.task_prompt}"')

    add_to_existing_dataset = input("Add to existing dataset? (y/n): ").strip().lower()

    if add_to_existing_dataset == "y":
        config.dataset_name = input("Enter the dataset name: ").strip().lower()
        # When adding to existing dataset, we don't need robot_id or operator usernames
        # as they should already be set in the existing dataset
    elif add_to_existing_dataset == "n":
        # robot_id = input("Enter the robot ID: ").strip().lower()
        # if robot_id not in G1_ROBOT_IDS:
        #     raise ValueError(f"Invalid robot ID: {robot_id}. Available robot IDs: {G1_ROBOT_IDS}")
        config.robot_id = "sim"
        config.dataset_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-G1-{config.robot_id}"

        # Only ask for operator usernames when creating a new dataset
        # print("Available teleoperator usernames:")
        # for i, username in enumerate(OPERATOR_USERNAMES):
        #     print(f"{i}: {username}")
        # teleop_idx = int(input("Select teleoperator username index: "))
        # config.teleoperator_username = OPERATOR_USERNAMES[teleop_idx]
        config.teleoperator_username = "NEW_USER"

        # print("\nAvailable support operator usernames:")
        # for i, username in enumerate(OPERATOR_USERNAMES):
        #     print(f"{i}: {username}")
        # support_idx = int(input("Select support operator username index: "))
        # config.support_operator_username = OPERATOR_USERNAMES[support_idx]
        config.support_operator_username = "NEW_USER"

    # pdb.set_trace()
    main(config)
