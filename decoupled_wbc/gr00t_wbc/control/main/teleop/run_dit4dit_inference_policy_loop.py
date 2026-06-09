"""DiT4DiT Inference Policy Loop for deploying finetuned models to real robots.

This script connects to a DiT4DiT inference server (ZMQ) and publishes
predicted actions to the G1 control loop via ROS2. It handles:
- Connection to remote DiT4DiT inference server (ZMQ + msgpack)
- Action denormalization using dataset statistics (server-side)
- Action caching with configurable execution horizon
- Smooth transitions between initial pose and inference modes
- Gripper value conversion between training and hardware conventions

Keyboard Controls:
    'i': Move to initial pose (open grippers)
    'l': Start inference (smooth transition from current pose)
    'o': Stop inference
    'r': Reset (clear cache and move to initial pose)

Usage:
    python run_dit4dit_inference_policy_loop.py \\
        --inference_host localhost \\
        --inference_port 5556 \\
        --camera_host 192.168.123.164 \\
        --camera_port 5555 \\
        --hand_type aloha \\
        --language_instruction "your language prompt"
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

import cv2
import numpy as np
import tyro

# Add md_gr00t to path for imports
gr00t_path = str(Path(__file__).parent.parent.parent.parent.parent / "md_gr00t")
sys.path.insert(0, gr00t_path)

from gr00t_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    STATE_TOPIC_NAME,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
)
from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
from gr00t_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from gr00t_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
from gr00t_wbc.control.utils.signal_handler import SignalHandler, is_shutdown_requested
from gr00t_wbc.control.utils.telemetry import Telemetry

INFERENCE_NODE_NAME = "Dit4ditInferencePolicy"


def _quat_wxyz_to_rpy(quat_wxyz):
    """Convert quaternion [w, x, y, z] to [roll, pitch, yaw]."""
    w, x, y, z = quat_wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


@dataclass
class Dit4ditInferenceConfig:
    """Configuration for DiT4DiT inference policy loop."""

    # Inference server settings
    inference_host: str = "localhost"
    """Host address of the DiT4DiT inference server."""

    inference_port: int = 5556
    """Port of the DiT4DiT inference server (ZMQ)."""

    # Task settings
    language_instruction: str = "pick the object"
    """Language instruction describing the task for the model."""

    # Robot settings
    robot: str = "g1"
    """Robot name."""

    hand_type: Literal["aloha", "three_finger"] = "aloha"
    """Hand type: aloha (1-DOF gripper) or three_finger (7-DOF hand)."""

    high_elbow_pose: bool = False
    """Enable high elbow pose configuration for default joint positions (must match data collection)."""

    # Camera settings
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    camera_key: str = "ego_view"
    """Camera key to use for inference (e.g., ego_view, head_view)."""

    # DiT4DiT specific settings
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    """Image size expected by DiT4DiT model [width, height]."""

    use_ddim: bool = True
    """Use deterministic DDIM sampling for diffusion."""

    num_ddim_steps: int = 4
    """Number of DDIM diffusion steps (DiT4DiT default: 4)."""

    # Policy settings
    inference_frequency: float = 10.0
    """Frequency of inference queries (Hz)."""

    action_horizon: int = 16
    """Number of action steps predicted by model (DiT4DiT chunk size, default: 16)."""

    action_execution_horizon: int = 8
    """Number of predicted actions to execute before re-querying."""

    # Telemetry settings
    telemetry_window_size: int = 100
    """Window size for telemetry averaging."""

    # WBC (whole-body control) settings
    enable_wbc: bool = False
    """Enable whole-body control mode. Sends rpy+height as state, extracts locomotion commands from action."""


class Dit4ditInferencePolicy:
    """Policy that queries DiT4DiT inference server for robot actions.

    This policy manages the inference lifecycle for deploying trained DiT4DiT
    models to real robots. It handles connection to remote inference server,
    action caching, denormalization, and smooth mode transitions.

    The DiT4DiT server returns unnormalized actions:
      Arms-only [T, 16]: [left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)]
      WBC mode  [T, 23]: above 16 + [rpy(3), height(1), vx(1), vy(1), vyaw(1)]

    Keyboard Controls:
        'i': Move to initial pose (open grippers)
        'l': Start inference (smooth transition from current pose)
        'o': Stop inference
        'r': Reset (clear cache and move to initial pose)

    Gripper Convention:
        Training data (from IK solver): 0.065 = OPEN, 0.0 = CLOSED
        Hardware trigger: 0.0 = released/OPEN, 1.0 = pressed/CLOSED
        Conversion: trigger = 1.0 - (gripper_val / CLOSED_GRIPPER_VALUE)
    """

    # Gripper configuration constants
    # Indices in upper_body array: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    LEFT_GRIPPER_INDEX = 7
    RIGHT_GRIPPER_INDEX = 15
    GRIPPER_INDICES = [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]

    # Gripper values in training data convention (from IK solver, matches Arduino hardware)
    OPEN_GRIPPER_VALUE = 0.0  # Gripper fingers together (confusingly named in training data)
    CLOSED_GRIPPER_VALUE = 0.065  # Gripper fingers spread apart

    # WBC safety clipping limits (defense-in-depth, mirrors G1GearWbcPolicy constraints)
    WBC_VX_RANGE = (-0.1, 0.3)  # Forward velocity (m/s)
    WBC_VY_RANGE = (-0.1, 0.1)  # Lateral velocity (m/s)
    WBC_VYAW_RANGE = (-0.8, 0.8)  # Yaw velocity (rad/s)
    WBC_HEIGHT_RANGE = (0.2, 0.74)  # Base height (m)
    WBC_RPY_RANGE = (-0.52, 0.52)  # Roll/pitch/yaw (~30 degrees in radians)

    def __init__(
        self,
        robot_model,
        hand_type: str = "aloha",
        host: str = "localhost",
        port: int = 5556,
        language_instruction: str = "pick the object",
        action_horizon: int = 16,
        action_execution_horizon: int = 8,
        camera_key: str = "ego_view",
        image_size: List[int] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 4,
        enable_wbc: bool = False,
    ):
        self.robot_model = robot_model
        self.hand_type = hand_type
        self.host = host
        self.port = port
        self.language_instruction = language_instruction
        self.action_horizon = action_horizon
        self.action_execution_horizon = action_execution_horizon
        self.camera_key = camera_key
        self.image_size = image_size or [224, 224]
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.enable_wbc = enable_wbc

        # State
        self._is_active = False
        self._cached_actions = None  # Stores raw (denormalized) actions [T, action_dim]
        self._action_index = 0
        self._latest_obs = None
        self._latest_image = None
        self._commanding_initial_pose = False  # Track if we're moving to initial pose
        self._need_toggle_policy_action = False  # One-shot flag to activate lower body RL policy

        # Initial pose (with open grippers)
        self._initial_upper_body_pose = robot_model.get_initial_upper_body_pose()
        # Ensure grippers are open in initial pose
        for idx in self.GRIPPER_INDICES:
            self._initial_upper_body_pose[idx] = self.OPEN_GRIPPER_VALUE

        # Determine hand/gripper joint group names based on hand_type
        if hand_type == "aloha":
            self._left_hand_group = "left_gripper"
            self._right_hand_group = "right_gripper"
        else:  # three_finger
            self._left_hand_group = "left_hand"
            self._right_hand_group = "right_hand"

        # Get joint indices
        self._upper_body_indices = robot_model.get_joint_group_indices("upper_body")
        self._left_arm_indices = robot_model.get_joint_group_indices("left_arm")
        self._right_arm_indices = robot_model.get_joint_group_indices("right_arm")
        self._left_hand_indices = robot_model.get_joint_group_indices(self._left_hand_group)
        self._right_hand_indices = robot_model.get_joint_group_indices(self._right_hand_group)

        # Initialize ZMQ + msgpack inference client
        self._init_client()

        print(
            f"[Dit4ditInferencePolicy] Initialized. Press 'i' for initial pose, 'l' to start inference."
        )

    def _init_client(self):
        """Initialize DiT4DiT inference client (ZMQ)."""
        from gr00t_wbc.control.utils.service import InferenceClient

        self.client = InferenceClient(host=self.host, port=self.port)
        print(f"[Dit4ditInferencePolicy] ZMQ client: {self.host}:{self.port}")

    def set_observation(self, obs: dict, image: np.ndarray):
        """Set current observation and image."""
        self._latest_obs = obs
        self._latest_image = image

    def activate(self):
        """Activate inference - smooth transition from current pose."""
        self._is_active = True
        self._cached_actions = None
        self._action_index = 0
        print("[Dit4ditInferencePolicy] Activated - starting inference")

    def deactivate(self):
        """Deactivate inference."""
        self._is_active = False
        print("[Dit4ditInferencePolicy] Deactivated")

    def command_initial_pose(self):
        """Move to initial pose with open grippers (non-blocking)."""
        self._commanding_initial_pose = True
        self._is_active = False  # Deactivate inference when going to initial pose
        if self.enable_wbc:
            self._need_toggle_policy_action = True
        print("[Dit4ditInferencePolicy] Moving to initial pose (grippers open)...")

    def is_commanding_initial_pose(self) -> bool:
        """Check if currently commanding initial pose."""
        return self._commanding_initial_pose

    def stop_initial_pose_command(self):
        """Stop commanding initial pose (called when activating inference)."""
        self._commanding_initial_pose = False

    def get_initial_pose_target(self) -> np.ndarray:
        """Get the initial pose target (with open grippers)."""
        return self._initial_upper_body_pose.copy()

    def _get_hold_position_action(self) -> dict:
        """Return action that maintains current robot position."""
        action = {"inference_active": self._is_active}
        if self._latest_obs and "q" in self._latest_obs:
            action["target_upper_body_pose"] = np.array(self._latest_obs["q"])[
                self._upper_body_indices
            ]
        # In WBC mode, include safe default locomotion commands
        if self.enable_wbc:
            action["base_height_command"] = DEFAULT_BASE_HEIGHT
            action["navigate_cmd"] = list(DEFAULT_NAV_CMD)
            action["torso_orientation_rpy"] = [0.0, 0.0, 0.0]
        return action

    @property
    def is_active(self) -> bool:
        """Whether inference is currently active."""
        return self._is_active

    def _prepare_observation(self) -> dict:
        """Convert robot observation to DiT4DiT input format.

        Prepares the observation dict in the format expected by DiT4DiT server:
        - image: resized to model expected size, uint8
        - lang: task description
        - state: joint positions [1, state_dim]
        """
        if self._latest_obs is None or self._latest_image is None:
            raise ValueError("No observation or image set")

        q = self._latest_obs.get("q")
        if q is None:
            raise ValueError("Observation missing 'q'")
        # Extract joint positions
        left_arm_q = q[self._left_arm_indices].reshape(1, -1).astype(np.float64)
        right_arm_q = q[self._right_arm_indices].reshape(1, -1).astype(np.float64)
        left_hand_q = q[self._left_hand_indices].reshape(1, -1).astype(np.float64)
        right_hand_q = q[self._right_hand_indices].reshape(1, -1).astype(np.float64)
        state_parts = [left_arm_q, right_arm_q, left_hand_q, right_hand_q]

        # Add WBC state inputs (rpy + height) when enabled
        if self.enable_wbc:
            torso_quat = self._latest_obs.get("torso_quat")
            if torso_quat is not None:
                rpy = _quat_wxyz_to_rpy(torso_quat)
            else:
                rpy = np.array([0.0, 0.0, 0.0])
                print("[Dit4ditInferencePolicy] WARNING: torso_quat not in state, using zero RPY")

            height = self._latest_obs.get("base_height_command", DEFAULT_BASE_HEIGHT)
            if isinstance(height, (list, np.ndarray)):
                height = float(np.asarray(height).flat[0])

            state_parts.append(rpy.reshape(1, -1).astype(np.float64))
            state_parts.append(np.array([[height]]).astype(np.float64))

        state = np.concatenate(state_parts, axis=1)

        # Prepare image - resize to expected size
        image = self._latest_image
        if image.ndim == 4:
            image = image[0]  # Remove batch dimension if present
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)

        # Build query in DiT4DiT format
        return {
            "examples": [
                {
                    "image": [image],  # List of images [H, W, 3] uint8
                    "lang": self.language_instruction,
                    "state": state,
                }
            ],
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

    def _query_server(self, obs: dict) -> Optional[dict]:
        """Query DiT4DiT inference server."""
        try:
            response = self.client.predict_action(obs)
            if response.get("status") == "ok" and "data" in response:
                return response["data"]
            print(f"[Dit4ditInferencePolicy] Server response error: {response.get('status')}")
            return None
        except Exception as e:
            print(f"[Dit4ditInferencePolicy] Query error: {e}")
            return None

    def _extract_upper_body_action(self, raw_actions: np.ndarray, step: int) -> np.ndarray:
        """Extract upper body pose from raw actions at the given step.

        DiT4DiT server output (already sliced to 16 dims):
            [0:7]   left_arm    (7 DOF)
            [7:14]  right_arm   (7 DOF)
            [14:15] left_gripper (1)
            [15:16] right_gripper (1)

        Robot upper_body group (sorted by URDF DOF index):
            left_arm (7) + left_gripper (1) + right_arm (7) + right_gripper (1) = 16 DOF

        Args:
            raw_actions: Denormalized actions [T, 16]
            step: Step index to extract

        Returns:
            np.ndarray of 16 joint positions for upper body
        """
        action = raw_actions[step]

        left_arm = action[0:7]
        right_arm = action[7:14]
        left_gripper = action[14:15]
        right_gripper = action[15:16]

        # Concatenate in robot_model upper_body order:
        return np.concatenate([left_arm, left_gripper, right_arm, right_gripper])

    def _debug_print_gripper(self, left_val, right_val, left_trigger=None, right_trigger=None):
        """Print gripper debug info at 1Hz."""
        if not hasattr(self, "_last_debug_time"):
            self._last_debug_time = 0
        now = time.time()
        if now - self._last_debug_time >= 1.0:
            trigger_info = ""
            if left_trigger is not None and right_trigger is not None:
                trigger_info = f" -> trigger L={left_trigger:.2f}, R={right_trigger:.2f}"
            print(
                f"[Model Output] left_gripper={left_val:.4f}, right_gripper={right_val:.4f}{trigger_info}"
            )
            self._last_debug_time = now

    def _debug_print_arm_joints(self, upper_body_pose: np.ndarray):
        """Print arm joint values at 1Hz for debugging."""
        if not hasattr(self, "_last_arm_debug_time"):
            self._last_arm_debug_time = 0
        now = time.time()
        if now - self._last_arm_debug_time >= 1.0:
            # Also print current robot state for comparison
            if self._latest_obs is not None and "q" in self._latest_obs:
                q = self._latest_obs["q"]
                current_left_arm = q[self._left_arm_indices]
                current_right_arm = q[self._right_arm_indices]
                print("[Current State] Robot joints:")
                print(f"  Left arm:  {' '.join([f'{v:+.3f}' for v in current_left_arm])}")
                print(f"  Right arm: {' '.join([f'{v:+.3f}' for v in current_right_arm])}")

            print("[Model Output] Upper body joints:")
            # Print left arm (indices 0-6)
            print(f"  Left arm:  {' '.join([f'{v:+.3f}' for v in upper_body_pose[0:7]])}")
            # Print right arm (indices 8-14)
            print(f"  Right arm: {' '.join([f'{v:+.3f}' for v in upper_body_pose[8:15]])}")
            self._last_arm_debug_time = now

    def _extract_wbc_action(self, raw_actions: np.ndarray, step: int) -> dict:
        """Extract whole-body control locomotion commands from flat action array.

        WBC action layout (dims 16-22 of unnormalized_actions):
            [16:19] rpy (3): torso roll, pitch, yaw
            [19:20] height (1): base height
            [20:21] torso_vx (1)
            [21:22] torso_vy (1)
            [22:23] torso_vyaw (1)

        Returns dict matching the control loop's expected format:
        - base_height_command (float)
        - navigate_cmd ([vx, vy, vyaw])
        - torso_orientation_rpy ([roll, pitch, yaw])
        """
        action = raw_actions[step]

        rpy = action[16:19]
        height_val = float(np.clip(action[19], *self.WBC_HEIGHT_RANGE))
        vx_val = float(np.clip(action[20], *self.WBC_VX_RANGE))
        vy_val = float(np.clip(action[21], *self.WBC_VY_RANGE))
        vyaw_val = float(np.clip(action[22], *self.WBC_VYAW_RANGE))
        rpy_clipped = np.clip(rpy, *self.WBC_RPY_RANGE)

        return {
            "base_height_command": height_val,
            "navigate_cmd": [vx_val, vy_val, vyaw_val],
            "torso_orientation_rpy": [
                float(rpy_clipped[0]),
                float(rpy_clipped[1]),
                0,
            ],
        }

    def _debug_print_wbc(self, wbc_data: dict):
        """Print WBC locomotion command values at 1Hz for debugging."""
        if not hasattr(self, "_last_wbc_debug_time"):
            self._last_wbc_debug_time = 0
        now = time.time()
        if now - self._last_wbc_debug_time >= 1.0:
            nav = wbc_data.get("navigate_cmd", [0, 0, 0])
            rpy = wbc_data.get("torso_orientation_rpy", [0, 0, 0])
            height = wbc_data.get("base_height_command", DEFAULT_BASE_HEIGHT)
            print(
                f"[Model Output WBC] height={height:.4f}, "
                f"nav=[{nav[0]:+.4f}, {nav[1]:+.4f}, {nav[2]:+.4f}], "
                f"rpy=[{rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}]"
            )
            self._last_wbc_debug_time = now

    def get_action(self) -> dict:
        """Get action from inference server, initial pose command, or hold current pose.

        The action selection follows this priority:
        1. Initial pose command (if _commanding_initial_pose is True)
        2. Hold current position (if inference is inactive)
        3. Hold current position (if waiting for observation data)
        4. Query inference server and return predicted action

        Returns:
            Action dict containing:
            - inference_active: bool indicating if inference is running
            - target_upper_body_pose: np.ndarray of 16 joint positions
            - commanding_initial_pose: bool (only when moving to initial pose)
        """
        # Priority 1: Initial pose command
        if self._commanding_initial_pose:
            action = {
                "inference_active": self._is_active,
                "target_upper_body_pose": self._initial_upper_body_pose.copy(),
                "commanding_initial_pose": True,
            }
            # One-shot: activate lower body RL policy on first initial pose frame
            if self._need_toggle_policy_action:
                action["toggle_policy_action"] = True
                self._need_toggle_policy_action = False
            return action

        # Priority 2: Inference inactive - hold position
        if not self._is_active:
            return self._get_hold_position_action()

        # Priority 3: Missing observation data - hold position
        if self._latest_obs is None or self._latest_image is None:
            return self._get_hold_position_action()

        # Priority 4: Active inference - query server if needed
        return self._get_inference_action()

    def _get_inference_action(self) -> dict:
        """Query inference server and return predicted action.

        Manages action caching: queries server when cache is empty or
        execution horizon is reached, otherwise returns next cached action.

        Returns:
            Action dict with predicted upper body pose, or hold position on error.
        """
        action = {"inference_active": self._is_active}

        # Check if we need to query the server
        need_query = (
            self._cached_actions is None or self._action_index >= self.action_execution_horizon
        )
        if self._cached_actions is None:
            print("[Dit4ditInferencePolicy] No cached actions - querying server")
        if self._action_index >= self.action_execution_horizon:
            print("[Dit4ditInferencePolicy] Execution horizon reached - querying server")

        if need_query:
            if not self._query_and_cache_actions():
                return self._get_hold_position_action()

        # Extract action from cached actions
        try:
            upper_body_pose = self._extract_upper_body_action(
                self._cached_actions, self._action_index
            )
            action["target_upper_body_pose"] = upper_body_pose

            # Extract WBC locomotion commands if enabled
            if self.enable_wbc:
                wbc_data = self._extract_wbc_action(self._cached_actions, self._action_index)
                action.update(wbc_data)
                self._debug_print_wbc(wbc_data)

            self._action_index += 1
            self._debug_print_arm_joints(upper_body_pose)
        except Exception as e:
            print(f"[Dit4ditInferencePolicy] Extract error: {e}")
            return self._get_hold_position_action()

        return action

    def _query_and_cache_actions(self) -> bool:
        """Query inference server and cache the predicted actions.

        Returns:
            True if query succeeded and actions were cached, False otherwise.
        """
        try:
            obs = self._prepare_observation()
            t_start = time.time()
            prediction = self._query_server(obs)
            query_time = time.time() - t_start

            if prediction and "unnormalized_actions" in prediction:
                self._cached_actions = prediction["unnormalized_actions"]  # [T, 23]
                self._action_index = 0
                print(
                    f"[Dit4ditInferencePolicy] Query: {query_time:.3f}s, "
                    f"actions shape: {self._cached_actions.shape}"
                )
                return True
            else:
                print(
                    "[Dit4ditInferencePolicy] Query returned None or missing unnormalized_actions"
                )
                return False
        except Exception as e:
            print(f"[Dit4ditInferencePolicy] Query error: {e}")
            return False

    def reset(self):
        """Reset policy state and move to initial pose."""
        self._cached_actions = None
        self._action_index = 0
        self._is_active = False
        self._commanding_initial_pose = True  # Move to initial pose on reset
        print("[Dit4ditInferencePolicy] Reset - moving to initial pose")


def main(config: Dit4ditInferenceConfig):
    """Main inference policy loop."""
    signal_handler = SignalHandler()

    ros_manager = None

    try:
        # Use ROSManager like the teleop policy does
        ros_manager = ROSManager(node_name=INFERENCE_NODE_NAME)
        node = ros_manager.node
    except KeyboardInterrupt:
        print("[Dit4ditInferencePolicy] Interrupted during ROS initialization")
        return

    # Get robot model
    robot_name = "g1_aloha" if config.hand_type == "aloha" else "g1"
    _, robot_model = get_robot_type_and_model(
        robot=robot_name,
        high_elbow_pose=config.high_elbow_pose,
    )

    # Create policy
    policy = Dit4ditInferencePolicy(
        robot_model=robot_model,
        hand_type=config.hand_type,
        host=config.inference_host,
        port=config.inference_port,
        language_instruction=config.language_instruction,
        action_horizon=config.action_horizon,
        action_execution_horizon=config.action_execution_horizon,
        camera_key=config.camera_key,
        image_size=config.image_size,
        use_ddim=config.use_ddim,
        num_ddim_steps=config.num_ddim_steps,
        enable_wbc=config.enable_wbc,
    )

    # Create camera client
    camera_client = ComposedCameraClientSensor(
        server_ip=config.camera_host,
        port=config.camera_port,
    )

    # Use ROSMsgPublisher/ROSMsgSubscriber like teleop policy does
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)

    # Keyboard listener for activation control
    keyboard_listener = KeyboardListenerSubscriber()

    rate = node.create_rate(config.inference_frequency)
    telemetry = Telemetry(window_size=config.telemetry_window_size)

    # Persistent storage for latest messages
    latest_state_msg = None
    latest_image = None

    print(f"[Dit4ditInferencePolicy] Starting loop at {config.inference_frequency} Hz")
    print(f"[Dit4ditInferencePolicy] Task: {config.language_instruction}")
    if config.enable_wbc:
        print(
            "[Dit4ditInferencePolicy] WBC MODE ENABLED - will send locomotion commands (height, rpy, vx, vy, vyaw)"
        )
    else:
        print("[Dit4ditInferencePolicy] Arms-only mode (no locomotion commands)")
    print("[Dit4ditInferencePolicy] Controls:")
    print("  'i' - Move to initial pose (open grippers)")
    print("  'l' - Start inference (smooth transition from current pose)")
    print("  'o' - Stop inference")
    print("  'r' - Reset (clear cache and move to initial pose)")

    try:
        while ros_manager.ok() and not is_shutdown_requested():
            with telemetry.timer("total_loop"):
                t_now = time.monotonic()

                # Handle keyboard input for activation/deactivation
                key = keyboard_listener.read_msg()
                if key == "i":
                    # Move to initial pose (non-blocking)
                    policy.command_initial_pose()
                elif key == "l":
                    # Activate inference immediately
                    policy.stop_initial_pose_command()
                    if not policy.is_active:
                        policy.activate()
                    else:
                        print("[Dit4ditInferencePolicy] Already active")
                elif key == "o":
                    if policy.is_active:
                        policy.deactivate()
                    else:
                        print("[Dit4ditInferencePolicy] Already inactive")
                elif key == "r":
                    policy.reset()

                # Poll robot state - update independently
                state_msg = state_subscriber.get_msg()
                if state_msg is not None:
                    latest_state_msg = state_msg
                    policy._latest_obs = latest_state_msg

                # Poll camera image - update independently
                try:
                    camera_data = camera_client.read()
                    if camera_data is not None and "images" in camera_data:
                        img = camera_data["images"].get(config.camera_key)
                        if img is not None:
                            latest_image = img
                            policy._latest_image = latest_image
                except Exception as e:
                    print(f"[Dit4ditInferencePolicy] Camera error: {e}")

                # Debug: Log observation status when policy is active but can't query
                if policy.is_active and (latest_state_msg is None or latest_image is None):
                    has_state = "Y" if latest_state_msg is not None else "N"
                    has_image = "Y" if latest_image is not None else "N"
                    print(
                        f"[Dit4ditInferencePolicy] Waiting for data: state={has_state}, image={has_image}"
                    )

                # Get action
                with telemetry.timer("get_action"):
                    action_data = policy.get_action()

                # Publish
                if action_data.get("target_upper_body_pose") is not None:
                    upper_body_pose = action_data["target_upper_body_pose"]
                    msg = {
                        "target_upper_body_pose": upper_body_pose,
                        "timestamp": t_now,
                    }
                    # Add wrist_pose for data exporter compatibility
                    if latest_state_msg is not None and "wrist_pose" in latest_state_msg:
                        msg["wrist_pose"] = np.array(
                            latest_state_msg["wrist_pose"], dtype=np.float64
                        )
                    else:
                        # Default wrist pose (14 values: left pos+quat + right pos+quat)
                        msg["wrist_pose"] = np.array(
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2, dtype=np.float64
                        )
                    # Forward toggle_policy_action to activate lower body RL policy
                    if "toggle_policy_action" in action_data:
                        msg["toggle_policy_action"] = action_data["toggle_policy_action"]
                    # Add WBC locomotion commands when present
                    if "base_height_command" in action_data:
                        msg["base_height_command"] = action_data["base_height_command"]
                    if "navigate_cmd" in action_data:
                        msg["navigate_cmd"] = action_data["navigate_cmd"]
                    if "torso_orientation_rpy" in action_data:
                        msg["torso_orientation_rpy"] = action_data["torso_orientation_rpy"]

                    # Use rolling target_time
                    if action_data.get("commanding_initial_pose", False):
                        msg["target_time"] = t_now + 0.5  # Rolling time, always 0.5s ahead
                        # Send trigger=0.0 to open grippers
                        msg["left_fingers"] = {"trigger": 0.0}
                        msg["right_fingers"] = {"trigger": 0.0}
                        # Safe WBC defaults during initial pose
                        if config.enable_wbc:
                            msg["base_height_command"] = DEFAULT_BASE_HEIGHT
                            msg["navigate_cmd"] = list(DEFAULT_NAV_CMD)
                            msg["torso_orientation_rpy"] = [0.0, 0.0, 0.0]
                    else:
                        msg["target_time"] = t_now + (1 / config.inference_frequency)
                        # Convert gripper values from model to trigger
                        # Training data convention (from IK solver, matches Arduino hardware):
                        #   0.065 = OPEN (gripper fingers spread apart)
                        #   0.0 = CLOSED (gripper fingers together)
                        # Trigger convention:
                        #   0.0 = released = gripper OPEN
                        #   1.0 = pressed = gripper CLOSED
                        left_gripper_val = upper_body_pose[policy.GRIPPER_INDICES[0]]
                        right_gripper_val = upper_body_pose[policy.GRIPPER_INDICES[1]]
                        left_trigger = 1.0 - np.clip(
                            left_gripper_val / policy.CLOSED_GRIPPER_VALUE, 0.0, 1.0
                        )
                        right_trigger = 1.0 - np.clip(
                            right_gripper_val / policy.CLOSED_GRIPPER_VALUE, 0.0, 1.0
                        )
                        msg["left_fingers"] = {"trigger": float(left_trigger)}
                        msg["right_fingers"] = {"trigger": float(right_trigger)}
                        # Debug logging at 1Hz
                        policy._debug_print_gripper(
                            left_gripper_val, right_gripper_val, left_trigger, right_trigger
                        )
                    control_publisher.publish(msg)

            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"\n[Dit4ditInferencePolicy] ROSManager interrupted by user: {e}")
    except KeyboardInterrupt:
        print("\n[Dit4ditInferencePolicy] Interrupted")
    except Exception as e:
        print(f"[Dit4ditInferencePolicy] Error in inference loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("[Dit4ditInferencePolicy] Cleaning up...")
        # Close DiT4DiT client
        try:
            if hasattr(policy, "client"):
                policy.client.close()
        except Exception as e:
            print(f"[Dit4ditInferencePolicy] Error closing client: {e}")
        try:
            if ros_manager is not None:
                print("[Dit4ditInferencePolicy] Shutting down ROS...")
                ros_manager.shutdown()
        except Exception as e:
            print(f"[Dit4ditInferencePolicy] Error shutting down ROS: {e}")
        signal_handler.cleanup()
        print("[Dit4ditInferencePolicy] Done")

        # Give DDS threads time to finish before exit
        time.sleep(0.5)

        # Force exit to ensure all threads terminate
        import os

        os._exit(0)


if __name__ == "__main__":
    config = tyro.cli(Dit4ditInferenceConfig)
    main(config)
