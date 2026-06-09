import collections
from pathlib import Path
import time as time_module
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
import torch

from gr00t_wbc.control.base.policy import Policy
from gr00t_wbc.control.utils.gear_wbc_utils import get_gravity_orientation, load_config


class G1GearWbcPolicy(Policy):
    """Simple G1 robot policy using OpenGearWbc trained neural network."""

    def __init__(self, robot_model, config: str, model_path: str):
        """Initialize G1GearWbcPolicy.

        Args:
            config_path: Path to gear_wbc YAML configuration file
        """
        self.config, self.LEGGED_GYM_ROOT_DIR = load_config(config)
        self.robot_model = robot_model
        self.use_teleop_policy_cmd = False

        package_root = Path(__file__).resolve().parents[2]
        self.sim2mujoco_root_dir = str(package_root / "sim2mujoco")
        model_path_1, model_path_2 = model_path.split(",")

        self.policy_1 = self.load_onnx_policy(
            self.sim2mujoco_root_dir + "/resources/robots/g1/" + model_path_1
        )
        self.policy_2 = self.load_onnx_policy(
            self.sim2mujoco_root_dir + "/resources/robots/g1/" + model_path_2
        )

        # Initialize observation history buffer
        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(self.config["num_obs"], dtype=np.float32)
        self.counter = 0

        # Initialize state variables
        self.use_policy_action = False
        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = self.config["height_cmd"]
        self.freq_cmd = self.config["freq_cmd"]
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]
        self.gait_indices = torch.zeros((1), dtype=torch.float32)
        
        # Command decay parameters for smooth stopping
        self.cmd_decay_rate = 0.98  # Exponential decay factor (0.95 = 5% decay per step)
        self.cmd_decay_threshold = 0.01  # Below this magnitude, set to zero
        self.last_teleop_update_time = None
        self.teleop_timeout = 0.1  # Start decay after 20ms without teleop update

    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions in [w, x, y, z] format. Returns q1 * q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def load_onnx_policy(self, model_path: str):
        print(f"Loading ONNX policy from {model_path}")
        model = ort.InferenceSession(model_path)

        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cpu")

        print(f"Successfully loaded ONNX policy from {model_path}")

        return run_inference

    def compute_observation(self, observation: Dict[str, Any]) -> tuple[np.ndarray, int]:
        """Compute the observation vector from current state"""
        # Get body joint indices (excluding waist roll and pitch)
        self.gait_indices = torch.remainder(self.gait_indices + 0.02 * self.freq_cmd, 1.0)
        durations = torch.full_like(self.gait_indices, 0.5)
        phases = 0.5
        foot_indices = [
            self.gait_indices + phases,  # FL
            self.gait_indices,  # FR
        ]
        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0
        )
        for fi in foot_indices:
            stance = fi < durations
            swing = fi >= durations
            fi[stance] = fi[stance] * (0.5 / durations[stance])
            fi[swing] = 0.5 + (fi[swing] - durations[swing]) * (0.5 / (1 - durations[swing]))

        self.clock_inputs = torch.stack([torch.sin(2 * np.pi * fi) for fi in foot_indices], dim=1)

        body_indices = self.robot_model.get_joint_group_indices("body")
        body_indices = [idx for idx in body_indices]

        n_joints = len(body_indices)

        # Extract joint data
        qj = observation["q"][body_indices].copy()
        dqj = observation["dq"][body_indices].copy()

        # Extract floating base data
        quat = observation["floating_base_pose"][3:7].copy()  # quaternion
        omega = observation["floating_base_vel"][3:6].copy()  # angular velocity

        # Apply IMU bias correction only for forward/backward movement
        # This compensates for robot leaning when walking forward/backward
        if abs(self.cmd[0]) > 0.01:  # Forward/backward command active
            imu_bias_quat = np.array(self.config.get("IMU_BIAS_QUAT", [1.0, 0.0, 0.0, 0.0]))
            quat = self._quaternion_multiply(imu_bias_quat, quat)

        # Handle default angles padding
        if len(self.config["default_angles"]) < n_joints:
            padded_defaults = np.zeros(n_joints, dtype=np.float32)
            padded_defaults[: len(self.config["default_angles"])] = self.config["default_angles"]
        else:
            padded_defaults = self.config["default_angles"][:n_joints]

        # Scale the values
        qj_scaled = (qj - padded_defaults) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.config["ang_vel_scale"]

        # Calculate single observation dimension
        single_obs_dim = 86  # 3 + 1 + 3 + 3 + 3 + n_joints + n_joints + 15, n_joints = 29

        ####COMMAND SAFETY CHECKS####
        # Safety check: enforce zero velocity if not at standing height
        current_height = float(self.height_cmd[0] if isinstance(self.height_cmd, np.ndarray) else self.height_cmd)
        at_standing_height = abs(current_height - 0.74) < 0.01  # Within 1cm of standing height
        
        # Apply safety: zero velocities if not at standing height
        safe_cmd = self.cmd.copy()
        if not at_standing_height:
            safe_cmd[0] = 0.0
            safe_cmd[1] = 0.0
            safe_cmd[2] = 0.0
        else:
            # Enforce velocity limits (only when at standing height)
            safe_cmd[0] = np.clip(safe_cmd[0], -0.1, 0.1)   # Forward: -0.1 to +0.1 m/s
            safe_cmd[1] = np.clip(safe_cmd[1], -0.1, 0.1)   # Strafe: -0.1 to +0.1 m/s
            safe_cmd[2] = np.clip(safe_cmd[2], -0.5, 0.5)   # Yaw: -0.8 to +0.8 rad/s
            
            # Single-axis constraint: only one velocity component can be non-zero
            # Priority: forward/backward > strafe > rotation
            abs_vals = [abs(safe_cmd[0]), abs(safe_cmd[1]), abs(safe_cmd[2])]
            max_idx = abs_vals.index(max(abs_vals))
            for i in range(3):
                if i != max_idx:
                    safe_cmd[i] = 0.0

        # Create single observation
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:3] = safe_cmd[:3] * self.config["cmd_scale"]
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = np.array([self.roll_cmd, self.pitch_cmd, self.yaw_cmd])
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_orientation
        # single_obs[14:17] = omega_scaled_torso
        # single_obs[17:20] = gravity_torso
        single_obs[13 : 13 + n_joints] = qj_scaled
        single_obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[13 + 2 * n_joints : 13 + 2 * n_joints + 15] = self.action
        # single_obs[13 + 2 * n_joints + 15 : 13 + 2 * n_joints + 15 + 2] = (
        #     processed_clock_inputs.detach().cpu().numpy()
        # )
        return single_obs, single_obs_dim

    def set_observation(self, observation: Dict[str, Any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing single observation from current state
                        Should include 'obs' key with current single observation
        """

        # Extract the single observation
        self.observation = observation
        single_obs, single_obs_dim = self.compute_observation(observation)

        # Update observation history every control_decimation steps
        # if self.counter % self.config['control_decimation'] == 0:
        # Add current observation to history
        self.obs_history.append(single_obs)

        # Fill history with zeros if not enough observations yet
        while len(self.obs_history) < self.config["obs_history_len"]:
            self.obs_history.appendleft(np.zeros_like(single_obs))

        # Construct full observation with history
        single_obs_dim = len(single_obs)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * single_obs_dim
            end_idx = start_idx + single_obs_dim
            self.obs_buffer[start_idx:end_idx] = hist_obs

        # Convert to tensor for policy
        self.obs_tensor = torch.from_numpy(self.obs_buffer).unsqueeze(0)
        # self.counter += 1

        assert self.obs_tensor.shape[1] == self.config["num_obs"]

    def set_use_teleop_policy_cmd(self, use_teleop_policy_cmd: bool):
        self.use_teleop_policy_cmd = use_teleop_policy_cmd
        # Safety: When teleop is disabled, reset navigation to stop
        if not use_teleop_policy_cmd:
            self.nav_cmd = self.config["cmd_init"].copy()  # Reset to safe default

    def set_goal(self, goal: Dict[str, Any]):
        """Set the goal for the policy.

        Args:
            goal: Dictionary containing the goal for the policy
        """

        if "toggle_policy_action" in goal:
            if goal["toggle_policy_action"]:
                self.use_policy_action = not self.use_policy_action

    def get_action(
        self,
        time: Optional[float] = None,
        arms_target_pose: Optional[np.ndarray] = None,
        base_height_command: Optional[np.ndarray] = None,
        torso_orientation_rpy: Optional[np.ndarray] = None,
        interpolated_navigate_cmd: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute and return the next action based on current observation.

        Args:
            time: Optional "monotonic time" for time-dependent policies (unused)

        Returns:
            Dictionary containing the action to be executed
        """
        if self.obs_tensor is None:
            raise ValueError("No observation set. Call set_observation() first.")

        # Track if any teleop command was updated
        teleop_updated = False

        if base_height_command is not None and self.use_teleop_policy_cmd:
            self.height_cmd = (
                base_height_command[0]
                if isinstance(base_height_command, list)
                else base_height_command
            )
            teleop_updated = True

        if interpolated_navigate_cmd is not None and self.use_teleop_policy_cmd:
            self.cmd = interpolated_navigate_cmd
            self.last_teleop_update_time = time_module.monotonic()
            teleop_updated = True

        if torso_orientation_rpy is not None and self.use_teleop_policy_cmd:
            self.roll_cmd = torso_orientation_rpy[0]
            self.pitch_cmd = torso_orientation_rpy[1]
            self.yaw_cmd = torso_orientation_rpy[2]
            teleop_updated = True

        # Apply command decay when no teleop updates received (smooth stopping)
        current_time = time_module.monotonic()
        if self.last_teleop_update_time is not None:
            time_since_update = current_time - self.last_teleop_update_time
            if time_since_update > self.teleop_timeout:
                # Apply exponential decay to velocity commands
                self.cmd = self.cmd * self.cmd_decay_rate
                # Set to zero if below threshold to prevent drift
                cmd_magnitude = np.linalg.norm(self.cmd)
                if cmd_magnitude < self.cmd_decay_threshold:
                    self.cmd = np.zeros_like(self.cmd)

        # Print teleop commands when updated (Pico, Joycon, etc.)
        if teleop_updated:
            # Extract scalar values for formatting
            height_val = float(self.height_cmd[0] if isinstance(self.height_cmd, np.ndarray) else self.height_cmd)
            roll_val = float(self.roll_cmd[0] if isinstance(self.roll_cmd, np.ndarray) else self.roll_cmd)
            pitch_val = float(self.pitch_cmd[0] if isinstance(self.pitch_cmd, np.ndarray) else self.pitch_cmd)
            yaw_val = float(self.yaw_cmd[0] if isinstance(self.yaw_cmd, np.ndarray) else self.yaw_cmd)
            freq_val = float(self.freq_cmd[0] if isinstance(self.freq_cmd, np.ndarray) else self.freq_cmd)

            # Commented out to reduce log spam
            # print("-------------teleop cmd-------------------")
            # print(f"Linear velocity command: [{self.cmd[0]:.4f}, {self.cmd[1]:.4f}, {self.cmd[2]:.4f}]")
            # print(f"Base height command: {height_val:.4f}")
            # print(f"Use policy action: {self.use_policy_action}")
            # print(f"roll deg angle: {np.rad2deg(roll_val):.4f}")
            # print(f"pitch deg angle: {np.rad2deg(pitch_val):.4f}")
            # print(f"yaw deg angle: {np.rad2deg(yaw_val):.4f}")
            # print(f"Gait frequency: {freq_val:.4f}")

        # Run policy inference
        with torch.no_grad():
            # Select appropriate policy based on command magnitude
            if np.linalg.norm(self.cmd) < 0.05:
                # Use standing policy for small commands
                policy = self.policy_1
            else:
                # Use walking policy for movement commands
                policy = self.policy_2

            self.action = policy(self.obs_tensor).detach().numpy().squeeze()

        # Transform action to target_dof_pos
        if self.use_policy_action:
            cmd_q = self.action * self.config["action_scale"] + self.config["default_angles"]
        else:
            cmd_q = self.observation["q"][self.robot_model.get_joint_group_indices("lower_body")]

        cmd_dq = np.zeros(self.config["num_actions"])
        cmd_tau = np.zeros(self.config["num_actions"])

        return {"body_action": (cmd_q, cmd_dq, cmd_tau)}

    def handle_keyboard_button(self, key):
        # Get current height for safety check
        current_height = float(self.height_cmd[0] if isinstance(self.height_cmd, np.ndarray) else self.height_cmd)
        at_standing_height = abs(current_height - 0.74) < 0.01  # Within 1cm of standing height
        
        if key == "]":
            self.use_policy_action = True
        elif key == "o":
            self.use_policy_action = False
        elif key == "w":
            if at_standing_height:
                # Zero other axes before setting forward velocity
                self.cmd[1] = 0.0
                self.cmd[2] = 0.0
                self.cmd[0] += 0.1
                self.cmd[0] = min(self.cmd[0], 0.3)  # Limit positive x velocity to 0.3
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to move. Use keys 1/2 to adjust height.")
        elif key == "s":
            if at_standing_height:
                # Zero other axes before setting backward velocity
                self.cmd[1] = 0.0
                self.cmd[2] = 0.0
                self.cmd[0] -= 0.1
                self.cmd[0] = max(self.cmd[0], -0.1)  # Limit negative x velocity to -0.1
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to move. Use keys 1/2 to adjust height.")
        elif key == "a":
            if at_standing_height:
                # Zero other axes before setting strafe velocity
                self.cmd[0] = 0.0
                self.cmd[2] = 0.0
                self.cmd[1] += 0.1
                self.cmd[1] = min(self.cmd[1], 0.1)  # Limit positive y velocity to 0.1
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to move. Use keys 1/2 to adjust height.")
        elif key == "d":
            if at_standing_height:
                # Zero other axes before setting strafe velocity
                self.cmd[0] = 0.0
                self.cmd[2] = 0.0
                self.cmd[1] -= 0.1
                self.cmd[1] = max(self.cmd[1], -0.1)  # Limit negative y velocity to -0.1
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to move. Use keys 1/2 to adjust height.")
        elif key == "q":
            if at_standing_height:
                # Zero other axes before setting rotation velocity
                self.cmd[0] = 0.0
                self.cmd[1] = 0.0
                self.cmd[2] += 0.1
                self.cmd[2] = min(self.cmd[2], 0.4)  # Limit positive angular velocity to 0.4
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to move. Use keys 1/2 to adjust height.")
        elif key == "e":
            if at_standing_height:
                # Zero other axes before setting rotation velocity
                self.cmd[0] = 0.0
                self.cmd[1] = 0.0
                self.cmd[2] -= 0.1
                self.cmd[2] = max(self.cmd[2], -0.4)  # Limit negative angular velocity to -0.4
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to move. Use keys 1/2 to adjust height.")
        elif key == "z":
            # Always allow zeroing velocities for safety
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
        elif key == "1":
            # Only decrease height if above minimum
            if self.height_cmd > 0.2:
                self.height_cmd -= 0.1
                self.height_cmd = max(self.height_cmd, 0.2)
        elif key == "2":
            # Only increase height if below maximum
            if self.height_cmd < 0.74:
                self.height_cmd += 0.1
                self.height_cmd = min(self.height_cmd, 0.74)
        elif key == "n":
            if at_standing_height:
                self.freq_cmd -= 0.1
                self.freq_cmd = max(1.0, self.freq_cmd)
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to change gait. Use keys 1/2 to adjust height.")
        elif key == "m":
            if at_standing_height:
                self.freq_cmd += 0.1
                self.freq_cmd = min(2.0, self.freq_cmd)
            else:
                print("SAFETY: Robot must be at standing height (0.74m) to change gait. Use keys 1/2 to adjust height.")
        elif key == "3":
            self.roll_cmd -= np.deg2rad(10)
            self.roll_cmd = max(self.roll_cmd, np.deg2rad(-30))  # Limit to -30 degrees
        elif key == "4":
            self.roll_cmd += np.deg2rad(10)
            self.roll_cmd = min(self.roll_cmd, np.deg2rad(30))   # Limit to +30 degrees
        elif key == "5":
            self.pitch_cmd -= np.deg2rad(10)
            self.pitch_cmd = max(self.pitch_cmd, np.deg2rad(-30))  # Limit to -30 degrees
        elif key == "6":
            self.pitch_cmd += np.deg2rad(10)
            self.pitch_cmd = min(self.pitch_cmd, np.deg2rad(30))   # Limit to +30 degrees
        elif key == "7":
            self.yaw_cmd -= np.deg2rad(10)
            self.yaw_cmd = max(self.yaw_cmd, np.deg2rad(-30))  # Limit to -30 degrees
        elif key == "8":
            self.yaw_cmd += np.deg2rad(10)
            self.yaw_cmd = min(self.yaw_cmd, np.deg2rad(30))   # Limit to +30 degrees
        
        # Enforce velocity zeroing if not at standing height
        if not at_standing_height and key not in ["1", "2", "z"]:
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

        if key:
            # Extract scalar values for formatting
            height_val = float(self.height_cmd[0] if isinstance(self.height_cmd, np.ndarray) else self.height_cmd)
            roll_val = float(self.roll_cmd[0] if isinstance(self.roll_cmd, np.ndarray) else self.roll_cmd)
            pitch_val = float(self.pitch_cmd[0] if isinstance(self.pitch_cmd, np.ndarray) else self.pitch_cmd)
            yaw_val = float(self.yaw_cmd[0] if isinstance(self.yaw_cmd, np.ndarray) else self.yaw_cmd)
            freq_val = float(self.freq_cmd[0] if isinstance(self.freq_cmd, np.ndarray) else self.freq_cmd)

            # Commented out to reduce log spam
            # print("-------------keyboard cmd-------------------")
            # print(f"Linear velocity command: [{self.cmd[0]:.4f}, {self.cmd[1]:.4f}, {self.cmd[2]:.4f}]")
            # print(f"Base height command: {height_val:.4f}")
            # print(f"Use policy action: {self.use_policy_action}")
            # print(f"roll deg angle: {np.rad2deg(roll_val):.4f}")
            # print(f"pitch deg angle: {np.rad2deg(pitch_val):.4f}")
            # print(f"yaw deg angle: {np.rad2deg(yaw_val):.4f}")
            # print(f"Gait frequency: {freq_val:.4f}")
