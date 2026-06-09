from copy import deepcopy
from typing import Dict

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.base.humanoid_env import Hands, HumanoidEnv
from gr00t_wbc.control.envs.g1.g1_body import G1Body
from gr00t_wbc.control.envs.g1.g1_hand import G1ThreeFingerHand, G1AlohaHand
from gr00t_wbc.control.envs.g1.sim.simulator_factory import SimulatorFactory, init_channel
from gr00t_wbc.control.envs.g1.utils.joint_safety import JointSafetyMonitor
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_aloha_supplemental_info import (
    AlohaGripperMapping,
)
from gr00t_wbc.control.utils.ros_utils import ROSManager


class G1Env(HumanoidEnv):
    def __init__(
        self,
        env_name: str = "default",
        robot_model: RobotModel = None,
        wbc_version: str = "v2",
        config: Dict[str, any] = None,
        **kwargs,
    ):
        super().__init__()
        self.robot_model = deepcopy(robot_model)  # need to cache FK results
        self.config = config

        # Initialize safety monitor (visualization disabled)
        self.safety_monitor = JointSafetyMonitor(
            robot_model, enable_viz=False, env_type=self.config.get("ENV_TYPE", "real")
        )
        self.last_obs = None
        self.last_safety_ok = True  # Track last safety status from queue_action

        init_channel(config=self.config)

        # Initialize body and hands
        self._body = G1Body(config=self.config)
        
        # Wait for robot to be ready (get mode_machine from robot) - only in real mode
        if self.config.get("ENV_TYPE") != "sim":
            if not self._body.wait_for_robot_ready(timeout=10.0):
                print("[G1Env] WARNING: Robot not responding, continuing anyway...")
            else:
                print("[G1Env] Robot ready")

        self.with_hands = config.get("with_hands", True)
        self.hand_type = config.get("hand_type", "aloha")  # "aloha" or "three_finger"

        # Debug: print hand configuration
        print(f"[G1Env] with_hands={self.with_hands}, hand_type={self.hand_type}")

        # Gravity compensation settings
        self.enable_gravity_compensation = config.get("enable_gravity_compensation", False)
        self.gravity_compensation_joints = config.get("gravity_compensation_joints", ["arms"])

        if self.enable_gravity_compensation:
            print(
                f"Gravity compensation enabled for joint groups: {self.gravity_compensation_joints}"
            )
        if self.with_hands:
            self._hands = Hands()
            if self.hand_type == "aloha":
                print("Initializing ALOHA grippers (1-DOF force-feedback)")
                aloha_dds_topic = config.get("aloha_dds_topic", "rt/aloha_hand/cmd")
                aloha_state_topic = config.get("aloha_state_topic", "rt/aloha_hand/state")
                use_aloha_feedback = config.get("use_aloha_feedback", True)  # Enable feedback by default

                # Create shared resources for both hands (avoids duplicate DDS subscribers)
                from gr00t_wbc.control.envs.g1.utils.command_sender import AlohaHandCommandSender
                from gr00t_wbc.control.envs.g1.utils.state_processor import AlohaSharedStateProcessor

                # Single command sender for both hands
                self.aloha_shared_command_sender = AlohaHandCommandSender(dds_topic=aloha_dds_topic)

                # Single state processor for both hands (subscribes once to feedback topic)
                self.aloha_shared_state_processor = AlohaSharedStateProcessor(
                    state_topic=aloha_state_topic,
                    use_feedback=use_aloha_feedback
                )

                # Pass the shared resources to both hands
                self._hands.left = G1AlohaHand(
                    is_left=True,
                    dds_topic=aloha_dds_topic,
                    use_feedback=use_aloha_feedback,
                    feedback_topic=aloha_state_topic,
                    shared_command_sender=self.aloha_shared_command_sender,
                    shared_state_processor=self.aloha_shared_state_processor
                )
                self._hands.right = G1AlohaHand(
                    is_left=False,
                    dds_topic=aloha_dds_topic,
                    use_feedback=use_aloha_feedback,
                    feedback_topic=aloha_state_topic,
                    shared_command_sender=self.aloha_shared_command_sender,
                    shared_state_processor=self.aloha_shared_state_processor
                )
                # Track current gripper positions for keyboard control
                # Hardware range: 0.0 = fully open, 0.065 = fully closed
                self.aloha_left_pos = 0.0  # Start open
                self.aloha_right_pos = 0.0  # Start open
            else:  # Default to three_finger
                print("Initializing G1 three-finger hands (7-DOF)")
                self._hands.left = G1ThreeFingerHand(is_left=True)
                self._hands.right = G1ThreeFingerHand(is_left=False)

        # Initialize simulator if in simulation mode
        self.use_sim = self.config.get("ENV_TYPE") == "sim"

        if self.use_sim:
            # Create simulator using factory

            kwargs.update(
                {
                    "onscreen": self.config.get("ENABLE_ONSCREEN", True),
                    "offscreen": self.config.get("ENABLE_OFFSCREEN", False),
                }
            )
            self.sim = SimulatorFactory.create_simulator(
                config=self.config,
                env_name=env_name,
                wbc_version=wbc_version,
                body_ik_solver_settings_type=kwargs.get("body_ik_solver_settings_type", "default"),
                **kwargs,
            )
        else:
            self.sim = None

            # using the real robot
            self.calibrate_hands()

        # Initialize ROS 2 node
        self.ros_manager = ROSManager(node_name="g1_env")
        self.ros_node = self.ros_manager.node

        self.delay_list = []
        self.visualize_delay = False
        self.print_delay_interval = 100
        self.cnt = 0

    def start_simulator(self):
        # imag epublish disabled since the sim is running in a sub-thread
        SimulatorFactory.start_simulator(self.sim, as_thread=True, enable_image_publish=False)

    def step_simulator(self):
        sim_num_steps = int(self.config["REWARD_DT"] / self.config["SIMULATE_DT"])
        for _ in range(sim_num_steps):
            self.sim.sim_env.sim_step()
        self.sim.sim_env.update_viewer()

    def body(self) -> G1Body:
        return self._body

    def hands(self) -> Hands:
        if not self.with_hands:
            raise RuntimeError(
                "Hands not initialized. Use --with_hands True to enable hand functionality."
            )
        return self._hands

    def observe(self) -> Dict[str, any]:
        # Get observations from body and hands
        body_obs = self.body().observe()

        body_q = body_obs["body_q"]
        body_dq = body_obs["body_dq"]
        body_ddq = body_obs["body_ddq"]
        body_tau_est = body_obs["body_tau_est"]

        if self.with_hands:
            left_hand_obs = self.hands().left.observe()
            right_hand_obs = self.hands().right.observe()
            left_hand_q = left_hand_obs["hand_q"]
            right_hand_q = right_hand_obs["hand_q"]
            left_hand_dq = left_hand_obs["hand_dq"]
            right_hand_dq = right_hand_obs["hand_dq"]
            left_hand_ddq = left_hand_obs["hand_ddq"]
            right_hand_ddq = right_hand_obs["hand_ddq"]
            left_hand_tau_est = left_hand_obs["hand_tau_est"]
            right_hand_tau_est = right_hand_obs["hand_tau_est"]

            # No padding needed - hand DOFs from observations match robot model:
            # - ALOHA robot model (31 DOF): 1-DOF grippers, G1AlohaHand returns 1-DOF
            # - Three-finger robot model (43 DOF): 7-DOF hands, G1ThreeFingerHand returns 7-DOF

            # Body and hand joint measurements come in actuator order, so we need to convert them to joint order
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=left_hand_q,
                right_hand_actuated_joint_values=right_hand_q,
            )
            whole_dq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_dq,
                left_hand_actuated_joint_values=left_hand_dq,
                right_hand_actuated_joint_values=right_hand_dq,
            )
            whole_ddq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_ddq,
                left_hand_actuated_joint_values=left_hand_ddq,
                right_hand_actuated_joint_values=right_hand_ddq,
            )
            whole_tau_est = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_tau_est,
                left_hand_actuated_joint_values=left_hand_tau_est,
                right_hand_actuated_joint_values=right_hand_tau_est,
            )
        else:
            # Body and hand joint measurements come in actuator order, so we need to convert them to joint order
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
            )
            whole_dq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_dq,
            )
            whole_ddq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_ddq,
            )
            whole_tau_est = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_tau_est,
            )

        eef_obs = self.get_eef_obs(whole_q)

        obs = {
            "q": whole_q,
            "dq": whole_dq,
            "ddq": whole_ddq,
            "tau_est": whole_tau_est,
            "floating_base_pose": body_obs["floating_base_pose"],
            "floating_base_vel": body_obs["floating_base_vel"],
            "floating_base_acc": body_obs["floating_base_acc"],
            "wrist_pose": np.concatenate([eef_obs["left_wrist_pose"], eef_obs["right_wrist_pose"]]),
            "torso_quat": body_obs["torso_quat"],
            "torso_ang_vel": body_obs["torso_ang_vel"],
        }

        if self.use_sim and self.sim:
            obs.update(self.sim.get_privileged_obs())

        # Store last observation for safety checking
        self.last_obs = obs

        return obs

    @property
    def observation_space(self) -> gym.Space:
        # @todo: check if the low and high bounds are correct for body_obs.
        q_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        dq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        ddq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        tau_est_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        floating_base_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        floating_base_vel_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        floating_base_acc_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        wrist_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 7,))
        return gym.spaces.Dict(
            {
                "floating_base_pose": floating_base_pose_space,
                "floating_base_vel": floating_base_vel_space,
                "floating_base_acc": floating_base_acc_space,
                "q": q_space,
                "dq": dq_space,
                "ddq": ddq_space,
                "tau_est": tau_est_space,
                "wrist_pose": wrist_pose_space,
            }
        )

    def queue_action(self, action: Dict[str, any]):
        # Safety check
        if self.last_obs is not None:
            safety_result = self.safety_monitor.handle_violations(self.last_obs, action)
            action = safety_result["action"]

        # Map action from joint order to actuator order
        body_actuator_q = self.robot_model.get_body_actuated_joints(action["q"])
        self.body().queue_action(
            {
                "body_q": body_actuator_q,
                "body_dq": np.zeros_like(body_actuator_q),
                "body_tau": np.zeros_like(body_actuator_q),
            }
        )

        if self.with_hands:
            # SHORTCUT PATH for Pico VR controller teleoperation with ALOHA grippers
            # This bypasses normal IK/WBC flow and sends trigger values directly to hardware.
            # Activated when:
            # 1. run_teleop_policy_loop.py is running (provides action with finger data)
            # 2. hand_control_device is "pico" (VR controller with triggers)
            # 3. Trigger data is present in the action
            # When run_g1_control_loop.py is running alone (no teleop), this shortcut is skipped.
            hand_control_device = self.config.get("hand_control_device", "pico")
            has_finger_data = "left_fingers" in action and "right_fingers" in action

            if (self.hand_type == "aloha" and
                hand_control_device == "pico" and
                has_finger_data):
                left_trigger = action["left_fingers"].get("trigger", None) if isinstance(action["left_fingers"], dict) else None
                right_trigger = action["right_fingers"].get("trigger", None) if isinstance(action["right_fingers"], dict) else None

                if left_trigger is not None and right_trigger is not None:
                    # Map trigger values (0.0-1.0) to hardware gripper positions
                    # AlohaGripperMapping.trigger_to_hardware() handles Arduino hardware inversion
                    # Result: Trigger released (0.0) → gripper opens, Trigger pressed (1.0) → gripper closes
                    left_gripper_hw = AlohaGripperMapping.trigger_to_hardware(left_trigger)
                    right_gripper_hw = AlohaGripperMapping.trigger_to_hardware(right_trigger)

                    # Only send command if position changed significantly (deadband = 0.0005m = 0.5mm)
                    POSITION_DEADBAND = 0.0005
                    left_changed = abs(left_gripper_hw - self.aloha_left_pos) > POSITION_DEADBAND
                    right_changed = abs(right_gripper_hw - self.aloha_right_pos) > POSITION_DEADBAND

                    if left_changed or right_changed:
                        # Update internal tracking variables (in hardware units)
                        self.aloha_left_pos = left_gripper_hw
                        self.aloha_right_pos = right_gripper_hw

                        # Send commands directly via DDS (hardware units)
                        if self.aloha_shared_command_sender:
                            # Diagnostic: Log trigger values and hardware commands
                            print(f"[GRIPPER_CMD] Triggers: L={left_trigger:.3f} R={right_trigger:.3f} | HW Commands: L={self.aloha_left_pos:.4f} R={self.aloha_right_pos:.4f}")
                            self.aloha_shared_command_sender.send_dual_command(
                                self.aloha_left_pos, self.aloha_right_pos
                            )

                    # No need to queue hand actions - DDS commands are sent directly
                    return
            

    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))

    def calibrate_hands(self):
        """Calibrate the hand joint qpos if real robot"""
        if self.with_hands:
            print("calibrating left hand")
            self.hands().left.calibrate_hand()
            print("calibrating right hand")
            self.hands().right.calibrate_hand()
        else:
            print("Skipping hand calibration - hands disabled")

    def set_ik_indicator(self, teleop_cmd):
        """Set the IK indicators for the simulator"""
        if self.config["SIMULATOR"] == "robocasa":
            if "left_wrist" in teleop_cmd and "right_wrist" in teleop_cmd:
                left_wrist_input_pose = teleop_cmd["left_wrist"]
                right_wrist_input_pose = teleop_cmd["right_wrist"]
                ik_wrapper = self.sim.env.env.unwrapped.env
                ik_wrapper.set_target_poses_outside_env(
                    [left_wrist_input_pose, right_wrist_input_pose]
                )
        else:
            raise NotImplementedError("IK indicators are only implemented for robocasa simulator")

    def set_sync_mode(self, sync_mode: bool, steps_per_action: int = 4):
        """When set to True, the simulator will wait for the action to be sent to it"""
        if self.config["SIMULATOR"] == "robocasa":
            self.sim.set_sync_mode(sync_mode, steps_per_action)

    def reset(self):
        if self.sim:
            self.sim.reset()

    def close(self):
        """Close all resources including DDS publishers/subscribers."""
        print("[G1Env] Closing environment...")

        # Close simulator if running
        if self.sim:
            try:
                print("[G1Env] Closing simulator...")
                self.sim.close()
            except Exception as e:
                print(f"[G1Env] Error closing simulator: {e}")

        # Close body (includes body state processor and command sender)
        if hasattr(self, '_body') and self._body:
            try:
                print("[G1Env] Closing body DDS resources...")
                self._body.close()
            except Exception as e:
                print(f"[G1Env] Error closing body: {e}")

        # Close hands
        if self.with_hands and hasattr(self, '_hands'):
            try:
                if self._hands.left:
                    print("[G1Env] Closing left hand DDS resources...")
                    self._hands.left.close()
            except Exception as e:
                print(f"[G1Env] Error closing left hand: {e}")

            try:
                if self._hands.right:
                    print("[G1Env] Closing right hand DDS resources...")
                    self._hands.right.close()
            except Exception as e:
                print(f"[G1Env] Error closing right hand: {e}")

        # Close shared ALOHA resources (if using ALOHA grippers)
        if hasattr(self, 'aloha_shared_command_sender') and self.aloha_shared_command_sender:
            try:
                print("[G1Env] Closing ALOHA shared command sender...")
                self.aloha_shared_command_sender.close()
            except Exception as e:
                print(f"[G1Env] Error closing ALOHA command sender: {e}")

        if hasattr(self, 'aloha_shared_state_processor') and self.aloha_shared_state_processor:
            try:
                print("[G1Env] Closing ALOHA shared state processor...")
                self.aloha_shared_state_processor.close()
            except Exception as e:
                print(f"[G1Env] Error closing ALOHA state processor: {e}")

        # Close ROS manager
        if hasattr(self, 'ros_manager') and self.ros_manager:
            try:
                print("[G1Env] Shutting down ROS node...")
                self.ros_manager.shutdown()
            except Exception as e:
                print(f"[G1Env] Error shutting down ROS: {e}")

        print("[G1Env] Environment closed")

    def robot_model(self) -> RobotModel:
        return self.robot_model

    def get_reward(self):
        if self.sim:
            return self.sim.get_reward()

    def reset_obj_pos(self):
        if hasattr(self.sim, "base_env") and hasattr(self.sim.base_env, "reset_obj_pos"):
            self.sim.base_env.reset_obj_pos()

    def get_eef_obs(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        self.robot_model.cache_forward_kinematics(q)
        eef_obs = {}
        for side in ["left", "right"]:
            wrist_placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            wrist_pos, wrist_quat = wrist_placement.translation[:3], R.from_matrix(
                wrist_placement.rotation
            ).as_quat(scalar_first=True)
            eef_obs[f"{side}_wrist_pose"] = np.concatenate([wrist_pos, wrist_quat])

        return eef_obs

    def get_joint_safety_status(self) -> bool:
        """Get current joint safety status from the last queue_action safety check.

        Returns:
            bool: True if joints are safe (no shutdown required), False if unsafe
        """
        return self.last_safety_ok

    def handle_keyboard_button(self, key):
        # Handle ALOHA gripper controls
        if self.with_hands and self.hand_type == "aloha":
            gripper_step = 0.013  # 13mm step size
            gripper_max = 0.065   # Maximum closed position

            # Track which gripper(s) to update
            update_left = False
            update_right = False

            if key == "j":  # Left gripper close
                self.aloha_left_pos = min(gripper_max, self.aloha_left_pos + gripper_step)
                update_left = True
                print(f"Left gripper closing: {self.aloha_left_pos:.4f}m")
            elif key == "u":  # Left gripper open (changed from 'k')
                self.aloha_left_pos = max(0.0, self.aloha_left_pos - gripper_step)
                update_left = True
                print(f"Left gripper opening: {self.aloha_left_pos:.4f}m")
            elif key == "m":  # Right gripper close (changed from 'l' to avoid conflict)
                self.aloha_right_pos = min(gripper_max, self.aloha_right_pos + gripper_step)
                update_right = True
                print(f"Right gripper closing: {self.aloha_right_pos:.4f}m")
            elif key == "n":  # Right gripper open (changed from ';')
                self.aloha_right_pos = max(0.0, self.aloha_right_pos - gripper_step)
                update_right = True
                print(f"Right gripper opening: {self.aloha_right_pos:.4f}m")

            # Send command if any gripper was updated
            if update_left or update_right:
                # For real robot: send DDS command
                if not self.use_sim:
                    self.aloha_shared_command_sender.send_dual_command(
                        self.aloha_left_pos, self.aloha_right_pos
                    )
                    # Update state processors
                    self._hands.left.state_processor.update_state_from_command(self.aloha_left_pos)
                    self._hands.right.state_processor.update_state_from_command(self.aloha_right_pos)
                return

        # Handle simulator keyboard buttons
        if self.use_sim and self.config.get("SIMULATOR", "mujoco") == "mujoco":
            self.sim.handle_keyboard_button(key)
    


if __name__ == "__main__":
    env = G1Env(robot_model=instantiate_g1_robot_model(), wbc_version="gear_wbc")
    while True:
        print(env.observe())
