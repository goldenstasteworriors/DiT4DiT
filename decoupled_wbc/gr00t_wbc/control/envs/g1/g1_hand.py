import time

import gymnasium as gym
import numpy as np

from gr00t_wbc.control.base.env import Env
from gr00t_wbc.control.envs.g1.utils.command_sender import HandCommandSender
from gr00t_wbc.control.envs.g1.utils.state_processor import HandStateProcessor


class G1ThreeFingerHand(Env):
    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        self.hand_state_processor = HandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = HandCommandSender(is_left=self.is_left)
        self.hand_q_offset = np.zeros(7)

    def observe(self) -> dict[str, any]:
        hand_state = self.hand_state_processor._prepare_low_state()  # (1, 28)
        assert hand_state.shape == (1, 28)

        # Apply offset to the hand state
        hand_state[0, :7] = hand_state[0, :7] + self.hand_q_offset

        hand_q = hand_state[0, :7]
        hand_dq = hand_state[0, 7:14]
        hand_ddq = hand_state[0, 21:28]
        hand_tau_est = hand_state[0, 14:21]

        # Return the state for this specific hand (left or right)
        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        # Apply offset to the hand target
        action["hand_q"] = action["hand_q"] - self.hand_q_offset

        # action should contain hand_q
        self.hand_command_sender.send_command(action["hand_q"])

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict({"hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))})

    def calibrate_hand(self):
        hand_obs = self.observe()
        hand_q = hand_obs["hand_q"]

        hand_q_target = np.zeros_like(hand_q)
        hand_q_target[0] = hand_q[0]

        # joint limit
        hand_q0_upper_limit = np.deg2rad(60)  # lower limit is -60

        # move the figure counterclockwise until the limit
        while True:

            if hand_q_target[0] - hand_q[0] < np.deg2rad(60):
                hand_q_target[0] += np.deg2rad(10)
            else:
                self.hand_q_offset[0] = hand_q0_upper_limit - hand_q[0]
                break

            self.queue_action({"hand_q": hand_q_target})

            hand_obs = self.observe()
            hand_q = hand_obs["hand_q"]

            time.sleep(0.1)

        print("done calibration, q0 offset (deg):", np.rad2deg(self.hand_q_offset[0]))

        # done calibrating, set target to zero
        self.hand_q_target = np.zeros_like(hand_q)
        self.queue_action({"hand_q": self.hand_q_target})

    def close(self):
        """Close DDS resources."""
        try:
            if hasattr(self, 'hand_state_processor') and self.hand_state_processor:
                self.hand_state_processor.close()
        except Exception as e:
            print(f"[G1ThreeFingerHand] Error closing state processor: {e}")

        try:
            if hasattr(self, 'hand_command_sender') and self.hand_command_sender:
                self.hand_command_sender.close()
        except Exception as e:
            print(f"[G1ThreeFingerHand] Error closing command sender: {e}")


class G1AlohaHand(Env):
    """ALOHA-style gripper hand for G1 robot.

    This class provides a simplified 1-DOF gripper interface that sends commands
    to the ALOHA feedback server via DDS. The server runs on the robot's onboard
    computer and manages the Arduino/OpenRB serial communication.

    Command flow: G1AlohaHand (host) -> DDS -> aloha_gripper_dds_bridge (robot) ->
                  aloha_feedback_client (robot) -> Arduino/OpenRB -> Dynamixel motors

    All values use hardware convention:
    - Range: 0.0 (fully open) to 0.065 (fully closed) meters
    """

    def __init__(self, is_left: bool = True, dds_topic: str = "rt/aloha_hand/cmd",
                 use_feedback: bool = False, feedback_topic: str = None,
                 shared_command_sender=None, shared_state_processor=None):
        """Initialize ALOHA hand interface.

        Args:
            is_left: True for left hand, False for right hand
            dds_topic: DDS topic name for sending gripper commands
            use_feedback: If True, subscribe to feedback topic for real-time state
            feedback_topic: Optional DDS topic for state feedback (e.g., "rt/aloha_hand/state")
            shared_command_sender: Optional shared AlohaHandCommandSender instance (recommended)
            shared_state_processor: Optional shared AlohaSharedStateProcessor instance (recommended)
                                   This avoids duplicate DDS subscribers for left/right hands.
        """
        super().__init__()
        self.is_left = is_left
        self.dds_topic = dds_topic
        self.use_feedback = use_feedback

        # Import command sender and state processor
        from gr00t_wbc.control.envs.g1.utils.command_sender import AlohaHandCommandSender
        from gr00t_wbc.control.envs.g1.utils.state_processor import AlohaHandStateProcessor

        # Use shared command sender if provided, otherwise create new one
        if shared_command_sender is not None:
            self.command_sender = shared_command_sender
        else:
            self.command_sender = AlohaHandCommandSender(dds_topic=dds_topic)

        # Use shared state processor if provided (recommended to avoid duplicate subscribers)
        self.state_processor = AlohaHandStateProcessor(
            is_left=is_left,
            state_topic=feedback_topic,
            use_feedback=use_feedback,
            shared_processor=shared_state_processor
        )

        # Hardware gripper range: 0.0 (fully open) to 0.065 (fully closed)
        self.gripper_min = 0.0
        self.gripper_max = 0.065

    def observe(self) -> dict[str, any]:
        """Get current gripper state.
        
        Returns state from AlohaHandStateProcessor, which uses either:
        - Real-time DDS feedback (if use_feedback=True and topic configured)
        - Command echo estimation (if use_feedback=False)
        
        Returns:
            Dictionary with single gripper position value and dynamics
        """
        # Get state from processor (shape: 1x4)
        hand_state = self.state_processor._prepare_low_state()
        
        if hand_state is None:
            # Fallback to zeros if no state available
            hand_state = np.zeros((1, 4))
        
        # Extract components: [q, dq, tau_est, ddq]
        hand_q = hand_state[0, 0:1]      # Position (1D)
        hand_dq = hand_state[0, 1:2]     # Velocity (1D)
        hand_tau_est = hand_state[0, 2:3]  # Load/Torque (1D)
        hand_ddq = hand_state[0, 3:4]    # Acceleration (1D)
        
        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        """Send gripper command to ALOHA controller.

        Args:
            action: Dictionary containing 'hand_q' with single gripper position value
                   Range: 0.0 (open) to 0.065 (closed) in hardware convention
        """
        hand_q = action["hand_q"]

        # Clamp to valid hardware range
        gripper_pos = np.clip(hand_q[0], self.gripper_min, self.gripper_max)

        # Update state processor with commanded position (for echo estimation)
        if not self.use_feedback:
            self.state_processor.update_state_from_command(gripper_pos)

        # Send command via DDS (left hand uses x component, right hand uses y component)
        self.command_sender.send_command(gripper_pos, self.is_left)

    def observation_space(self) -> gym.Space:
        """Define observation space for ALOHA gripper (1-DOF) in hardware convention."""
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=self.gripper_min, high=self.gripper_max, shape=(1,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            }
        )

    def action_space(self) -> gym.Space:
        """Define action space for ALOHA gripper (1-DOF) in hardware convention."""
        return gym.spaces.Dict({
            "hand_q": gym.spaces.Box(low=self.gripper_min, high=self.gripper_max, shape=(1,))
        })

    def calibrate_hand(self):
        """ALOHA grippers don't require calibration - they use absolute position control."""
        print(f"ALOHA {'left' if self.is_left else 'right'} gripper: No calibration needed (absolute positioning)")
        # Open gripper to starting position (0.0 = fully open)
        self.queue_action({"hand_q": np.array([self.gripper_min])})

    def close(self):
        """Close DDS resources.

        Note: If using a shared command sender, it should be closed separately
        by the G1Env that owns it.
        """
        try:
            if hasattr(self, 'state_processor') and self.state_processor:
                self.state_processor.close()
        except Exception as e:
            print(f"[G1AlohaHand] Error closing state processor: {e}")