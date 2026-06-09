from typing import Any, Dict

import gymnasium as gym
import numpy as np

from gr00t_wbc.control.base.env import Env
from gr00t_wbc.control.envs.g1.utils.command_sender import BodyCommandSender
from gr00t_wbc.control.envs.g1.utils.state_processor import BodyStateProcessor


class G1Body(Env):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.body_state_processor = BodyStateProcessor(config=config)
        self.body_command_sender = BodyCommandSender(config=config)
    
    def wait_for_robot_ready(self, timeout: float = 10.0) -> bool:
        """Wait for robot to be ready to receive commands.

        Robot readiness is determined by BodyStateProcessor receiving low state,
        which happens during its initialization.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if robot is ready (state processor has received low state)
        """
        # BodyStateProcessor already waits for robot_low_state during __init__
        # If we reach here, the robot is ready
        return self.body_state_processor.robot_low_state is not None

    def observe(self) -> dict[str, any]:
        body_state = self.body_state_processor._prepare_low_state()  # (1, 148)
        assert body_state.shape == (1, 148)
        body_q = body_state[
            0, 7 : 7 + 12 + 3 + 7 + 7
        ]  # leg (12) + waist (3) + left arm (7) + right arm (7)
        body_dq = body_state[0, 42 : 42 + 12 + 3 + 7 + 7]
        body_ddq = body_state[0, 112 : 112 + 12 + 3 + 7 + 7]
        body_tau_est = body_state[0, 77 : 77 + 12 + 3 + 7 + 7]
        floating_base_pose = body_state[0, 0:7]
        floating_base_vel = body_state[0, 36:42]
        floating_base_acc = body_state[0, 106:112]
        torso_quat = body_state[0, 141:145]
        torso_ang_vel = body_state[0, 145:148]

        return {
            "body_q": body_q,
            "body_dq": body_dq,
            "body_ddq": body_ddq,
            "body_tau_est": body_tau_est,
            "floating_base_pose": floating_base_pose,
            "floating_base_vel": floating_base_vel,
            "floating_base_acc": floating_base_acc,
            "torso_quat": torso_quat,
            "torso_ang_vel": torso_ang_vel,
        }

    def queue_action(self, action: dict[str, any]):
        # action should contain body_q, body_dq, body_tau
        # Pass mode_machine from state processor to command sender
        # This is required for G1/H1-2 robots to echo mode_machine in commands
        mode_machine = None
        if self.body_state_processor.robot_low_state is not None:
            mode_machine = self.body_state_processor.robot_low_state.mode_machine

        self.body_command_sender.send_command(
            action["body_q"], action["body_dq"], action["body_tau"], mode_machine
        )

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "body_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "body_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "floating_base_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "floating_base_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "body_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "body_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "body_tau": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
            }
        )

    def close(self):
        """Close DDS resources."""
        print("[G1Body] Closing DDS resources...")
        try:
            if hasattr(self, 'body_state_processor') and self.body_state_processor:
                self.body_state_processor.close()
        except Exception as e:
            print(f"[G1Body] Error closing state processor: {e}")

        try:
            if hasattr(self, 'body_command_sender') and self.body_command_sender:
                self.body_command_sender.close()
        except Exception as e:
            print(f"[G1Body] Error closing command sender: {e}")
