import time

import numpy as np
import pandas as pd

from gr00t_wbc.control.base.policy import Policy
from gr00t_wbc.control.main.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
)
from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_aloha_supplemental_info import (
    AlohaGripperMapping,
)
from gr00t_wbc.data.viz.rerun_viz import RerunViz


class LerobotReplayPolicy(Policy):
    """Replay policy for Lerobot dataset, so we can replay the dataset
    and just use the action from the dataset.

    Args:
        parquet_path: Path to the parquet file containing the dataset.
    """

    is_active = True  # by default, the replay policy is active

    def __init__(self, robot_model: RobotModel, parquet_path: str, use_viz: bool = False):
        # self.dataset = LerobotDataset(dataset_path)
        self.parquet_path = parquet_path
        self._ctr = 0
        # read the parquet file
        self.df = pd.read_parquet(self.parquet_path)
        self._max_ctr = len(self.df)
        # get the action from the dataframe
        self.action = self.df.iloc[self._ctr]["action"]
        self.use_viz = use_viz
        if self.use_viz:
            self.viz = RerunViz(
                image_keys=["egoview_image"],
                tensor_keys=[
                    "left_arm_qpos",
                    "left_hand_qpos",
                    "right_arm_qpos",
                    "right_hand_qpos",
                ],
                window_size=5.0,
            )
        self.robot_model = robot_model
        self.upper_body_joint_indices = self.robot_model.get_joint_group_indices("upper_body")

        # Get gripper joint indices for extracting trigger values during replay
        try:
            self.left_gripper_indices = self.robot_model.get_joint_group_indices("left_gripper")
            self.right_gripper_indices = self.robot_model.get_joint_group_indices("right_gripper")
            self.has_gripper = True
        except (KeyError, ValueError):
            self.has_gripper = False

    def get_action(self) -> dict[str, any]:
        # get the action from the dataframe
        action = self.df.iloc[self._ctr]["action"]
        # action = self.df.iloc[self._ctr]["observation.state"]

        wrist_pose = self.df.iloc[self._ctr]["action.eef"]
        navigate_cmd = self.df.iloc[self._ctr].get("teleop.navigate_command", DEFAULT_NAV_CMD)
        base_height_cmd = self.df.iloc[self._ctr].get(
            "teleop.base_height_command", DEFAULT_BASE_HEIGHT
        )

        self._ctr += 1
        if self._ctr >= self._max_ctr:
            self._ctr = 0
        # print(f"Replay {self._ctr} / {self._max_ctr}")
        if self.use_viz:
            self.viz.plot_tensors(
                {
                    "left_arm_qpos": action[self.robot_model.get_joint_group_indices("left_arm")]
                    + 15,
                    "left_hand_qpos": action[self.robot_model.get_joint_group_indices("left_hand")]
                    + 15,
                    "right_arm_qpos": action[self.robot_model.get_joint_group_indices("right_arm")]
                    + 15,
                    "right_hand_qpos": action[
                        self.robot_model.get_joint_group_indices("right_hand")
                    ]
                    + 15,
                },
                time.monotonic(),
            )

        result = {
            "target_upper_body_pose": action[self.upper_body_joint_indices],
            "wrist_pose": wrist_pose,
            "navigate_cmd": navigate_cmd,
            "base_height_cmd": base_height_cmd,
            "timestamp": time.time(),
        }

        # Extract gripper values from action and convert to trigger format
        # so that G1Env's ALOHA gripper shortcut path can send DDS commands.
        #
        # Parquet action stores gripper in Arduino/DDS convention (from feedback):
        #   0.0 = closed, 0.065 = open
        # Trigger convention (what trigger_to_hardware expects):
        #   0.0 = released/open, 1.0 = pressed/closed
        # So: trigger = 1.0 - (gripper_val / HW_RANGE)
        if self.has_gripper:
            left_gripper_val = float(action[self.left_gripper_indices[0]])
            right_gripper_val = float(action[self.right_gripper_indices[0]])
            left_trigger = np.clip(1.0 - left_gripper_val / AlohaGripperMapping.HW_RANGE, 0.0, 1.0)
            right_trigger = np.clip(1.0 - right_gripper_val / AlohaGripperMapping.HW_RANGE, 0.0, 1.0)
            result["left_fingers"] = {"trigger": left_trigger}
            result["right_fingers"] = {"trigger": right_trigger}

        return result

    def action_to_cmd(self, action: dict[str, any]) -> dict[str, any]:
        action["target_upper_body_pose"] = action["q"][
            self.robot_model.get_joint_group_indices("upper_body")
        ]
        del action["q"]
        return action

    def set_observation(self, observation: dict[str, any]):
        pass

    def get_observation(self) -> dict[str, any]:
        return {
            "wrist_pose": self.df.iloc[self._ctr - 1].get(
                "observation.eef_state", DEFAULT_WRIST_POSE
            ),
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    policy = LerobotReplayPolicy(
        parquet_path="outputs/g1-open-hands-may7/data/chunk-000/episode_000000.parquet"
    )
    action = policy.get_action()
    print(action)
