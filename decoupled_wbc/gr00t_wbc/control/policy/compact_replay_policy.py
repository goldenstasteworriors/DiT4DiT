"""Replay policy for compact dataset format.

Reads parquet files using the dataset's own ``meta/modality.json`` so the policy
is robust to schema changes (e.g. with or without leg/waist columns, different
extra fields). All indexing into ``observation.state`` / ``action`` is done by
key name lookup against modality.json, not by hard-coded offsets.

Usage:
    python gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
        --compact_replay_path g1_real_data/test/data/chunk-000/episode_000000.parquet
"""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from gr00t_wbc.control.base.policy import Policy
from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_aloha_supplemental_info import (
    AlohaGripperMapping,
)


def _slice(arr: np.ndarray, span: dict) -> np.ndarray:
    """Slice an array using a {'start','end'} span from modality.json."""
    return arr[span["start"] : span["end"]]


class CompactReplayPolicy(Policy):
    """Replay policy that resolves field positions via meta/modality.json.

    The dataset directory layout is the LeRobot v2.1 layout:
        <root>/data/chunk-XXX/episode_YYYYYY.parquet
        <root>/meta/modality.json

    Every field needed for replay (`left_arm`, `right_arm`, `left_gripper`/
    `left_hand`, `right_gripper`/`right_hand`, `rpy`, `height`, `torso_vx`,
    `torso_vy`, `torso_vyaw`, `target_yaw`) is looked up by key name, so this
    works whether `observation.state` / `action` include leg/waist or not.
    """

    is_active = True

    def __init__(self, robot_model: RobotModel, parquet_path: str):
        self.robot_model = robot_model
        parquet_path = Path(parquet_path)
        self.df = pd.read_parquet(parquet_path)
        self._ctr = 0
        self._max_ctr = len(self.df)

        # Load modality.json from the dataset root: <root>/data/chunk-XXX/...
        # so root is parquet_path.parents[2].
        modality_path = parquet_path.parents[2] / "meta" / "modality.json"
        with open(modality_path, "r") as f:
            self.modality = json.load(f)
        self._state_modality = self.modality["state"]
        self._action_modality = self.modality["action"]

        # Detect hand-vs-gripper naming by what the modality.json declares.
        if "left_gripper" in self._state_modality:
            self._left_hand_key = "left_gripper"
            self._right_hand_key = "right_gripper"
            self.has_gripper = True
        elif "left_hand" in self._state_modality:
            self._left_hand_key = "left_hand"
            self._right_hand_key = "right_hand"
            self.has_gripper = False
        else:
            raise ValueError(
                "modality.json state must contain 'left_gripper' or 'left_hand'"
            )

        # Build the 4 spans we slice out of state to assemble the upper-body
        # pose sent to the lower body / WBC controller.
        # Order matters: this matches the original CompactReplayPolicy contract
        # (left_arm, left_gripper/hand, right_arm, right_gripper/hand).
        self._upper_body_spans = [
            self._state_modality["left_arm"],
            self._state_modality[self._left_hand_key],
            self._state_modality["right_arm"],
            self._state_modality[self._right_hand_key],
        ]

        # Gripper element positions inside observation.state (single index each).
        self._left_gripper_idx = self._state_modality[self._left_hand_key]["start"]
        self._right_gripper_idx = self._state_modality[self._right_hand_key]["start"]

        self._first_action = True  # Toggle lower body policy on first frame

        upper_body_dim = sum(s["end"] - s["start"] for s in self._upper_body_spans)
        print(
            f"[CompactReplayPolicy] Loaded {self._max_ctr} frames from {parquet_path}"
            f" (modality.json: {modality_path.name}, upper_body_dim={upper_body_dim})"
        )

    def _get_modality_field(
        self, row, modality_section: dict, key: str, default_column: str, default=None
    ):
        """Look up `key` under a modality section and slice the right parquet column.

        - If `key` is absent from the section -> return `default`.
        - If the entry has `original_key`, slice that column instead of
          `default_column` (this is how the legacy schema points action's
          `navigate_command` at the standalone `teleop.navigate_command`
          column, etc.).
        """
        span = modality_section.get(key)
        if span is None:
            return default
        column = span.get("original_key", default_column)
        if column not in row.index:
            return default
        arr = np.atleast_1d(np.asarray(row[column]))
        return _slice(arr, span)

    def get_action(self) -> dict:
        row = self.df.iloc[self._ctr]
        state = np.asarray(row["observation.state"])

        self._ctr += 1
        if self._ctr >= self._max_ctr:
            self._ctr = 0

        target_upper_body_pose = np.concatenate(
            [_slice(state, span) for span in self._upper_body_spans]
        )

        # rpy / height / nav: prefer new compact keys, fall back to legacy keys
        # (`base_height_command`, `navigate_command`) that point at separate
        # `teleop.*` columns via `original_key`.
        rpy = self._get_modality_field(
            row, self._action_modality, "rpy", "action", default=np.zeros(3)
        )
        height_arr = self._get_modality_field(
            row, self._action_modality, "height", "action",
            default=self._get_modality_field(
                row, self._action_modality, "base_height_command", "action",
                default=np.array([0.0]),
            ),
        )
        vx = self._get_modality_field(
            row, self._action_modality, "torso_vx", "action", default=None
        )
        vy = self._get_modality_field(
            row, self._action_modality, "torso_vy", "action", default=None
        )
        vyaw = self._get_modality_field(
            row, self._action_modality, "torso_vyaw", "action", default=None
        )
        if vx is None or vy is None or vyaw is None:
            navigate_cmd = self._get_modality_field(
                row, self._action_modality, "navigate_command", "action",
                default=np.zeros(3),
            )
        else:
            navigate_cmd = np.concatenate([vx, vy, vyaw])

        result = {
            "target_upper_body_pose": target_upper_body_pose,
            "navigate_cmd": navigate_cmd,
            "base_height_command": float(height_arr[0]),
            "torso_orientation_rpy": np.asarray(rpy),
            "timestamp": time.time(),
        }

        # Activate lower body policy on the first frame
        if self._first_action:
            result["toggle_policy_action"] = True
            self._first_action = False

        # Convert gripper values to trigger format for ALOHA DDS commands.
        # Parquet stores Arduino/DDS convention: 0.0 = closed, 0.065 = open
        # Trigger convention: 0.0 = released/open, 1.0 = pressed/closed.
        if self.has_gripper:
            left_gripper_val = float(state[self._left_gripper_idx])
            right_gripper_val = float(state[self._right_gripper_idx])
            left_trigger = np.clip(
                1.0 - left_gripper_val / AlohaGripperMapping.HW_RANGE, 0.0, 1.0
            )
            right_trigger = np.clip(
                1.0 - right_gripper_val / AlohaGripperMapping.HW_RANGE, 0.0, 1.0
            )
            result["left_fingers"] = {"trigger": left_trigger}
            result["right_fingers"] = {"trigger": right_trigger}

        return result

    def set_observation(self, observation: dict):
        pass

    def get_observation(self) -> dict:
        return {"timestamp": time.time()}
