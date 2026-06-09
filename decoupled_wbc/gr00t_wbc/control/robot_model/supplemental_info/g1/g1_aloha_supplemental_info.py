"""G1 Body 29DOF + ALOHA Gripper supplemental info.

This module provides the supplemental info for the G1 robot with:
- 29 DOF body (legs, waist, arms)
- 2 DOF ALOHA grippers (1 per hand, prismatic)
Total: 31 DOF
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AlohaGripperMapping:
    """Mapping between URDF joint values and hardware gripper commands.

    This class is provided for simulation<->hardware conversion when needed.

    NOTE: The control system and training data use HARDWARE convention (0.0 to 0.065).
    Only use this mapping when interfacing with MuJoCo simulation that uses URDF values.

    ALOHA gripper has two fingers that move symmetrically (mimic joint).
    The URDF joint represents ONE finger's position:
    - URDF joint range: -0.01 to 0.0225 meters (one finger, 32.5mm travel)
    - Total aperture = 2x joint value (both fingers move)

    Semantic/Training convention (what policy outputs and learns):
    - Semantic range: 0.0 (fully open) to 0.065 (fully closed) meters
    - This is what appears in training data and policy outputs

    Arduino hardware reality (inverted):
    - Arduino interprets: 0.0 = closed, 0.065 = open (opposite!)
    - The trigger_to_hardware() method handles this inversion automatically

    URDF to Semantic mapping:
    - URDF joint -0.01 (open) -> Semantic 0.0 (open)  [fingers at max spread]
    - URDF joint 0.0225 (closed) -> Semantic 0.065 (closed) [fingers touching]
    """

    # URDF joint limits (single finger position)
    URDF_MIN = -0.01    # Fully open (fingers spread apart)
    URDF_MAX = 0.0225   # Fully closed (fingers together)
    URDF_RANGE = URDF_MAX - URDF_MIN  # 0.0325m = 32.5mm per finger

    # Semantic/training convention (NOT Arduino hardware reality!)
    # These define the semantic meaning for training data and policy outputs
    HW_MIN = 0.0    # Semantic: Fully open (Arduino actually interprets as closed!)
    HW_MAX = 0.065  # Semantic: Fully closed (Arduino actually interprets as open!)
    HW_RANGE = HW_MAX - HW_MIN  # 0.065m = 65mm total

    @classmethod
    def urdf_to_hardware(cls, urdf_value: float) -> float:
        """Convert URDF joint value to hardware command.

        Args:
            urdf_value: URDF joint position (-0.01 to 0.0225)

        Returns:
            Hardware gripper command (0.0 to 0.065)
        """
        # Linear mapping: urdf_min -> hw_min, urdf_max -> hw_max
        # normalized = (urdf_value - URDF_MIN) / URDF_RANGE  # 0 to 1
        # hw_value = HW_MIN + normalized * HW_RANGE
        normalized = (urdf_value - cls.URDF_MIN) / cls.URDF_RANGE
        hw_value = cls.HW_MIN + normalized * cls.HW_RANGE
        return float(np.clip(hw_value, cls.HW_MIN, cls.HW_MAX))

    @classmethod
    def hardware_to_urdf(cls, hw_value: float) -> float:
        """Convert hardware command to URDF joint value.

        Args:
            hw_value: Hardware gripper command (0.0 to 0.065)

        Returns:
            URDF joint position (-0.01 to 0.0225)
        """
        # Linear mapping: hw_min -> urdf_min, hw_max -> urdf_max
        normalized = (hw_value - cls.HW_MIN) / cls.HW_RANGE
        urdf_value = cls.URDF_MIN + normalized * cls.URDF_RANGE
        return float(np.clip(urdf_value, cls.URDF_MIN, cls.URDF_MAX))

    @classmethod
    def trigger_to_urdf(cls, trigger: float) -> float:
        """Convert VR trigger value to URDF joint value.

        Args:
            trigger: VR controller trigger (0.0=released, 1.0=pressed)

        Returns:
            URDF joint position (-0.01 to 0.0225)
        """
        # Trigger 0.0 -> open (URDF_MIN), Trigger 1.0 -> closed (URDF_MAX)
        return cls.URDF_MIN + trigger * cls.URDF_RANGE

    @classmethod
    def trigger_to_hardware(cls, trigger: float) -> float:
        """Convert VR trigger value to hardware command.

        The Arduino/hardware uses inverted convention where:
        - HW_MAX (0.065) = gripper OPEN
        - HW_MIN (0.0) = gripper CLOSED

        So we invert the mapping to get intuitive behavior:
        - Trigger released (0.0) → send HW_MAX → gripper opens
        - Trigger pressed (1.0) → send HW_MIN → gripper closes

        Args:
            trigger: VR controller trigger (0.0=released, 1.0=pressed)

        Returns:
            Hardware gripper command (0.0 to 0.065)
        """
        return cls.HW_MAX - trigger * cls.HW_RANGE

from gr00t_wbc.control.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)


class ElbowPose(Enum):
    """Enum for elbow pose configuration."""

    LOW = "low"
    HIGH = "high"


@dataclass
class G1AlohaSupplementalInfo(RobotSupplementalInfo):
    """
    Supplemental information for the G1 robot with ALOHA grippers.

    This is for G1 29DOF body + 2 DOF ALOHA grippers (31 DOF total).

    Body DOF breakdown (29 total):
    - Left leg: 6 DOF
    - Right leg: 6 DOF
    - Waist: 3 DOF
    - Left arm: 7 DOF
    - Right arm: 7 DOF

    Hand DOF (2 total):
    - Left ALOHA gripper: 1 DOF (prismatic)
    - Right ALOHA gripper: 1 DOF (prismatic)

    Waist joints stay in the lower-body group and never participate in
    upper-body IK.

    Args:
        elbow_pose: Which elbow pose configuration to use for default joint positions
    """

    def __init__(
        self,
        elbow_pose: ElbowPose = ElbowPose.LOW,
    ):
        name = "G1_AlohaGripper"

        # Define all actuated joints (29 body DOF)
        body_actuated_joints = [
            # Left leg (6 DOF)
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right leg (6 DOF)
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Waist (3 DOF)
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Left arm (7 DOF)
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right arm (7 DOF)
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # ALOHA gripper joints (1 DOF each, prismatic)
        left_hand_actuated_joints = [
            "left_gripper_joint",
        ]

        right_hand_actuated_joints = [
            "right_gripper_joint",
        ]

        # Define joint limits from URDF
        joint_limits = {
            # Left leg
            "left_hip_pitch_joint": [-2.5307, 2.8798],
            "left_hip_roll_joint": [-0.5236, 2.9671],
            "left_hip_yaw_joint": [-2.7576, 2.7576],
            "left_knee_joint": [-0.087267, 2.8798],
            "left_ankle_pitch_joint": [-0.87267, 0.5236],
            "left_ankle_roll_joint": [-0.2618, 0.2618],
            # Right leg
            "right_hip_pitch_joint": [-2.5307, 2.8798],
            "right_hip_roll_joint": [-2.9671, 0.5236],
            "right_hip_yaw_joint": [-2.7576, 2.7576],
            "right_knee_joint": [-0.087267, 2.8798],
            "right_ankle_pitch_joint": [-0.87267, 0.5236],
            "right_ankle_roll_joint": [-0.2618, 0.2618],
            # Waist
            "waist_yaw_joint": [-2.618, 2.618],
            "waist_roll_joint": [-0.52, 0.52],
            "waist_pitch_joint": [-0.52, 0.52],
            # Left arm
            "left_shoulder_pitch_joint": [-3.0892, 2.6704],
            "left_shoulder_roll_joint": [-1.5882, 2.2515],
            "left_shoulder_yaw_joint": [-2.618, 2.618],
            "left_elbow_joint": [-1.0472, 2.0944],
            "left_wrist_roll_joint": [-1.972222054, 1.972222054],
            "left_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "left_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # Right arm
            "right_shoulder_pitch_joint": [-3.0892, 2.6704],
            "right_shoulder_roll_joint": [-2.2515, 1.5882],
            "right_shoulder_yaw_joint": [-2.618, 2.618],
            "right_elbow_joint": [-1.0472, 2.0944],
            "right_wrist_roll_joint": [-1.972222054, 1.972222054],
            "right_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "right_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # ALOHA grippers (prismatic, in meters) - Hardware convention
            # Hardware Range: 0.0 (fully open) to 0.065 (fully closed)
            # Note: This differs from URDF (-0.01 to 0.0225) but matches training data
            # Use AlohaGripperMapping if conversion to/from URDF is needed
            "left_gripper_joint": [0.0, 0.065],
            "right_gripper_joint": [0.0, 0.065],
        }

        # Define joint groups
        joint_groups = {
            # Body groups
            "waist": {
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "groups": [],
            },
            # Leg groups
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            # Arm groups
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            # Gripper groups (ALOHA has 1 DOF per gripper)
            "left_gripper": {
                "joints": ["left_gripper_joint"],
                "groups": [],
            },
            "right_gripper": {
                "joints": ["right_gripper_joint"],
                "groups": [],
            },
            "grippers": {"joints": [], "groups": ["left_gripper", "right_gripper"]},
            # Aliases for backward compatibility with code that uses "left_hand"/"right_hand"
            # These point to the same gripper joints
            "left_hand": {
                "joints": ["left_gripper_joint"],
                "groups": [],
            },
            "right_hand": {
                "joints": ["right_gripper_joint"],
                "groups": [],
            },
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            # Full body groups
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "grippers"]},
        }

        # Define joint name mapping from generic types to robot-specific names
        joint_name_mapping = {
            # Waist joints
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            # Shoulder joints
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            # Elbow joints
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            # Wrist joints
            "wrist_pitch": {"left": "left_wrist_pitch_joint", "right": "right_wrist_pitch_joint"},
            "wrist_roll": {"left": "left_wrist_roll_joint", "right": "right_wrist_roll_joint"},
            "wrist_yaw": {"left": "left_wrist_yaw_joint", "right": "right_wrist_yaw_joint"},
            # Gripper joints
            "gripper": {"left": "left_gripper_joint", "right": "right_gripper_joint"},
        }

        root_frame_name = "pelvis"

        # Hand frame names - use wrist frames (same as regular G1) for consistent IK behavior
        # The gripper is controlled separately via hand IK solver
        # Note: Using aloha_base_link would require a different hand_rotation_correction
        # due to the fixed joint rotation (rpy="-90° 0° 90°") in the URDF
        hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

        elbow_calibration_joint_angles = {"left": 0.0, "right": 0.0}

        # Same rotation correction as regular G1 since we use the same wrist frames
        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # Configure default joint positions based on elbow pose
        # Gripper semantic convention (from AlohaGripperMapping):
        #   0.0 = fully OPEN (fingers spread apart)
        #   0.065 = fully CLOSED (fingers together)
        # Set to 0.0 for open grippers at initial pose
        if elbow_pose == ElbowPose.HIGH:
            default_joint_q = {
                "shoulder_roll": {"left": 0.5, "right": -0.5},
                "shoulder_pitch": {"left": -0.2, "right": -0.2},
                "shoulder_yaw": {"left": -0.5, "right": 0.5},
                "wrist_roll": {"left": -0.5, "right": 0.5},
                "wrist_yaw": {"left": 0.5, "right": -0.5},
                "wrist_pitch": {"left": -0.2, "right": -0.2},
                "gripper": {"left": 0.0, "right": 0.0},  # Open grippers (0.0 = open)
            }
        else:  # ElbowPose.LOW
            default_joint_q = {
                "shoulder_roll": {"left": 0.2, "right": -0.2},
                "gripper": {"left": 0.0, "right": 0.0},  # Open grippers (0.0 = open)
            }
        
        ### HERE

        teleop_upper_body_motion_scale = 0.8

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            elbow_calibration_joint_angles=elbow_calibration_joint_angles,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )
