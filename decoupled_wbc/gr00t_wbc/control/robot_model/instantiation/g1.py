import os
from pathlib import Path

from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    G1SupplementalInfo,
)
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_aloha_supplemental_info import (
    ElbowPose as AlohaElbowPose,
    G1AlohaSupplementalInfo,
)


def instantiate_g1_robot_model(high_elbow_pose: bool = False):
    """
    Instantiate a G1 robot model. Waist joints are kept in the lower body group
    (controlled by the lower-body policy) and never participate in upper-body IK.

    Args:
        high_elbow_pose: Whether to use high elbow pose configuration for default joint positions

    Returns:
        RobotModel: Configured G1 robot model
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    robot_model_config = {
        "asset_path": os.path.join(project_root, "gr00t_wbc/control/robot_model/model_data/g1"),
        "urdf_path": os.path.join(
            project_root, "gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
        ),
    }

    elbow_pose_enum = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW

    robot_model_supplemental_info = G1SupplementalInfo(elbow_pose=elbow_pose_enum)

    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model


def instantiate_g1_aloha_robot_model(high_elbow_pose: bool = False):
    """
    Instantiate a G1 robot model with ALOHA grippers (31 DOF total).

    This is for the G1 29DOF body + 2 DOF ALOHA grippers configuration.
    Waist joints are kept in the lower body group and do not participate in
    upper-body IK.

    Body DOF breakdown (29 total):
    - Left leg: 6 DOF
    - Right leg: 6 DOF
    - Waist: 3 DOF
    - Left arm: 7 DOF
    - Right arm: 7 DOF

    Hand DOF (2 total):
    - Left ALOHA gripper: 1 DOF (prismatic)
    - Right ALOHA gripper: 1 DOF (prismatic)

    Args:
        high_elbow_pose: Whether to use high elbow pose configuration for default joint positions

    Returns:
        RobotModel: Configured G1 robot model with ALOHA grippers
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    robot_model_config = {
        "asset_path": os.path.join(project_root, "gr00t_wbc/control/robot_model/model_data/g1"),
        # Use reduced URDF without mimic joints for WBC control (31 actuated DOF)
        # MuJoCo simulation uses full XML with mimic joints via equality constraints
        "urdf_path": os.path.join(
            project_root, "gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_aloha_reduced.urdf"
        ),
    }

    elbow_pose_enum = AlohaElbowPose.HIGH if high_elbow_pose else AlohaElbowPose.LOW

    robot_model_supplemental_info = G1AlohaSupplementalInfo(elbow_pose=elbow_pose_enum)

    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
