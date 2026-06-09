from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH


def _get_hand_joint_group_name(robot_model: RobotModel) -> str:
    """
    Determine the hand/gripper joint group naming based on robot model.

    Returns:
        "hand" for three-finger hand robots (G1 43 DOF)
        "gripper" for ALOHA gripper robots (G1 31 DOF)
    """
    # Check if robot has "gripper" joint groups (ALOHA) or "hand" joint groups (three-finger)
    try:
        robot_model.get_joint_group_indices("left_gripper")
        return "gripper"
    except (KeyError, ValueError):
        return "hand"


def _get_compact_joint_layout(robot_model: RobotModel, group_names: list = None) -> dict:
    """
    Compute the compact joint layout for state/action arrays.

    The compact array contains the joints in `group_names` (sorted by original index),
    followed by extra fields (rpy, height, etc.).

    If `group_names` is None, defaults to: left_arm, right_arm, left_gripper/hand,
    right_gripper/hand.

    Returns dict with keys: indices (sorted original indices), n_joints,
        and per-group start/end in the compact array.
    """
    hand_group_name = _get_hand_joint_group_name(robot_model)
    left_hand_key = f"left_{hand_group_name}"
    right_hand_key = f"right_{hand_group_name}"

    if group_names is None:
        group_names = ["left_arm", "right_arm", left_hand_key, right_hand_key]

    # Collect indices per group
    groups = {name: sorted(robot_model.get_joint_group_indices(name)) for name in group_names}

    # All selected indices sorted (preserves original joint ordering)
    all_indices = sorted(sum(groups.values(), []))
    # Map original index -> position in compact array
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    layout = {
        "indices": all_indices,
        "n_joints": len(all_indices),
        "hand_group_name": hand_group_name,
        "groups": {},
    }
    for name, orig_indices in groups.items():
        compact_positions = [idx_to_pos[i] for i in orig_indices]
        layout["groups"][name] = {"start": compact_positions[0], "end": compact_positions[-1] + 1}

    return layout


def get_modality_config(robot_model: RobotModel, add_stereo_camera: bool = False) -> dict:
    """
    Get the modality config for the robot model.

    Automatically handles both three-finger hand (G1 43 DOF) and ALOHA gripper (G1 31 DOF)
    configurations based on the robot model's joint groups.
    """
    hand_group_name = _get_hand_joint_group_name(robot_model)
    left_hand_key = f"left_{hand_group_name}"
    right_hand_key = f"right_{hand_group_name}"

    # State and action both include legs and waist in addition to arms and grippers/hands
    full_layout = _get_compact_joint_layout(
        robot_model,
        ["left_leg", "right_leg", "waist", "left_arm", "right_arm", left_hand_key, right_hand_key],
    )
    n = full_layout["n_joints"]

    full_joint_modality = {
        "left_leg": full_layout["groups"]["left_leg"],
        "right_leg": full_layout["groups"]["right_leg"],
        "waist": full_layout["groups"]["waist"],
        "left_arm": full_layout["groups"]["left_arm"],
        "right_arm": full_layout["groups"]["right_arm"],
        left_hand_key: full_layout["groups"][left_hand_key],
        right_hand_key: full_layout["groups"][right_hand_key],
    }

    modality_config = {
        "state": {
            **full_joint_modality,
            "rpy": {"start": n, "end": n + 3},
            "height": {"start": n + 3, "end": n + 4},
        },
        "action": {
            **full_joint_modality,
            "rpy": {"start": n, "end": n + 3},
            "height": {"start": n + 3, "end": n + 4},
            "torso_vx": {"start": n + 4, "end": n + 5},
            "torso_vy": {"start": n + 5, "end": n + 6},
            "torso_vyaw": {"start": n + 6, "end": n + 7},
            "target_yaw": {"start": n + 7, "end": n + 8},
            "left_wrist_pos": {"start": 0, "end": 3, "original_key": "action.eef"},
            "left_wrist_abs_quat": {
                "start": 3,
                "end": 7,
                "original_key": "action.eef",
                "rotation_type": "quaternion",
            },
            "right_wrist_pos": {"start": 7, "end": 10, "original_key": "action.eef"},
            "right_wrist_abs_quat": {
                "start": 10,
                "end": 14,
                "original_key": "action.eef",
                "rotation_type": "quaternion",
            },
        },
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    if add_stereo_camera:
        modality_config["video"].update(
            {
                "ego_view_left_mono": {"original_key": "observation.images.ego_view_left_mono"},
                "ego_view_right_mono": {"original_key": "observation.images.ego_view_right_mono"},
            }
        )

    return modality_config


def get_dataset_features(robot_model: RobotModel, add_stereo_camera: bool = False) -> dict:
    """
    Get the dataset features for the robot model.
    """
    hand_group_name = _get_hand_joint_group_name(robot_model)
    left_hand_key = f"left_{hand_group_name}"
    right_hand_key = f"right_{hand_group_name}"

    full_layout = _get_compact_joint_layout(
        robot_model,
        ["left_leg", "right_leg", "waist", "left_arm", "right_arm", left_hand_key, right_hand_key],
    )
    n = full_layout["n_joints"]
    full_joint_names = [robot_model.joint_names[i] for i in full_layout["indices"]]

    dataset_features = {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (n + 4,),
            "names": full_joint_names + ["roll", "pitch", "yaw", "height"],
        },
        "action": {
            "dtype": "float64",
            "shape": (n + 8,),
            "names": full_joint_names + ["roll", "pitch", "yaw", "height", "torso_vx", "torso_vy", "torso_vyaw", "target_yaw"],
        },
        "action.eef": {
            "dtype": "float64",
            "shape": (14,),
            "names": [
                "left_wrist_x", "left_wrist_y", "left_wrist_z",
                "left_wrist_qw", "left_wrist_qx", "left_wrist_qy", "left_wrist_qz",
                "right_wrist_x", "right_wrist_y", "right_wrist_z",
                "right_wrist_qw", "right_wrist_qx", "right_wrist_qy", "right_wrist_qz",
            ],
        },
        "observation.img_state_delta": {
            "dtype": "float32",
            "shape": (1,),
            "names": "img_state_delta",
        },
    }
    if add_stereo_camera:
        dataset_features.update(
            {
                "observation.images.ego_view_left_mono": {
                    "dtype": "video",
                    "shape": [RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH, 3],
                    "names": ["height", "width", "channel"],
                },
                "observation.images.ego_view_right_mono": {
                    "dtype": "video",
                    "shape": [RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH, 3],
                    "names": ["height", "width", "channel"],
                },
            }
        )

    return dataset_features
