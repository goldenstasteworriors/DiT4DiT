#!/usr/bin/env python3
"""Create four action-only representations for grab-red-bottle train/test data."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(22, 29)
LEFT_HAND = slice(29, 35)
RIGHT_HAND = slice(35, 41)
LEFT_WRIST_POS = slice(0, 3)
LEFT_WRIST_QUAT = slice(3, 7)
RIGHT_WRIST_POS = slice(7, 10)
RIGHT_WRIST_QUAT = slice(10, 14)
PASSTHROUGH = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
REPRESENTATIONS = (
    "right_joints",
    "right_wrist_delta",
    "right_target_joints",
    "bimanual_wrist_delta",
)
DATASET_NAMES = {
    "right_joints": "pick_up_pipette_right_joints",
    "right_wrist_delta": "pick_up_pipette_right_wrist_delta",
    "right_target_joints": "pick_up_pipette_right_target_joints",
    "bimanual_wrist_delta": "pick_up_pipette_bimanual_wrist_delta",
}


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def wrist_delta(position: np.ndarray, quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = quat_wxyz_to_matrix(quaternion)
    next_position = np.concatenate([position[1:], position[-1:]], axis=0)
    next_rotation = np.concatenate([rotation[1:], rotation[-1:]], axis=0)
    delta_position = np.einsum(
        "nij,nj->ni", np.transpose(rotation, (0, 2, 1)), next_position - position
    )
    delta_rotation = np.einsum(
        "nij,njk->nik", np.transpose(rotation, (0, 2, 1)), next_rotation
    )
    return delta_position, delta_rotation[:, :2, :].reshape(-1, 6)


def wrist_fields(
    eef: np.ndarray,
    observed_joints: np.ndarray,
    target_joints: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if side == "left":
        position = eef[:, LEFT_WRIST_POS]
        quaternion = eef[:, LEFT_WRIST_QUAT]
        hand_slice = LEFT_HAND
    elif side == "right":
        position = eef[:, RIGHT_WRIST_POS]
        quaternion = eef[:, RIGHT_WRIST_QUAT]
        hand_slice = RIGHT_HAND
    else:
        raise ValueError(f"Unsupported side: {side}")
    delta_position, delta_rotation_6d = wrist_delta(position, quaternion)
    state = np.concatenate([position, quaternion, observed_joints[:, hand_slice]], axis=-1)
    action = np.concatenate(
        [delta_position, delta_rotation_6d, target_joints[:, hand_slice]], axis=-1
    )
    return state.astype(np.float32), action.astype(np.float32)


def derive_episode(frame: pd.DataFrame, representation: str) -> pd.DataFrame:
    observed = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    target = np.stack(frame["action.wbc"].to_numpy()).astype(np.float32)
    eef = np.stack(frame["observation.eef_state"].to_numpy()).astype(np.float32)
    result = frame[PASSTHROUGH].copy()

    if representation == "right_joints":
        result["observation.right_arm_hand"] = list(
            np.concatenate([observed[:, RIGHT_ARM], observed[:, RIGHT_HAND]], axis=-1)
        )
        result["action.right_arm_hand"] = list(
            np.concatenate([target[:, RIGHT_ARM], target[:, RIGHT_HAND]], axis=-1)
        )
    elif representation == "right_target_joints":
        right_target = np.concatenate([target[:, RIGHT_ARM], target[:, RIGHT_HAND]], axis=-1)
        result["observation.right_arm_hand_target"] = list(right_target)
        result["action.right_arm_hand"] = list(right_target)
    elif representation == "right_wrist_delta":
        state, action = wrist_fields(eef, observed, target, "right")
        result["observation.right_wrist_hand"] = list(state)
        result["action.right_wrist_delta_hand"] = list(action)
    elif representation == "bimanual_wrist_delta":
        left_state, left_action = wrist_fields(eef, observed, target, "left")
        right_state, right_action = wrist_fields(eef, observed, target, "right")
        result["observation.bimanual_wrist_hand"] = list(
            np.concatenate([left_state, right_state], axis=-1)
        )
        result["action.bimanual_wrist_delta_hand"] = list(
            np.concatenate([left_action, right_action], axis=-1)
        )
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return result


def modality(representation: str) -> dict:
    common = {
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    if representation == "right_joints":
        common["state"] = {
            "right_arm": {"start": 0, "end": 7, "original_key": "observation.right_arm_hand"},
            "right_hand": {"start": 7, "end": 13, "original_key": "observation.right_arm_hand"},
        }
        common["action"] = {
            "right_arm": {"start": 0, "end": 7, "original_key": "action.right_arm_hand"},
            "right_hand": {"start": 7, "end": 13, "original_key": "action.right_arm_hand"},
        }
    elif representation == "right_target_joints":
        common["state"] = {
            "right_arm_target": {
                "start": 0,
                "end": 7,
                "original_key": "observation.right_arm_hand_target",
            },
            "right_hand_target": {
                "start": 7,
                "end": 13,
                "original_key": "observation.right_arm_hand_target",
            },
        }
        common["action"] = {
            "right_arm": {"start": 0, "end": 7, "original_key": "action.right_arm_hand"},
            "right_hand": {"start": 7, "end": 13, "original_key": "action.right_arm_hand"},
        }
    elif representation == "right_wrist_delta":
        common["state"] = wrist_state_modality("right", "observation.right_wrist_hand", 0)
        common["action"] = wrist_action_modality("right", "action.right_wrist_delta_hand", 0)
    elif representation == "bimanual_wrist_delta":
        state_key = "observation.bimanual_wrist_hand"
        action_key = "action.bimanual_wrist_delta_hand"
        common["state"] = {
            **wrist_state_modality("left", state_key, 0),
            **wrist_state_modality("right", state_key, 13),
        }
        common["action"] = {
            **wrist_action_modality("left", action_key, 0),
            **wrist_action_modality("right", action_key, 15),
        }
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return common


def wrist_state_modality(side: str, original_key: str, offset: int) -> dict:
    return {
        f"{side}_wrist_pos": {"start": offset, "end": offset + 3, "original_key": original_key},
        f"{side}_wrist_abs_quat": {
            "start": offset + 3,
            "end": offset + 7,
            "original_key": original_key,
            "rotation_type": "quaternion",
        },
        f"{side}_hand": {"start": offset + 7, "end": offset + 13, "original_key": original_key},
    }


def wrist_action_modality(side: str, original_key: str, offset: int) -> dict:
    return {
        f"{side}_wrist_delta_pos": {
            "start": offset,
            "end": offset + 3,
            "original_key": original_key,
            "absolute": False,
        },
        f"{side}_wrist_delta_rot_6d": {
            "start": offset + 3,
            "end": offset + 9,
            "original_key": original_key,
            "rotation_type": "rotation_6d",
            "absolute": False,
        },
        f"{side}_hand": {
            "start": offset + 9,
            "end": offset + 15,
            "original_key": original_key,
        },
    }


def prepare(source: Path, destination: Path, representation: str) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    (destination / "data" / "chunk-000").mkdir(parents=True)
    shutil.copytree(source / "videos", destination / "videos")
    shutil.copytree(source / "meta", destination / "meta")
    for parquet in sorted((source / "data" / "chunk-000").glob("*.parquet")):
        derive_episode(pd.read_parquet(parquet), representation).to_parquet(
            destination / "data" / "chunk-000" / parquet.name,
            index=False,
        )
    (destination / "meta" / "modality.json").write_text(
        json.dumps(modality(representation), indent=4) + "\n"
    )
    for cache_name in ("stats.json", "stats_gr00t.json", "steps_data_index.pkl"):
        cache_path = destination / "meta" / cache_name
        if cache_path.exists():
            cache_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for split, source in (("train", args.train_source), ("test", args.test_source)):
        for representation in REPRESENTATIONS:
            prepare(source, args.output_root / split / DATASET_NAMES[representation], representation)


if __name__ == "__main__":
    main()
