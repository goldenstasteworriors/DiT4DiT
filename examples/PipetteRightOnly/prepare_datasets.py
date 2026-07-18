#!/usr/bin/env python3
"""Create right-only LeRobot datasets for the two pipette action spaces."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RIGHT_ARM = slice(22, 29)
RIGHT_HAND = slice(35, 41)
RIGHT_WRIST_POS = slice(7, 10)
RIGHT_WRIST_QUAT = slice(10, 14)
PASSTHROUGH = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(1e-12)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def derive_episode(frame: pd.DataFrame, representation: str) -> pd.DataFrame:
    obs = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    wbc = np.stack(frame["action.wbc"].to_numpy()).astype(np.float32)
    eef = np.stack(frame["observation.eef_state"].to_numpy()).astype(np.float32)
    result = frame[PASSTHROUGH].copy()

    if representation == "joints":
        result["observation.right_arm_hand"] = list(np.concatenate([obs[:, RIGHT_ARM], obs[:, RIGHT_HAND]], -1))
        result["action.right_arm_hand"] = list(np.concatenate([wbc[:, RIGHT_ARM], wbc[:, RIGHT_HAND]], -1))
        return result

    pos = eef[:, RIGHT_WRIST_POS]
    quat = eef[:, RIGHT_WRIST_QUAT]
    rot = quat_wxyz_to_matrix(quat)

    # State uses the motion that arrived at the current frame:
    # T_{t-1}^{-1} T_t, expressed in the previous wrist frame.  This avoids
    # leaking the robot/table placement through an absolute wrist pose and
    # matches EgoHumanoid's local-frame delta EEF representation.
    prev_pos = np.concatenate([pos[:1], pos[:-1]], axis=0)
    prev_rot = np.concatenate([rot[:1], rot[:-1]], axis=0)
    state_delta_pos = np.einsum(
        "nij,nj->ni", np.transpose(prev_rot, (0, 2, 1)), pos - prev_pos
    )
    state_delta_rot = np.einsum(
        "nij,njk->nik", np.transpose(prev_rot, (0, 2, 1)), rot
    )
    state_delta_rot_6d = state_delta_rot[:, :2, :].reshape(-1, 6)

    next_pos = np.concatenate([pos[1:], pos[-1:]], axis=0)
    next_rot = np.concatenate([rot[1:], rot[-1:]], axis=0)
    # T_current^-1 T_next: translation and rotation are expressed in the current wrist frame.
    delta_pos = np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), next_pos - pos)
    delta_rot = np.einsum("nij,njk->nik", np.transpose(rot, (0, 2, 1)), next_rot)
    delta_rot_6d = delta_rot[:, :2, :].reshape(-1, 6)
    result["observation.right_wrist_delta_hand"] = list(
        np.concatenate([state_delta_pos, state_delta_rot_6d, obs[:, RIGHT_HAND]], -1).astype(np.float32)
    )
    result["action.right_wrist_delta_hand"] = list(
        np.concatenate([delta_pos, delta_rot_6d, wbc[:, RIGHT_HAND]], -1).astype(np.float32)
    )
    return result


def modality(representation: str) -> dict:
    common = {
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    if representation == "joints":
        common["state"] = {
            "right_arm": {"start": 0, "end": 7, "original_key": "observation.right_arm_hand"},
            "right_hand": {"start": 7, "end": 13, "original_key": "observation.right_arm_hand"},
        }
        common["action"] = {
            "right_arm": {"start": 0, "end": 7, "original_key": "action.right_arm_hand"},
            "right_hand": {"start": 7, "end": 13, "original_key": "action.right_arm_hand"},
        }
    else:
        common["state"] = {
            "right_wrist_delta_pos": {
                "start": 0, "end": 3, "original_key": "observation.right_wrist_delta_hand", "absolute": False
            },
            "right_wrist_delta_rot_6d": {
                "start": 3, "end": 9, "original_key": "observation.right_wrist_delta_hand",
                "rotation_type": "rotation_6d", "absolute": False,
            },
            "right_hand": {"start": 9, "end": 15, "original_key": "observation.right_wrist_delta_hand"},
        }
        common["action"] = {
            "right_wrist_delta_pos": {
                "start": 0, "end": 3, "original_key": "action.right_wrist_delta_hand", "absolute": False
            },
            "right_wrist_delta_rot_6d": {
                "start": 3, "end": 9, "original_key": "action.right_wrist_delta_hand",
                "rotation_type": "rotation_6d", "absolute": False,
            },
            "right_hand": {"start": 9, "end": 15, "original_key": "action.right_wrist_delta_hand"},
        }
    return common


def prepare(source: Path, destination: Path, representation: str) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    (destination / "data" / "chunk-000").mkdir(parents=True)
    shutil.copytree(source / "videos", destination / "videos")
    shutil.copytree(source / "meta", destination / "meta")
    for parquet in sorted((source / "data" / "chunk-000").glob("*.parquet")):
        derive_episode(pd.read_parquet(parquet), representation).to_parquet(
            destination / "data" / "chunk-000" / parquet.name, index=False
        )
    (destination / "meta" / "modality.json").write_text(json.dumps(modality(representation), indent=4) + "\n")
    # Force statistics to be recomputed from the right-only columns on first load.
    stats = destination / "meta" / "stats.json"
    if stats.exists():
        stats.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output_root / "pick_up_pipette_right_joints", "joints")
    prepare(args.source, args.output_root / "pick_up_pipette_right_wrist_delta", "wrist_delta")


if __name__ == "__main__":
    main()
