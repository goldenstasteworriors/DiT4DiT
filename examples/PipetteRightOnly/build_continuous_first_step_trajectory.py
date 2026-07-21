"""Integrate one predicted wrist delta per frame without ground-truth wrist resets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from examples.PipetteRightOnly.convert_wrist_delta_to_joint_chunks import (
    RightArmIK,
    default_model_path,
    quat_wxyz_to_matrix,
    rotation_6d_to_matrix,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--position-tolerance", type=float, default=2e-5)
    parser.add_argument("--orientation-tolerance", type=float, default=2e-4)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = args.input.expanduser().resolve()
    source_root = args.source_dataset.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()

    matches = sorted(source_root.glob(f"data/chunk-*/episode_{args.episode:06d}.parquet"))
    if len(matches) != 1:
        raise SystemExit(f"expected one source parquet, found {len(matches)} under {source_root}")
    source_path = matches[0]
    source = pd.read_parquet(source_path)
    source_states = np.stack(source["observation.state"]).astype(np.float64)
    source_actions = np.stack(source["action.wbc"]).astype(np.float64)
    source_eef = np.stack(source["observation.eef_state"]).astype(np.float64)
    source_timestamps = source["timestamp"].to_numpy(dtype=np.float64)

    with np.load(input_path) as raw:
        timestamps = np.asarray(raw["timestamps"], dtype=np.float64)
        wrist_chunks = np.asarray(raw["predicted_wrist_delta_chunks"], dtype=np.float64)
        latencies = np.asarray(raw["latencies_seconds"], dtype=np.float64)

    episode_length = len(source)
    if timestamps.shape != (episode_length,) or wrist_chunks.shape != (episode_length, 16, 15):
        raise SystemExit(
            f"complete episode required: timestamps={timestamps.shape}, wrist_chunks={wrist_chunks.shape}"
        )
    if not np.allclose(timestamps, source_timestamps, atol=1e-9, rtol=0.0):
        raise SystemExit("inference/source timestamps do not match")

    ik = RightArmIK(model_path, args)
    predicted_actions = np.empty((episode_length, 13), dtype=np.float32)
    predicted_wrist_positions = np.empty((episode_length, 3), dtype=np.float64)
    predicted_wrist_rotations = np.empty((episode_length, 3, 3), dtype=np.float64)
    position_residuals = np.empty(episode_length, dtype=np.float64)
    orientation_residuals = np.empty(episode_length, dtype=np.float64)
    iteration_counts = np.empty(episode_length, dtype=np.int32)

    # The only ground-truth wrist anchor: episode frame 0.
    position = source_eef[0, 7:10].copy()
    rotation = quat_wxyz_to_matrix(source_eef[0, 10:14])
    previous_joints = source_states[0, 22:29].copy()

    for timestep in range(episode_length):
        wrist_action = wrist_chunks[timestep, 0]
        position = position + rotation @ wrist_action[:3]
        rotation = rotation @ rotation_6d_to_matrix(wrist_action[3:9])

        # Update the recorded non-arm body/pelvis state, but never reset the
        # right arm to its ground-truth trajectory after frame 0.
        ik.set_source_state(source_states[timestep])
        ik.data.qpos[ik.arm_qpos] = previous_joints
        mujoco.mj_forward(ik.model, ik.data)
        pelvis_position, pelvis_rotation = ik.pelvis_pose()
        target_position = pelvis_position + pelvis_rotation @ position
        target_rotation = pelvis_rotation @ rotation
        joints, pos_residual, rot_residual, iterations = ik.solve(target_position, target_rotation)

        predicted_actions[timestep, :7] = joints
        predicted_actions[timestep, 7:13] = np.clip(wrist_action[9:15], 0.0, 1.0)
        predicted_wrist_positions[timestep] = position
        predicted_wrist_rotations[timestep] = rotation
        position_residuals[timestep] = pos_residual
        orientation_residuals[timestep] = rot_residual
        iteration_counts[timestep] = iterations
        previous_joints = joints

    input_states = np.concatenate([source_states[:, 22:29], source_states[:, 35:41]], axis=1)
    ground_truth_actions = np.concatenate(
        [source_actions[:, 22:29], source_actions[:, 35:41]], axis=1
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamps=timestamps,
        frame_indices=source["frame_index"].to_numpy(dtype=np.int64),
        input_states=input_states.astype(np.float32),
        ground_truth_actions=ground_truth_actions.astype(np.float32),
        continuous_predicted_actions=predicted_actions,
        first_predicted_actions=predicted_actions,
        continuous_wrist_positions=predicted_wrist_positions,
        continuous_wrist_rotations=predicted_wrist_rotations,
        first_wrist_delta_actions=wrist_chunks[:, 0].astype(np.float32),
        latencies_seconds=latencies,
        ik_position_residuals=position_residuals,
        ik_orientation_residuals=orientation_residuals,
        ik_iteration_counts=iteration_counts,
    )

    arm_mae = float(np.mean(np.abs(predicted_actions[:, :7] - ground_truth_actions[:, :7])))
    summary = {
        "source": "continuous first-step wrist-delta integration",
        "episode": args.episode,
        "frames": episode_length,
        "fps": float(1.0 / np.median(np.diff(timestamps))),
        "input": str(input_path),
        "source_parquet": str(source_path),
        "model": str(model_path),
        "ground_truth_wrist_resets": 1,
        "right_arm_mae_rad": arm_mae,
        "final_wrist_position_error_m": float(
            np.linalg.norm(predicted_wrist_positions[-1] - source_eef[-1, 7:10])
        ),
        "ik_position_residual_max_m": float(position_residuals.max()),
        "ik_orientation_residual_max_rad": float(orientation_residuals.max()),
        "npz": output.name,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
