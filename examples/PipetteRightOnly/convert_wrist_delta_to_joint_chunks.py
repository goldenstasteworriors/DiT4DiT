"""Convert full wrist-delta predictions to G1 right-arm joint chunks with MuJoCo IK."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
BODY_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", *RIGHT_ARM_NAMES,
]


def default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml"
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
    parser.add_argument("--max-position-residual", type=float, default=0.01)
    parser.add_argument("--max-orientation-residual", type=float, default=0.1)
    return parser


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    first = np.asarray(rotation_6d[:3], dtype=np.float64)
    second = np.asarray(rotation_6d[3:6], dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second -= float(np.dot(first, second)) * first
    if np.linalg.norm(second) < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
    second /= max(float(np.linalg.norm(second)), 1e-12)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=0)


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    matrix = target @ current.T
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(float(cosine))
    skew = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )
    if angle < 1e-8:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * skew


class RightArmIK:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.args = args
        self.wrist_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        self.pelvis_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_qpos = np.array([self._qpos_address(name) for name in BODY_NAMES])
        self.arm_qpos = np.array([self._qpos_address(name) for name in RIGHT_ARM_NAMES])
        self.arm_dofs = np.array([self._dof_address(name) for name in RIGHT_ARM_NAMES])
        joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM_NAMES]
        )
        self.lower = self.model.jnt_range[joint_ids, 0] + 1e-6
        self.upper = self.model.jnt_range[joint_ids, 1] - 1e-6

    def _qpos_address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"joint missing from model: {name}")
        return int(self.model.jnt_qposadr[joint])

    def _dof_address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[joint])

    def set_source_state(self, body_state: np.ndarray) -> None:
        self.data.qpos[self.body_qpos] = body_state[:29]
        mujoco.mj_forward(self.model, self.data)

    def pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.xpos[self.pelvis_body].copy(),
            self.data.xmat[self.pelvis_body].reshape(3, 3).copy(),
        )

    def wrist_pose_in_pelvis(self) -> tuple[np.ndarray, np.ndarray]:
        pelvis_position, pelvis_rotation = self.pelvis_pose()
        wrist_position = self.data.xpos[self.wrist_body]
        wrist_rotation = self.data.xmat[self.wrist_body].reshape(3, 3)
        return (
            pelvis_rotation.T @ (wrist_position - pelvis_position),
            pelvis_rotation.T @ wrist_rotation,
        )

    def solve(self, target_position: np.ndarray, target_rotation: np.ndarray) -> tuple[np.ndarray, float, float, int]:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        iterations = 0
        for iterations in range(1, self.args.max_iterations + 1):
            current_position = self.data.xpos[self.wrist_body]
            current_rotation = self.data.xmat[self.wrist_body].reshape(3, 3)
            position_error = target_position - current_position
            orientation_error = rotation_error(target_rotation, current_rotation)
            if (
                np.linalg.norm(position_error) <= self.args.position_tolerance
                and np.linalg.norm(orientation_error) <= self.args.orientation_tolerance
            ):
                break
            mujoco.mj_jacBody(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.wrist_body,
            )
            jacobian = np.vstack(
                [jacobian_position[:, self.arm_dofs], jacobian_rotation[:, self.arm_dofs]]
            )
            error = np.concatenate([position_error, orientation_error])
            system = jacobian @ jacobian.T + self.args.damping * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)
            peak = float(np.max(np.abs(delta)))
            if peak > self.args.max_step:
                delta *= self.args.max_step / peak
            self.data.qpos[self.arm_qpos] = np.clip(
                self.data.qpos[self.arm_qpos] + delta, self.lower, self.upper
            )
            mujoco.mj_forward(self.model, self.data)
        current_position = self.data.xpos[self.wrist_body]
        current_rotation = self.data.xmat[self.wrist_body].reshape(3, 3)
        position_residual = float(np.linalg.norm(target_position - current_position))
        orientation_residual = float(np.linalg.norm(rotation_error(target_rotation, current_rotation)))
        return self.data.qpos[self.arm_qpos].copy(), position_residual, orientation_residual, iterations


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode < 0 or args.max_iterations <= 0:
        raise SystemExit("--episode must be non-negative and --max-iterations must be positive")
    if min(args.damping, args.max_step, args.position_tolerance, args.orientation_tolerance) <= 0:
        raise SystemExit("IK damping, step and tolerances must be positive")

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
        input_states = np.asarray(raw["input_states"], dtype=np.float64)
        latencies = np.asarray(raw["latencies_seconds"], dtype=np.float64)
    episode_length = len(source)
    if timestamps.shape != (episode_length,) or wrist_chunks.shape != (episode_length, 16, 15):
        raise SystemExit(
            f"complete episode required: timestamps={timestamps.shape}, wrist_chunks={wrist_chunks.shape}"
        )
    if input_states.shape != (episode_length, 13):
        raise SystemExit(f"input_states must be [T,13], got {input_states.shape}")
    if not np.allclose(timestamps, source_timestamps, atol=1e-9, rtol=0.0):
        raise SystemExit("inference/source timestamps do not match")
    expected_inputs = np.concatenate([source_states[:, 22:29], source_states[:, 35:41]], axis=1)
    if not np.allclose(input_states, expected_inputs, atol=1e-5, rtol=0.0):
        raise SystemExit("inference/source right-arm input states do not match")

    ik = RightArmIK(model_path, args)
    joint_chunks = np.empty((episode_length, 16, 13), dtype=np.float32)
    position_residuals = np.empty((episode_length, 16), dtype=np.float64)
    orientation_residuals = np.empty((episode_length, 16), dtype=np.float64)
    iteration_counts = np.empty((episode_length, 16), dtype=np.int32)
    source_position_errors = []
    source_orientation_errors = []
    hand_clip_count = int(np.count_nonzero((wrist_chunks[:, :, 9:15] < 0.0) | (wrist_chunks[:, :, 9:15] > 1.0)))

    for timestep in range(episode_length):
        ik.set_source_state(source_states[timestep])
        fk_position, fk_rotation = ik.wrist_pose_in_pelvis()
        source_position = source_eef[timestep, 7:10]
        source_rotation = quat_wxyz_to_matrix(source_eef[timestep, 10:14])
        source_position_errors.append(float(np.linalg.norm(fk_position - source_position)))
        source_orientation_errors.append(float(np.linalg.norm(rotation_error(source_rotation, fk_rotation))))
        pelvis_position, pelvis_rotation = ik.pelvis_pose()
        position = source_position.copy()
        rotation = source_rotation.copy()
        for horizon in range(16):
            wrist_action = wrist_chunks[timestep, horizon]
            position = position + rotation @ wrist_action[:3]
            rotation = rotation @ rotation_6d_to_matrix(wrist_action[3:9])
            target_position = pelvis_position + pelvis_rotation @ position
            target_rotation = pelvis_rotation @ rotation
            joints, pos_residual, rot_residual, iterations = ik.solve(
                target_position, target_rotation
            )
            joint_chunks[timestep, horizon, :7] = joints
            joint_chunks[timestep, horizon, 7:13] = np.clip(wrist_action[9:15], 0.0, 1.0)
            position_residuals[timestep, horizon] = pos_residual
            orientation_residuals[timestep, horizon] = rot_residual
            iteration_counts[timestep, horizon] = iterations
        if (timestep + 1) % 25 == 0 or timestep + 1 == episode_length:
            print(
                f"[{timestep + 1}/{episode_length}] "
                f"max_pos_residual={position_residuals[:timestep + 1].max():.6f}m "
                f"max_rot_residual={orientation_residuals[:timestep + 1].max():.6f}rad",
                flush=True,
            )

    max_source_position_error = float(np.max(source_position_errors))
    max_source_orientation_error = float(np.max(source_orientation_errors))
    if max_source_position_error > 1e-5 or max_source_orientation_error > 1e-5:
        raise SystemExit(
            f"source EEF/FK mismatch: {max_source_position_error:.6g}m, "
            f"{max_source_orientation_error:.6g}rad"
        )
    max_position_residual = float(position_residuals.max())
    max_orientation_residual = float(orientation_residuals.max())
    if (
        max_position_residual > args.max_position_residual
        or max_orientation_residual > args.max_orientation_residual
    ):
        raise SystemExit(
            f"IK residual exceeds safety threshold: {max_position_residual:.6f}m, "
            f"{max_orientation_residual:.6f}rad"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamps=timestamps,
        frame_indices=source["frame_index"].to_numpy(dtype=np.int64),
        input_states=expected_inputs.astype(np.float32),
        ground_truth_actions=np.concatenate(
            [source_actions[:, 22:29], source_actions[:, 35:41]], axis=1
        ).astype(np.float32),
        predicted_action_chunks=joint_chunks,
        first_predicted_actions=joint_chunks[:, 0],
        wrist_delta_action_chunks=wrist_chunks.astype(np.float32),
        latencies_seconds=latencies,
        ik_position_residuals=position_residuals,
        ik_orientation_residuals=orientation_residuals,
        ik_iteration_counts=iteration_counts,
    )
    raw_summary_path = input_path.with_suffix(".json")
    raw_summary = {}
    if raw_summary_path.exists():
        with raw_summary_path.open(encoding="utf-8") as handle:
            raw_summary = json.load(handle)
    summary = {
        **raw_summary,
        "source": f"complete episode_{args.episode:06d}, wrist delta converted by G1 MuJoCo IK",
        "source_parquet": str(source_path),
        "model": str(model_path),
        "frames": episode_length,
        "prediction_shape": list(joint_chunks.shape),
        "first_action_trajectory_shape": list(joint_chunks[:, 0].shape),
        "ik": {
            "max_source_fk_position_error_m": max_source_position_error,
            "max_source_fk_orientation_error_rad": max_source_orientation_error,
            "position_residual_m": {
                "mean": float(position_residuals.mean()),
                "p99": float(np.quantile(position_residuals, 0.99)),
                "max": max_position_residual,
            },
            "orientation_residual_rad": {
                "mean": float(orientation_residuals.mean()),
                "p99": float(np.quantile(orientation_residuals, 0.99)),
                "max": max_orientation_residual,
            },
            "iteration_count": {
                "mean": float(iteration_counts.mean()),
                "max": int(iteration_counts.max()),
            },
            "hand_values_clipped": hand_clip_count,
        },
        "npz": output.name,
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
