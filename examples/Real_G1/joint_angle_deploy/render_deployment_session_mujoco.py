"""Loop a recorded deployment session using the exact online IK action steps."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

if os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
    os.environ.pop("MUJOCO_GL", None)

import mujoco
import numpy as np


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def build_argparser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="deployment_model_io session; default selects the newest session",
    )
    parser.add_argument(
        "--record-root",
        type=Path,
        default=root / "inference_records/deployment_model_io",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=root / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/scene_29dof.xml",
    )
    parser.add_argument(
        "--pose-source",
        choices=("ik-target", "commanded", "measured"),
        default="ik-target",
        help="ik-target replays sequentially accumulated wrist-delta IK results",
    )
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--no-loop", action="store_true")
    return parser


def newest_session(root: Path) -> Path:
    sessions = [path for path in root.expanduser().resolve().iterdir() if path.is_dir()]
    if not sessions:
        raise SystemExit(f"no deployment sessions under {root}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def load_ik_targets(session: Path, horizon: int | None) -> np.ndarray:
    chunks = []
    for path in sorted(session.glob("inference_*.npz")):
        with np.load(path, allow_pickle=False) as record:
            chunk = np.asarray(record["joint_action_output"], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"invalid joint_action_output in {path}: {chunk.shape}")
        chunks.append(chunk[:horizon, :7] if horizon is not None else chunk[:, :7])
    if not chunks:
        raise SystemExit(f"no inference NPZ records in {session}")
    return np.concatenate(chunks, axis=0)


def load_tracking(session: Path, source: str) -> np.ndarray:
    path = session / "execution_tracking.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing; this session predates execution tracking")
    prefix = "commanded_arm" if source == "commanded" else "measured_arm"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no execution rows in {path}")
    return np.asarray(
        [[float(row[f"{prefix}_{index}"]) for index in range(7)] for row in rows],
        dtype=np.float64,
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.frequency <= 0 or (args.execution_horizon is not None and args.execution_horizon <= 0):
        raise SystemExit("frequency and execution-horizon must be positive")
    session = args.session.expanduser().resolve() if args.session else newest_session(args.record_root)
    poses = (
        load_ik_targets(session, args.execution_horizon)
        if args.pose_source == "ik-target"
        else load_tracking(session, args.pose_source)
    )
    model = mujoco.MjModel.from_xml_path(str(args.model_xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    qpos_indices = []
    for name in RIGHT_ARM_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"joint missing from model: {name}")
        qpos_indices.append(model.jnt_qposadr[joint])

    from mujoco import viewer as mj_viewer

    print(f"session: {session}")
    print(f"pose source/steps: {args.pose_source} / {len(poses)}")
    print(f"playback: {args.frequency:.1f} Hz, {'once' if args.no_loop else 'loop'}")
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
        viewer.cam.distance = 2.15
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -12
        while viewer.is_running():
            for pose in poses:
                if not viewer.is_running():
                    return
                started = time.monotonic()
                data.qpos[qpos_indices] = pose
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, 1.0 / args.frequency - (time.monotonic() - started)))
            if args.no_loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                return


if __name__ == "__main__":
    main()
