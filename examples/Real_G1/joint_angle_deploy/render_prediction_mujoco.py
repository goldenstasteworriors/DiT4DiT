"""Render a recorded G1 right-arm prediction with the project's MuJoCo model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
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
DEFAULT_INITIAL_RIGHT_ARM = np.array([-0.060281, -0.251992, -0.072517, -0.577184, 0.402035, 0.493582, -0.250482])


def minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def build_timeline(actions: np.ndarray, fps: int, init_seconds: float, action_dt: float):
    frames = []
    down_pose = np.zeros(7)
    for i in range(round(init_seconds * fps)):
        q = minimum_jerk(down_pose, DEFAULT_INITIAL_RIGHT_ARM, (i + 1) / (init_seconds * fps))
        frames.append((q, "INITIALIZING: smoothly raising right arm"))
    frames.extend([(DEFAULT_INITIAL_RIGHT_ARM.copy(), "READY: training initial pose")] * fps)
    interpolation_frames = max(1, round(action_dt * fps))
    previous = DEFAULT_INITIAL_RIGHT_ARM.copy()
    for step, action in enumerate(actions):
        for i in range(interpolation_frames):
            q = minimum_jerk(previous, action, (i + 1) / interpolation_frames)
            frames.append((q, f"PREDICTION step {step + 1:02d}/{len(actions):02d}"))
        previous = action.copy()
    frames.extend([(previous.copy(), "HOLD: final predicted pose")] * (2 * fps))
    return frames


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=root / "inference_records/local_to_a800_training_episode_000000_result.json",
    )
    parser.add_argument("--output", type=Path, default=root / "inference_records/g1_joint_prediction_simulation.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--init-seconds", type=float, default=4.0)
    parser.add_argument("--action-dt", type=float, default=0.25, help="visual playback seconds per predicted step")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo window instead of rendering MP4")
    parser.add_argument("--loop", action="store_true", help="restart from the arms-down pose after the trajectory ends")
    args = parser.parse_args()

    record = json.loads(args.result.read_text())
    actions = np.asarray(record["actions"], dtype=np.float64)[:, :7]
    if actions.shape != (16, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite (16, 7) right-arm actions, got {actions.shape}")

    xml = root / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/scene_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    qpos_indices = []
    for name in RIGHT_ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        qpos_indices.append(model.jnt_qposadr[joint_id])

    timeline = build_timeline(actions, args.fps, args.init_seconds, args.action_dt)
    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance = 2.15
            viewer.cam.azimuth = 150
            viewer.cam.elevation = -12
            print("MuJoCo viewer started. Close the window or press Ctrl-C to stop.")
            while viewer.is_running():
                for q, _ in timeline:
                    if not viewer.is_running():
                        break
                    start = time.monotonic()
                    data.qpos[qpos_indices] = q
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - start)))
                if not args.loop:
                    break
                data.qpos[qpos_indices] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.5)
        return

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 2.15
    camera.azimuth = 150
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.output}")

    try:
        for frame_index, (q, phase) in enumerate(timeline):
            data.qpos[qpos_indices] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (18, 16), (940, 122), (10, 10, 10), -1)
            cv2.putText(frame, phase, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2)
            cv2.putText(
                frame,
                f"t={frame_index / args.fps:5.2f}s   right arm q(rad): "
                + " ".join(f"{value:+.3f}" for value in q),
                (32, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
            )
            cv2.putText(
                frame,
                "SP       SR       SY       EL       WR       WP       WY",
                (310, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()
        renderer.close()

    print(f"Rendered {len(timeline)} frames ({len(timeline) / args.fps:.2f}s) to {args.output}")


if __name__ == "__main__":
    main()
