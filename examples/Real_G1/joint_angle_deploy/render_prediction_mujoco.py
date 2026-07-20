"""Render recorded G1 right-arm and Inspire-hand predictions in MuJoCo."""

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
RIGHT_HAND_JOINTS = [
    "right_hand_little_joint",
    "right_hand_ring_joint",
    "right_hand_middle_joint",
    "right_hand_index_joint",
    "right_hand_thumb_rotate_joint",
    "right_hand_thumb_bend_joint",
]
DEFAULT_INITIAL_RIGHT_ARM = np.array(
    [-0.36188766, -0.19208317, 0.33666086, -0.45916361, 0.39308259, 0.59385431, -0.44077981]
)
DEFAULT_INITIAL_RIGHT_HAND = np.ones(6)


def minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def inspire_dds_to_mujoco(q: np.ndarray) -> np.ndarray:
    """Convert DDS order/scale to MuJoCo order/radians (DDS: 1=open, 0=closed)."""
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
    if q.shape != (6,):
        raise ValueError(f"Expected 6 Inspire-hand values, got {q.shape}")
    return np.array(
        [
            (1.0 - q[0]) * 1.7,  # little
            (1.0 - q[1]) * 1.7,  # ring
            (1.0 - q[2]) * 1.7,  # middle
            (1.0 - q[3]) * 1.7,  # index
            (1.0 - q[5]) * 1.4 - 0.1,  # thumb rotate
            (1.0 - q[4]) * 0.5,  # thumb bend
        ]
    )


def build_timeline(
    arm_actions: np.ndarray,
    hand_actions: np.ndarray,
    initial_hand: np.ndarray,
    fps: int,
    init_seconds: float,
    action_dt: float,
):
    frames = []
    down_pose = np.zeros(7)
    for i in range(round(init_seconds * fps)):
        q = minimum_jerk(down_pose, DEFAULT_INITIAL_RIGHT_ARM, (i + 1) / (init_seconds * fps))
        frames.append((q, initial_hand.copy(), "INITIALIZING: smoothly raising right arm"))
    frames.extend(
        [(DEFAULT_INITIAL_RIGHT_ARM.copy(), initial_hand.copy(), "READY: training initial pose")] * fps
    )
    interpolation_frames = max(1, round(action_dt * fps))
    previous_arm = DEFAULT_INITIAL_RIGHT_ARM.copy()
    previous_hand = initial_hand.copy()
    for step, (arm_action, hand_action) in enumerate(zip(arm_actions, hand_actions)):
        for i in range(interpolation_frames):
            progress = (i + 1) / interpolation_frames
            arm_q = minimum_jerk(previous_arm, arm_action, progress)
            hand_q = minimum_jerk(previous_hand, hand_action, progress)
            frames.append((arm_q, hand_q, f"PREDICTION step {step + 1:02d}/{len(arm_actions):02d}"))
        previous_arm = arm_action.copy()
        previous_hand = hand_action.copy()
    frames.extend(
        [(previous_arm.copy(), previous_hand.copy(), "HOLD: final predicted pose")] * (2 * fps)
    )
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
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=root.parents[1]
        / "SONICMJ/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/scene_29dof_inspire.xml",
        help="G1 + Inspire-hand MuJoCo scene (defaults to the SONICMJ reference project)",
    )
    parser.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo window instead of rendering MP4")
    parser.add_argument("--loop", action="store_true", help="restart from the arms-down pose after the trajectory ends")
    args = parser.parse_args()

    record = json.loads(args.result.read_text())
    actions = np.asarray(record["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 13 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite (T, 13) arm+hand actions, got {actions.shape}")
    arm_actions = actions[:, :7]
    hand_actions = actions[:, 7:13]
    state = np.asarray(record.get("state", []), dtype=np.float64)
    initial_hand = state[7:13] if state.shape == (13,) else DEFAULT_INITIAL_RIGHT_HAND.copy()

    if not args.model_xml.is_file():
        raise FileNotFoundError(f"G1 Inspire-hand MuJoCo scene not found: {args.model_xml}")
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    arm_qpos_indices = []
    hand_qpos_indices = []
    for name in RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        target = arm_qpos_indices if name in RIGHT_ARM_JOINTS else hand_qpos_indices
        target.append(model.jnt_qposadr[joint_id])

    timeline = build_timeline(
        arm_actions, hand_actions, initial_hand, args.fps, args.init_seconds, args.action_dt
    )
    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance = 2.15
            viewer.cam.azimuth = 150
            viewer.cam.elevation = -12
            print("MuJoCo viewer started. Close the window or press Ctrl-C to stop.")
            while viewer.is_running():
                previous_phase = None
                for arm_q, hand_q, phase in timeline:
                    if not viewer.is_running():
                        break
                    start = time.monotonic()
                    data.qpos[arm_qpos_indices] = arm_q
                    data.qpos[hand_qpos_indices] = inspire_dds_to_mujoco(hand_q)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    if phase != previous_phase:
                        print(f"{phase} | right hand DDS [L R M I TB TR]: " + " ".join(f"{x:.3f}" for x in hand_q))
                        previous_phase = phase
                    time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - start)))
                if not args.loop:
                    break
                data.qpos[arm_qpos_indices] = 0.0
                data.qpos[hand_qpos_indices] = inspire_dds_to_mujoco(initial_hand)
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
        for frame_index, (arm_q, hand_q, phase) in enumerate(timeline):
            data.qpos[arm_qpos_indices] = arm_q
            data.qpos[hand_qpos_indices] = inspire_dds_to_mujoco(hand_q)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (18, 16), (940, 154), (10, 10, 10), -1)
            cv2.putText(frame, phase, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2)
            cv2.putText(
                frame,
                f"t={frame_index / args.fps:5.2f}s   right arm q(rad): "
                + " ".join(f"{value:+.3f}" for value in arm_q),
                (32, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
            )
            cv2.putText(
                frame,
                "right hand DDS (1=open) [L R M I TB TR]: "
                + " ".join(f"{value:.3f}" for value in hand_q),
                (32, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (170, 255, 170),
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
