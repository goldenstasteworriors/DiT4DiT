"""Render the exact rate-limited full-episode right-arm commands in MuJoCo."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

if "--viewer" in sys.argv and os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
    print("[viewer] ignoring headless MUJOCO_GL; using the windowed GLFW backend", file=sys.stderr)
    os.environ.pop("MUJOCO_GL", None)

import mujoco
import numpy as np

from play_right_arm_trajectory import (
    build_rate_limited_commands,
    load_full_episode,
    minimum_jerk,
    resolve_trajectory,
    validate_commands,
)


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
    parser.add_argument("--episode", type=int, default=0, help="episode selected from inference_records")
    parser.add_argument("--trajectory", type=Path, default=None, help="explicit NPZ overrides --episode lookup")
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=root
        / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/scene_29dof.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--aggregation", choices=("temporal-ensemble", "first"), default="temporal-ensemble")
    parser.add_argument("--slowdown", type=float, default=2.0)
    parser.add_argument("--control-frequency", type=float, default=100.0)
    parser.add_argument("--max-speed", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--init-seconds", type=float, default=5.0)
    parser.add_argument("--ready-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--viewer", action="store_true")
    return parser


def make_render_timeline(
    initial_pose: np.ndarray,
    commands: np.ndarray,
    control_frequency: float,
    render_fps: int,
    init_seconds: float,
    ready_seconds: float,
    hold_seconds: float,
) -> list[tuple[np.ndarray, str, float]]:
    timeline: list[tuple[np.ndarray, str, float]] = []
    down_pose = np.zeros(7, dtype=np.float64)
    for frame in range(round(init_seconds * render_fps)):
        pose = minimum_jerk(down_pose, initial_pose, (frame + 1) / (init_seconds * render_fps))
        timeline.append((pose, "INITIALIZING", 0.0))
    timeline.extend([(initial_pose.copy(), "READY", 0.0)] * round(ready_seconds * render_fps))

    playback_duration = (len(commands) - 1) / control_frequency
    previous = commands[0]
    for frame in range(round(playback_duration * render_fps) + 1):
        elapsed = frame / render_fps
        command_index = min(round(elapsed * control_frequency), len(commands) - 1)
        command = commands[command_index]
        speed = float(np.max(np.abs(command - previous)) * render_fps)
        timeline.append((command.copy(), "PLAYING 2x SLOW", speed))
        previous = command
    timeline.extend([(commands[-1].copy(), "HOLD FINAL POSE", 0.0)] * round(hold_seconds * render_fps))
    return timeline


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(__file__).resolve().parents[3]
    if min(
        args.slowdown,
        args.control_frequency,
        args.max_speed,
        args.fps,
        args.width,
        args.height,
        args.init_seconds,
        args.ready_seconds,
        args.hold_seconds,
    ) <= 0:
        raise SystemExit("all timing, speed, FPS, and image-size arguments must be positive")

    trajectory_path = resolve_trajectory(args.episode, args.trajectory)
    model_path = args.model_xml.expanduser().resolve()
    initial_pose, targets, timestamps, metadata = load_full_episode(
        trajectory_path, args.aggregation
    )
    source_dt = float(np.median(np.diff(timestamps)))
    target_dt = source_dt * args.slowdown
    commands, tracking_lag = build_rate_limited_commands(
        initial_pose, targets, target_dt, args.control_frequency, args.max_speed
    )
    command_peak_speed = validate_commands(commands, args.control_frequency, args.max_speed)
    raw_peak_speed = float(np.max(np.abs(np.diff(targets, axis=0))) / target_dt)
    timeline = make_render_timeline(
        initial_pose,
        commands,
        args.control_frequency,
        args.fps,
        args.init_seconds,
        args.ready_seconds,
        args.hold_seconds,
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    qpos_indices = []
    for joint_name in RIGHT_ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {joint_name}")
        qpos_indices.append(model.jnt_qposadr[joint_id])

    print(f"trajectory: {trajectory_path}")
    print(f"checkpoint: {metadata.get('checkpoint', 'unknown')}")
    print(f"targets/commands: {targets.shape} / {commands.shape}")
    print(f"raw/limited peak speed: {raw_peak_speed:.3f} / {command_peak_speed:.3f} rad/s")
    print(f"maximum target tracking lag: {tracking_lag:.3f} rad")

    if args.viewer:
        from mujoco import viewer as mj_viewer

        try:
            with mj_viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
                viewer.cam.distance = 2.15
                viewer.cam.azimuth = 150
                viewer.cam.elevation = -12
                for pose, _, _ in timeline:
                    if not viewer.is_running():
                        break
                    started = time.monotonic()
                    data.qpos[qpos_indices] = pose
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - started)))
        except KeyboardInterrupt:
            print("\nviewer interrupted by Ctrl-C")
        return

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 2.15
    camera.azimuth = 150
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root
        / "inference_records"
        / f"g1_inference_episode_{args.episode:06d}_2x_slow_simulation.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")

    try:
        for frame_index, (pose, phase, displayed_speed) in enumerate(timeline):
            data.qpos[qpos_indices] = pose
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (16, 14), (944, 145), (10, 10, 10), -1)
            cv2.putText(image, phase, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2)
            cv2.putText(
                image,
                f"t={frame_index / args.fps:5.2f}s  sampled speed={displayed_speed:.3f} rad/s  "
                f"validated 100Hz peak={command_peak_speed:.3f} rad/s",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
            )
            cv2.putText(
                image,
                "right arm q(rad): " + " ".join(f"{value:+.3f}" for value in pose),
                (30, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (170, 255, 170),
                1,
            )
            writer.write(image)
    finally:
        writer.release()
        renderer.close()

    print(f"rendered {len(timeline)} frames ({len(timeline) / args.fps:.2f}s) to {output}")


if __name__ == "__main__":
    main()
