"""Render the legacy SONIC pick_up_pipette LeRobot format in MuJoCo.

This adapter intentionally targets the current 41-value ``observation.state``
and ``action.wbc`` schema only.  A future data format should get a separate,
explicit adapter rather than silent shape guessing here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_DATASET = Path(
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/pick_up_pipette"
)
DEFAULT_MODEL = Path(
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/"
    "decoupled_wbc/control/robot_model/model_data/g1/scene_29dof_inspire.xml"
)
HAND_JOINT_SUFFIXES = (
    "little_joint",
    "ring_joint",
    "middle_joint",
    "index_joint",
    "thumb_rotate_joint",
    "thumb_bend_joint",
)


def build_argparser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=("observation", "action"),
        default="observation",
        help="observation replays measured state; action replays action.wbc targets",
    )
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "inference_records/collected_episode_000000_simulation.mp4",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 is recorded wall-clock speed")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    return parser


def episode_path(dataset: Path, info: dict, episode: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return dataset / pattern.format(
        episode_chunk=episode // chunk_size, episode_index=episode
    )


def validate_legacy_modality(dataset: Path) -> None:
    modality_path = dataset / "meta" / "modality.json"
    with modality_path.open(encoding="utf-8") as handle:
        state = json.load(handle)["state"]
    expected = {
        "left_leg": (0, 6),
        "right_leg": (6, 12),
        "waist": (12, 15),
        "left_arm": (15, 22),
        "right_arm": (22, 29),
        "left_hand": (29, 35),
        "right_hand": (35, 41),
    }
    actual = {key: (state[key]["start"], state[key]["end"]) for key in expected if key in state}
    if actual != expected:
        raise ValueError(f"unsupported modality layout: expected {expected}, got {actual}")


def load_episode(dataset: Path, episode: int, source: str) -> tuple[np.ndarray, np.ndarray, float, Path]:
    with (dataset / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    validate_legacy_modality(dataset)
    parquet = episode_path(dataset, info, episode)
    frame = pd.read_parquet(parquet)
    column = "observation.state" if source == "observation" else "action.wbc"
    if column not in frame or frame.empty:
        raise ValueError(f"episode is empty or missing {column}: {parquet}")
    order_key = "frame_index" if "frame_index" in frame else "timestamp"
    frame = frame.sort_values(order_key)
    values = np.stack(frame[column].to_numpy()).astype(np.float64)
    if values.shape != (len(frame), 41) or not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite [T,41] values, got {values.shape}")
    timestamps = (
        frame["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in frame
        else np.arange(len(frame), dtype=np.float64) / float(info.get("fps", 50.0))
    )
    fps = float(info.get("fps", 50.0))
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("episode timestamps must be strictly increasing")
    return values, timestamps, fps, parquet


def inspire_dds_to_mujoco(values: np.ndarray) -> np.ndarray:
    """Convert native Inspire order/scale to the six MuJoCo finger joints."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.array(
        [
            (1.0 - values[0]) * 1.7,
            (1.0 - values[1]) * 1.7,
            (1.0 - values[2]) * 1.7,
            (1.0 - values[3]) * 1.7,
            (1.0 - values[5]) * 1.4 - 0.1,
            (1.0 - values[4]) * 0.5,
        ],
        dtype=np.float64,
    )


def joint_qpos_indices(model: mujoco.MjModel) -> tuple[list[int], list[int], list[int]]:
    body_indices = []
    for actuator_index in range(29):
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index
        )
        if joint_name != actuator_name:
            raise ValueError(
                f"body actuator/joint order mismatch at {actuator_index}: "
                f"{actuator_name} != {joint_name}"
            )
        body_indices.append(int(model.jnt_qposadr[joint_id]))

    hand_indices = []
    for side in ("left", "right"):
        indices = []
        for suffix in HAND_JOINT_SUFFIXES:
            name = f"{side}_hand_{suffix}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Inspire joint not found: {name}")
            indices.append(int(model.jnt_qposadr[joint_id]))
        hand_indices.append(indices)
    return body_indices, hand_indices[0], hand_indices[1]


def validate_body_limits(model: mujoco.MjModel, body_indices: list[int], body: np.ndarray) -> None:
    violations = []
    for actuator_index, qpos_index in enumerate(body_indices):
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        if not model.jnt_limited[joint_id]:
            continue
        low, high = model.jnt_range[joint_id]
        minimum = float(body[:, actuator_index].min())
        maximum = float(body[:, actuator_index].max())
        if minimum < low - 1e-6 or maximum > high + 1e-6:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append(f"{name}: [{minimum:.3f}, {maximum:.3f}] vs [{low:.3f}, {high:.3f}]")
    if violations:
        raise ValueError("recorded body joints exceed MuJoCo limits: " + "; ".join(violations))


def set_pose(
    data: mujoco.MjData,
    sample: np.ndarray,
    body_indices: list[int],
    left_hand_indices: list[int],
    right_hand_indices: list[int],
    replay_hands: bool,
) -> None:
    data.qpos[body_indices] = sample[:29]
    if replay_hands:
        data.qpos[left_hand_indices] = inspire_dds_to_mujoco(sample[29:35])
        data.qpos[right_hand_indices] = inspire_dds_to_mujoco(sample[35:41])


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 12, image.width - 14, 116), fill=(10, 10, 10))
    for index, line in enumerate(lines):
        draw.text((28, 24 + index * 27), line, fill=(235, 235, 235))
    return np.asarray(image)


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode < 0 or min(args.speed, args.fps, args.width, args.height) <= 0:
        raise SystemExit("episode must be non-negative; speed/FPS/image size must be positive")
    dataset = args.dataset.expanduser().resolve()
    model_path = args.model_xml.expanduser().resolve()
    samples, timestamps, dataset_fps, parquet = load_episode(dataset, args.episode, args.source)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    body_indices, left_hand_indices, right_hand_indices = joint_qpos_indices(model)
    validate_body_limits(model, body_indices, samples[:, :29])

    duration = float(timestamps[-1] - timestamps[0]) / args.speed
    body_speed = np.abs(np.diff(samples[:, :29], axis=0)) / np.diff(timestamps)[:, None]
    peak_speed = float(body_speed.max() * args.speed)
    peak_joint_index = int(np.unravel_index(np.argmax(body_speed), body_speed.shape)[1])
    peak_joint_id = int(model.actuator_trnid[peak_joint_index, 0])
    peak_joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, peak_joint_id)
    print(f"dataset: {dataset}")
    print(f"parquet: {parquet}")
    print(f"source: {'observation.state (measured)' if args.source == 'observation' else 'action.wbc (commanded)'}")
    print(f"episode: {args.episode}, frames={len(samples)}, fps={dataset_fps:g}, duration={duration:.2f}s")
    print(f"peak sampled body-joint speed: {peak_speed:.3f} rad/s ({peak_joint})")
    print(f"hands: {'disabled' if args.no_hands else 'enabled'}")

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance = 2.15
            viewer.cam.azimuth = 150
            viewer.cam.elevation = -12
            started = time.monotonic()
            while viewer.is_running():
                elapsed = (time.monotonic() - started) * args.speed
                index = min(int(np.searchsorted(timestamps - timestamps[0], elapsed)), len(samples) - 1)
                set_pose(
                    data, samples[index], body_indices, left_hand_indices, right_hand_indices, not args.no_hands
                )
                mujoco.mj_forward(model, data)
                viewer.sync()
                if index == len(samples) - 1:
                    break
                time.sleep(1.0 / args.fps)
        return

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 2.15
    camera.azimuth = 150
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8)
    render_frames = round(duration * args.fps) + 1
    try:
        for render_index in range(render_frames):
            recorded_time = render_index / args.fps * args.speed
            source_index = min(
                int(np.searchsorted(timestamps - timestamps[0], recorded_time)), len(samples) - 1
            )
            set_pose(
                data,
                samples[source_index],
                body_indices,
                left_hand_indices,
                right_hand_indices,
                not args.no_hands,
            )
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = overlay(
                renderer.render(),
                [
                    f"COLLECTED episode {args.episode:06d} | {args.source} | {args.speed:g}x",
                    f"t={recorded_time:5.2f}/{timestamps[-1] - timestamps[0]:5.2f}s  frame={source_index + 1}/{len(samples)}",
                    f"peak sampled body speed={peak_speed:.3f} rad/s  hands={'off' if args.no_hands else 'on'}",
                ],
            )
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()
    print(f"rendered {render_frames} frames to {output}")


if __name__ == "__main__":
    main()
