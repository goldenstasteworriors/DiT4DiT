"""Run direct DiT4DiT wrist-delta inference for every timestep of one episode."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.framework.share_tools import read_mode_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--joint-dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", default="pick up the pipette")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def episode_paths(dataset_root: Path, episode: int) -> tuple[Path, Path]:
    stem = f"episode_{episode:06d}"
    parquet = dataset_root / "data" / "chunk-000" / f"{stem}.parquet"
    video = (
        dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / f"{stem}.mp4"
    )
    return parquet, video


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def relative_wrist_state(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_rotation = quat_wxyz_to_matrix(previous[3:7])
    current_rotation = quat_wxyz_to_matrix(current[3:7])
    relative_position = previous_rotation.T @ (current[:3] - previous[:3])
    relative_rotation = previous_rotation.T @ current_rotation
    state = np.concatenate(
        [relative_position, relative_rotation[:2].reshape(6), current[7:13]], axis=0
    ).astype(np.float32)
    return np.pad(state, (0, 1))


def save_result(
    output: Path,
    *,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    input_states: np.ndarray,
    input_wrist_states: np.ndarray,
    ground_truth_actions: np.ndarray,
    ground_truth_wrist_actions: np.ndarray,
    wrist_chunks: np.ndarray,
    latencies: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamps=timestamps,
        frame_indices=frame_indices,
        input_states=input_states,
        input_wrist_states=input_wrist_states,
        ground_truth_actions=ground_truth_actions,
        ground_truth_wrist_delta_actions=ground_truth_wrist_actions,
        predicted_wrist_delta_chunks=wrist_chunks,
        first_predicted_wrist_delta_actions=wrist_chunks[:, 0],
        latencies_seconds=latencies,
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode < 0 or args.save_every <= 0:
        raise SystemExit("--episode must be non-negative and --save-every must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    joint_dataset_root = args.joint_dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    partial_output = output.with_name(f"{output.stem}.partial{output.suffix}")
    wrist_parquet, video_path = episode_paths(dataset_root, args.episode)
    joint_parquet, _ = episode_paths(joint_dataset_root, args.episode)

    wrist_frame = pd.read_parquet(wrist_parquet)
    joint_frame = pd.read_parquet(joint_parquet)
    if len(wrist_frame) != len(joint_frame):
        raise SystemExit(f"wrist/joint episode length mismatch: {len(wrist_frame)} != {len(joint_frame)}")
    timestamps = wrist_frame["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = wrist_frame["frame_index"].to_numpy(dtype=np.int64)
    joint_timestamps = joint_frame["timestamp"].to_numpy(dtype=np.float64)
    if not np.allclose(timestamps, joint_timestamps, atol=1e-9, rtol=0.0):
        raise SystemExit("wrist/joint episode timestamps do not match")

    wrist_states = np.stack(wrist_frame["observation.right_wrist_hand"]).astype(np.float32)
    ground_truth_wrist = np.stack(wrist_frame["action.right_wrist_delta_hand"]).astype(np.float32)
    input_states = np.stack(joint_frame["observation.right_arm_hand"]).astype(np.float32)
    ground_truth_actions = np.stack(joint_frame["action.right_arm_hand"]).astype(np.float32)
    expected = len(wrist_frame)
    for name, values, width in (
        ("wrist state", wrist_states, 13),
        ("wrist action", ground_truth_wrist, 15),
        ("joint state", input_states, 13),
        ("joint action", ground_truth_actions, 13),
    ):
        if values.shape != (expected, width) or not np.isfinite(values).all():
            raise SystemExit(f"invalid {name}: {values.shape}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open episode video: {video_path}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if video_frames != expected:
        raise SystemExit(f"video/parquet length mismatch: {video_frames} != {expected}")

    print(f"Loading checkpoint: {checkpoint}", flush=True)
    policy = baseframework.from_pretrained(str(checkpoint))
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()
    _, norm_stats = read_mode_config(checkpoint)
    unnorm_key = baseframework._check_unnorm_key(norm_stats, None)
    action_stats = norm_stats[unnorm_key]["action"]
    action_dim = len(action_stats["q01"])
    if action_dim != 15:
        raise SystemExit(f"checkpoint action statistics must be 15-D, got {action_dim}")

    wrist_chunks = np.empty((expected, 16, 15), dtype=np.float32)
    latencies = np.empty(expected, dtype=np.float64)
    completed = 0
    started = time.time()
    try:
        for index in range(expected):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"video decode failed at frame {index}")
            image = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224))
            previous_index = max(0, index - 1)
            state = relative_wrist_state(wrist_states[previous_index], wrist_states[index])
            example = {
                "image": [image],
                "lang": args.instruction,
                "state": state[None],
            }
            begin = time.perf_counter()
            output_dict = policy.predict_action([example])
            latencies[index] = time.perf_counter() - begin
            normalized = np.asarray(output_dict["normalized_actions"][0], dtype=np.float32)
            normalized = normalized[:16, :action_dim]
            actions = baseframework.unnormalize_actions(normalized, action_stats).astype(np.float32)
            if actions.shape != (16, 15) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid inference output at frame {index}: {actions.shape}")
            wrist_chunks[index] = actions
            completed = index + 1
            if completed % args.save_every == 0 or completed == expected:
                save_result(
                    partial_output,
                    timestamps=timestamps[:completed],
                    frame_indices=frame_indices[:completed],
                    input_states=input_states[:completed],
                    input_wrist_states=wrist_states[:completed],
                    ground_truth_actions=ground_truth_actions[:completed],
                    ground_truth_wrist_actions=ground_truth_wrist[:completed],
                    wrist_chunks=wrist_chunks[:completed],
                    latencies=latencies[:completed],
                )
                elapsed = time.time() - started
                eta = elapsed / completed * (expected - completed)
                print(
                    f"[{completed}/{expected}] latency={latencies[index]:.3f}s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    finally:
        capture.release()

    partial_output.replace(output)
    step_match = re.search(r"steps_(\d+)_pytorch_model\.pt$", checkpoint.name)
    summary = {
        "source": f"complete episode_{args.episode:06d}",
        "dataset_root": str(dataset_root),
        "joint_dataset_root": str(joint_dataset_root),
        "frames": expected,
        "fps": video_fps,
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "instruction": args.instruction,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step_match.group(1)) if step_match else None,
        "prediction_shape": list(wrist_chunks.shape),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "median": float(np.median(latencies)),
            "min": float(latencies.min()),
            "max": float(latencies.max()),
            "total": float(latencies.sum()),
        },
        "npz": output.name,
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
