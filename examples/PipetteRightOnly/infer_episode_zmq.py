"""Run frame-by-frame ZMQ inference for a complete derived LeRobot episode."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import zmq

from deployment.model_server.tools import msgpack_numpy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--instruction", default="pick up the pipette")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def episode_paths(dataset_root: Path, episode: int) -> tuple[Path, Path]:
    filename = f"episode_{episode:06d}"
    parquet = dataset_root / "data" / "chunk-000" / f"{filename}.parquet"
    video = (
        dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / f"{filename}.mp4"
    )
    return parquet, video


def save_result(
    output: Path,
    *,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    input_states: np.ndarray,
    ground_truth_actions: np.ndarray,
    action_chunks: np.ndarray,
    latencies: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamps=timestamps,
        frame_indices=frame_indices,
        input_states=input_states,
        ground_truth_actions=ground_truth_actions,
        predicted_action_chunks=action_chunks,
        first_predicted_actions=action_chunks[:, 0],
        latencies_seconds=latencies,
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.timeout_ms <= 0 or args.save_every <= 0:
        raise SystemExit("--timeout-ms and --save-every must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    partial_output = output.with_name(f"{output.stem}.partial{output.suffix}")
    parquet_path, video_path = episode_paths(dataset_root, args.episode)
    frame = pd.read_parquet(parquet_path)
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    input_states = np.stack(frame["observation.right_arm_hand"].to_numpy()).astype(np.float32)
    ground_truth_actions = np.stack(frame["action.right_arm_hand"].to_numpy()).astype(np.float32)
    if input_states.shape != (len(frame), 13) or ground_truth_actions.shape != (len(frame), 13):
        raise SystemExit(
            f"expected [T,13] state/action, got {input_states.shape}/{ground_truth_actions.shape}"
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open episode video: {video_path}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if video_frames != len(frame):
        raise SystemExit(f"video/parquet length mismatch: {video_frames} != {len(frame)}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{args.server}:{args.port}")
    action_chunks = np.empty((len(frame), 16, 13), dtype=np.float32)
    latencies = np.empty(len(frame), dtype=np.float64)
    completed = 0
    started = time.time()

    try:
        for index in range(len(frame)):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"video decode failed at frame {index}")
            image = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224))
            request = {
                "endpoint": "get_action",
                "data": {
                    "examples": [
                        {
                            "image": [image],
                            "lang": args.instruction,
                            "state": input_states[index][None],
                        }
                    ]
                },
            }
            begin = time.perf_counter()
            socket.send(msgpack_numpy.packb(request))
            response = msgpack_numpy.unpackb(socket.recv())
            latencies[index] = time.perf_counter() - begin
            if not response.get("ok"):
                raise RuntimeError(f"inference failed at frame {index}: {response.get('error')}")
            actions = np.asarray(response["data"]["unnormalized_actions"], dtype=np.float32)
            if actions.shape != (16, 13) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid inference output at frame {index}: {actions.shape}")
            action_chunks[index] = actions
            completed = index + 1
            if completed % args.save_every == 0 or completed == len(frame):
                save_result(
                    partial_output,
                    timestamps=timestamps[:completed],
                    frame_indices=frame_indices[:completed],
                    input_states=input_states[:completed],
                    ground_truth_actions=ground_truth_actions[:completed],
                    action_chunks=action_chunks[:completed],
                    latencies=latencies[:completed],
                )
                elapsed = time.time() - started
                remaining = elapsed / completed * (len(frame) - completed)
                print(
                    f"[{completed}/{len(frame)}] latency={latencies[index]:.3f}s "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                    flush=True,
                )
    finally:
        capture.release()
        socket.close()
        context.term()

    partial_output.replace(output)
    summary = {
        "source": f"complete episode_{args.episode:06d}",
        "dataset_root": str(dataset_root),
        "parquet": str(parquet_path),
        "video": str(video_path),
        "frames": len(frame),
        "fps": video_fps,
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "instruction": args.instruction,
        "checkpoint": args.checkpoint,
        "prediction_shape": list(action_chunks.shape),
        "first_action_trajectory_shape": list(action_chunks[:, 0].shape),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "median": float(np.median(latencies)),
            "min": float(latencies.min()),
            "max": float(latencies.max()),
            "total": float(latencies.sum()),
        },
        "npz": output.name,
    }
    summary_path = output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
