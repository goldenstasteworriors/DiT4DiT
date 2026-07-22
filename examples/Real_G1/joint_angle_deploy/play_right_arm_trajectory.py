"""Validate and safely replay a complete DiT4DiT right-arm episode on G1.

The full inference NPZ contains one 16-step prediction chunk per observation.
This script causally ensembles overlapping chunks, plays targets two times
slower than the dataset by default, and precomputes rate-limited right-arm and
right-Inspire-hand command sequences before any DDS connection is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np


DEFAULT_RECORDS_DIR = Path(__file__).resolve().parents[3] / "inference_records"
DEFAULT_SOURCE_DATASET = Path(
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/pick_up_pipette"
)
LEFT_ARM_LOWER = np.array([-3.0892, -1.5882, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
LEFT_ARM_UPPER = np.array([2.6704, 2.2515, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
RIGHT_ARM_LOWER = np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
RIGHT_ARM_UPPER = np.array([2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
ARM_JOINT_NAMES = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)


class TrajectoryTrackingRecorder:
    """Stream desired, sent, and measured right-arm joints for each control cycle."""

    def __init__(self, path: Path, trajectory: Path, frequency: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        self._started = time.monotonic()
        self._rows = 0
        fields = [
            "wall_time_unix", "elapsed_s", "phase", "control_cycle",
            "command_index", "trajectory_index", "trajectory_file", "frequency_hz",
        ]
        for prefix in ("trajectory_target", "sent_target", "measured", "trajectory_error", "sent_error", "init_correction"):
            fields.extend(f"{prefix}_{name}" for name in ARM_JOINT_NAMES)
        self._writer.writerow(fields)
        self._trajectory = str(trajectory)
        self._frequency = frequency

    def record(
        self,
        phase: str,
        control_cycle: int,
        command_index: int,
        trajectory_index: int,
        trajectory_target: np.ndarray,
        sent_target: np.ndarray,
        measured: np.ndarray,
        init_correction: np.ndarray,
    ) -> None:
        trajectory_target = np.asarray(trajectory_target, dtype=np.float64)
        sent_target = np.asarray(sent_target, dtype=np.float64)
        measured = np.asarray(measured, dtype=np.float64)
        init_correction = np.asarray(init_correction, dtype=np.float64)
        arrays = (trajectory_target, sent_target, measured, init_correction)
        if any(array.shape != (7,) or not np.isfinite(array).all() for array in arrays):
            raise ValueError("trajectory recorder received an invalid 7-D joint vector")
        row = [
            time.time(), time.monotonic() - self._started, phase, control_cycle,
            command_index, trajectory_index, self._trajectory, self._frequency,
        ]
        for values in (
            trajectory_target,
            sent_target,
            measured,
            trajectory_target - measured,
            sent_target - measured,
            init_correction,
        ):
            row.extend(values.tolist())
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % 100 == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()
            print(f"[RECORD] 已保存 {self._rows} 个控制周期：{self.path}")


class EStop:
    def __init__(self) -> None:
        self.latched = False
        self.reason = ""

    def trigger(self, reason: str) -> None:
        if not self.latched:
            self.latched = True
            self.reason = reason
            print(
                f"\n[E-STOP] 已锁存：{reason}；状态有效时保持实测角，"
                "状态失联时双臂切换零位置增益阻尼",
                flush=True,
            )


def read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1).lower()
    return None


def minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def resolve_trajectory(episode: int, explicit: Path | None) -> Path:
    if episode < 0:
        raise ValueError("episode must be non-negative")
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    pattern = f"joints_steps_*_episode_{episode:06d}_full.npz"
    candidates = []
    for path in DEFAULT_RECORDS_DIR.glob(pattern):
        match = re.fullmatch(
            rf"joints_steps_(\d+)_episode_{episode:06d}_full\.npz", path.name
        )
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"no full inference result for episode {episode}: {DEFAULT_RECORDS_DIR / pattern}"
        )
    return max(candidates, key=lambda item: item[0])[1].resolve()


def temporal_ensemble(action_chunks: np.ndarray) -> np.ndarray:
    """Average all past chunks that predict each current episode timestep."""
    episode_length, horizon, _ = action_chunks.shape
    result = np.empty((episode_length, 7), dtype=np.float64)
    for timestep in range(episode_length):
        first_source = max(0, timestep - horizon + 1)
        predictions = [
            action_chunks[source, timestep - source, :7]
            for source in range(first_source, timestep + 1)
        ]
        result[timestep] = np.mean(predictions, axis=0)
    return result


def temporal_ensemble_hand(action_chunks: np.ndarray) -> np.ndarray:
    """Average the right-hand part of all past chunks predicting each timestep."""
    episode_length, horizon, _ = action_chunks.shape
    result = np.empty((episode_length, 6), dtype=np.float64)
    for timestep in range(episode_length):
        first_source = max(0, timestep - horizon + 1)
        predictions = [
            action_chunks[source, timestep - source, 7:13]
            for source in range(first_source, timestep + 1)
        ]
        result[timestep] = np.mean(predictions, axis=0)
    return result


def load_full_episode(path: Path, aggregation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path) as data:
        required = {"timestamps", "input_states", "first_predicted_actions"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ missing arrays: {sorted(missing)}")
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        input_states = np.asarray(data["input_states"], dtype=np.float64)
        chunks = (
            np.asarray(data["predicted_action_chunks"], dtype=np.float64)
            if "predicted_action_chunks" in data.files
            else None
        )
        first_actions = np.asarray(data["first_predicted_actions"], dtype=np.float64)

    episode_length = len(timestamps)
    if input_states.shape != (episode_length, 13):
        raise ValueError(f"input_states must be [T,13], got {input_states.shape}")
    if chunks is not None and chunks.shape != (episode_length, 16, 13):
        raise ValueError(f"predicted_action_chunks must be [T,16,13], got {chunks.shape}")
    if first_actions.shape != (episode_length, 13):
        raise ValueError(f"first_predicted_actions must be [T,13], got {first_actions.shape}")
    if episode_length < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must contain at least two strictly increasing values")
    arrays = (timestamps, input_states, first_actions) + (() if chunks is None else (chunks,))
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("trajectory contains NaN or Inf")

    if aggregation == "temporal-ensemble":
        if chunks is None:
            raise ValueError("temporal-ensemble requires predicted_action_chunks")
        targets = temporal_ensemble(chunks)
    else:
        targets = first_actions[:, :7].copy()
    initial_pose = input_states[0, :7].copy()
    if np.any(targets < RIGHT_ARM_LOWER) or np.any(targets > RIGHT_ARM_UPPER):
        raise ValueError("predicted right-arm targets exceed G1 URDF joint limits")
    if np.any(initial_pose < RIGHT_ARM_LOWER) or np.any(initial_pose > RIGHT_ARM_UPPER):
        raise ValueError("episode0 initial pose exceeds G1 URDF joint limits")

    metadata = {}
    summary_path = path.with_suffix(".json")
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    return initial_pose, targets, timestamps, metadata


def load_hand_trajectory(path: Path, aggregation: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        input_states = np.asarray(data["input_states"], dtype=np.float64)
        chunks = np.asarray(data["predicted_action_chunks"], dtype=np.float64)
        first_actions = np.asarray(data["first_predicted_actions"], dtype=np.float64)
    targets = (
        temporal_ensemble_hand(chunks)
        if aggregation == "temporal-ensemble"
        else first_actions[:, 7:13].copy()
    )
    initial_state = input_states[0, 7:13].copy()
    if not np.isfinite(targets).all() or np.any(targets < 0.0) or np.any(targets > 1.0):
        raise ValueError("predicted right-hand targets must be finite normalized values in [0, 1]")
    return initial_state, targets


def load_episode_initials(
    dataset_root: Path,
    episode: int,
    inferred_right_pose: np.ndarray,
    inferred_right_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Read corresponding raw first-frame arm/hand states and hand commands."""
    import pandas as pd

    dataset_root = dataset_root.expanduser().resolve()
    matches = sorted(dataset_root.glob(f"data/chunk-*/episode_{episode:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one raw parquet for episode {episode}, found {len(matches)} under {dataset_root}"
        )
    parquet = matches[0]
    frame = pd.read_parquet(parquet, columns=["observation.state", "action.wbc"])
    if frame.empty:
        raise ValueError(f"raw episode contains no frames: {parquet}")
    state = np.asarray(frame.iloc[0]["observation.state"], dtype=np.float64)
    action = np.asarray(frame.iloc[0]["action.wbc"], dtype=np.float64)
    if state.shape != (41,) or action.shape != (41,) or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"expected finite 41-D observation.state/action.wbc, got {state.shape}/{action.shape}")
    left_pose = state[15:22].copy()
    right_pose = state[22:29].copy()
    left_hand_state = state[29:35].copy()
    right_hand_state = state[35:41].copy()
    left_hand_command = action[29:35].copy()
    right_hand_command = action[35:41].copy()
    if not np.allclose(right_pose, inferred_right_pose, atol=1e-5, rtol=0.0):
        mismatch = float(np.max(np.abs(right_pose - inferred_right_pose)))
        raise ValueError(
            f"raw/inference episode mismatch: initial right-arm difference is {mismatch:.6f} rad"
        )
    if not np.allclose(right_hand_state, inferred_right_hand, atol=1e-5, rtol=0.0):
        mismatch = float(np.max(np.abs(right_hand_state - inferred_right_hand)))
        raise ValueError(
            f"raw/inference episode mismatch: initial right-hand difference is {mismatch:.6f}"
        )
    if np.any(left_pose < LEFT_ARM_LOWER) or np.any(left_pose > LEFT_ARM_UPPER):
        raise ValueError("raw initial left-arm pose exceeds G1 URDF joint limits")
    for name, values in (
        ("left hand state", left_hand_state),
        ("right hand state", right_hand_state),
        ("left hand command", left_hand_command),
        ("right hand command", right_hand_command),
    ):
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"raw {name} must be normalized to [0, 1]")
    return (
        left_pose,
        right_pose,
        left_hand_state,
        right_hand_state,
        left_hand_command,
        right_hand_command,
        parquet,
    )


def build_rate_limited_commands(
    initial_pose: np.ndarray,
    targets: np.ndarray,
    target_dt: float,
    control_frequency: float,
    max_speed: float,
) -> tuple[np.ndarray, float]:
    """Simulate the exact target follower used for real-robot command playback."""
    control_dt = 1.0 / control_frequency
    intervals = int(np.ceil((len(targets) - 1) * target_dt * control_frequency))
    commands = np.empty((intervals + 1, initial_pose.size), dtype=np.float64)
    command = initial_pose.copy()
    commands[0] = command
    maximum_tracking_error = 0.0
    max_step = max_speed * control_dt
    for step in range(1, intervals + 1):
        elapsed = step * control_dt
        target_index = min(int(elapsed / target_dt), len(targets) - 1)
        target = targets[target_index]
        command += np.clip(target - command, -max_step, max_step)
        commands[step] = command
        maximum_tracking_error = max(maximum_tracking_error, float(np.max(np.abs(target - command))))
    return commands, maximum_tracking_error


def validate_commands(commands: np.ndarray, frequency: float, max_speed: float) -> float:
    if not np.isfinite(commands).all():
        raise ValueError("generated command trajectory contains NaN or Inf")
    if np.any(commands < RIGHT_ARM_LOWER) or np.any(commands > RIGHT_ARM_UPPER):
        raise ValueError("generated command trajectory exceeds G1 URDF joint limits")
    peak_speed = float(np.max(np.abs(np.diff(commands, axis=0))) * frequency)
    if peak_speed > max_speed + 1e-9:
        raise ValueError(f"generated commands require {peak_speed:.6f} rad/s")
    return peak_speed


def validate_hand_commands(commands: np.ndarray, frequency: float, max_speed: float) -> float:
    if commands.ndim != 2 or commands.shape[1] != 6 or not np.isfinite(commands).all():
        raise ValueError(f"generated Inspire commands must be finite [T,6], got {commands.shape}")
    if np.any(commands < 0.0) or np.any(commands > 1.0):
        raise ValueError("generated Inspire commands exceed normalized limits [0, 1]")
    peak_speed = float(np.max(np.abs(np.diff(commands, axis=0))) * frequency)
    if peak_speed > max_speed + 1e-9:
        raise ValueError(f"generated Inspire commands require {peak_speed:.6f} normalized units/s")
    return peak_speed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0, help="episode selected from inference_records")
    parser.add_argument("--trajectory", type=Path, default=None, help="explicit NPZ overrides --episode lookup")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help="raw collected dataset used to load both arms' episode initial pose",
    )
    parser.add_argument("--aggregation", choices=("temporal-ensemble", "first"), default="temporal-ensemble")
    parser.add_argument("--slowdown", type=float, default=2.0, help="2.0 means two-times slower than dataset time")
    parser.add_argument("--network-interface", default="enp7s0")
    parser.add_argument("--frequency", type=float, default=100.0, help="DDS command frequency in Hz")
    parser.add_argument("--max-speed", type=float, default=0.25, help="hard command limit in rad/s per joint")
    parser.add_argument("--max-hand-speed", type=float, default=0.5, help="right-hand limit in normalized units/s")
    parser.add_argument("--hand-frequency", type=float, default=10.0, help="rt/inspire/cmd publish frequency in Hz")
    parser.add_argument("--lowstate-warning-age", type=float, default=0.2)
    parser.add_argument("--lowstate-timeout", type=float, default=0.5)
    parser.add_argument("--lowstate-warning-interval", type=float, default=2.0)
    parser.add_argument("--estop-damping-duration", type=float, default=10.0)
    parser.add_argument("--initial-duration", type=float, default=8.0)
    parser.add_argument("--initial-speed", type=float, default=0.1, help="initialization limit in rad/s")
    parser.add_argument("--initial-tolerance", type=float, default=0.05, help="READY tolerance in rad")
    parser.add_argument("--initial-hand-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--enable-initial-outer-loop-compensation",
        action="store_true",
        help="enable initialization/READY integral position-offset compensation; disabled by default",
    )
    parser.add_argument(
        "--initial-correction-rate", type=float, default=1.0,
        help="post-interpolation outer-loop correction rate in 1/s",
    )
    parser.add_argument(
        "--initial-correction-speed", type=float, default=0.03,
        help="maximum correction-offset change speed in rad/s",
    )
    parser.add_argument(
        "--initial-correction-limit", type=float, default=0.15,
        help="maximum outer-loop position offset per joint in rad",
    )
    parser.add_argument(
        "--initial-correction-deadband", type=float, default=0.003,
        help="do not integrate smaller initialization errors",
    )
    parser.add_argument(
        "--initial-stable-duration", "--stable-duration", dest="initial_stable_duration",
        type=float, default=1.0, help="required continuous in-tolerance time before READY",
    )
    parser.add_argument(
        "--lower-body-mode", choices=("damping", "zero-torque"), default="damping"
    )
    parser.add_argument(
        "--no-gravity-compensation", action="store_true",
        help="disable the default full-time dual-arm Pinocchio/RNEA gravity feed-forward",
    )
    parser.add_argument(
        "--gravity-urdf", type=Path,
        default=Path(__file__).resolve().parents[3]
        / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/g1_29dof.urdf",
    )
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="gravity feed-forward scale in [0,1]")
    parser.add_argument(
        "--gravity-hand-model", choices=("rh56e2", "rh56dftp", "rubber"), default="rh56e2",
        help="end-effector inertial model; defaults to the installed 0.790 kg RH56E2 hand",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=DEFAULT_RECORDS_DIR / "trajectory_tracking",
        help="directory for per-control-cycle target/measured CSV logs",
    )
    parser.add_argument("--no-record", action="store_true", help="disable target/measured CSV recording")
    parser.add_argument("--arm", action="store_true", help="actually release motion mode and publish rt/lowcmd")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if min(
        args.slowdown,
        args.frequency,
        args.max_speed,
        args.max_hand_speed,
        args.hand_frequency,
        args.initial_duration,
        args.initial_speed,
        args.initial_tolerance,
        args.initial_hand_tolerance,
    ) <= 0.0:
        raise SystemExit("all slowdown, timing, speed, and tolerance arguments must be positive")
    if min(
        args.initial_correction_rate,
        args.initial_correction_speed,
        args.initial_correction_limit,
        args.initial_correction_deadband,
        args.initial_stable_duration,
    ) < 0.0:
        raise SystemExit("initial correction and stable-duration arguments must be non-negative")
    if args.initial_correction_deadband >= args.initial_tolerance:
        raise SystemExit("--initial-correction-deadband must be smaller than --initial-tolerance")
    if not 0.0 <= args.gravity_scale <= 1.0:
        raise SystemExit("--gravity-scale must be in [0, 1]")
    if min(
        args.lowstate_warning_age,
        args.lowstate_timeout,
        args.lowstate_warning_interval,
        args.estop_damping_duration,
    ) <= 0.0:
        raise SystemExit("LowState watchdog and e-stop damping parameters must be positive")
    if args.lowstate_warning_age >= args.lowstate_timeout:
        raise SystemExit("--lowstate-warning-age must be smaller than --lowstate-timeout")
    hand_stride = round(args.frequency / args.hand_frequency)
    if args.hand_frequency > args.frequency or not np.isclose(
        hand_stride * args.hand_frequency, args.frequency
    ):
        raise SystemExit("--frequency must be an integer multiple of --hand-frequency")

    try:
        trajectory_path = resolve_trajectory(args.episode, args.trajectory)
        initial_pose, targets, timestamps, metadata = load_full_episode(
            trajectory_path, args.aggregation
        )
        inferred_right_hand, hand_targets = load_hand_trajectory(
            trajectory_path, args.aggregation
        )
        (
            initial_left_pose,
            initial_pose,
            initial_left_hand_state,
            initial_right_hand_state,
            initial_left_hand_command,
            initial_right_hand_command,
            initial_parquet,
        ) = load_episode_initials(
            args.dataset, args.episode, initial_pose, inferred_right_hand
        )
        source_dt = float(np.median(np.diff(timestamps)))
        target_dt = source_dt * args.slowdown
        raw_peak_speed = float(np.max(np.abs(np.diff(targets, axis=0))) / target_dt)
        commands, max_tracking_error = build_rate_limited_commands(
            initial_pose, targets, target_dt, args.frequency, args.max_speed
        )
        command_peak_speed = validate_commands(commands, args.frequency, args.max_speed)
        hand_commands, hand_tracking_error = build_rate_limited_commands(
            initial_right_hand_command,
            hand_targets,
            target_dt,
            args.frequency,
            args.max_hand_speed,
        )
        hand_peak_speed = validate_hand_commands(
            hand_commands, args.frequency, args.max_hand_speed
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"trajectory validation failed: {exc}") from exc

    source_duration = float(timestamps[-1] - timestamps[0])
    playback_duration = (len(commands) - 1) / args.frequency
    print(f"trajectory: {trajectory_path}")
    print(f"checkpoint: {metadata.get('checkpoint', 'unknown')}")
    print(f"episode targets: {len(targets)} x 7, aggregation={args.aggregation}")
    print(f"dataset: dt={source_dt:.4f}s duration={source_duration:.2f}s")
    print(f"playback: slowdown={args.slowdown:.2f}x dt={target_dt:.4f}s duration={playback_duration:.2f}s")
    print(f"raw target peak speed: {raw_peak_speed:.3f} rad/s")
    print(f"rate-limited command peak speed: {command_peak_speed:.3f}/{args.max_speed:.3f} rad/s")
    print(f"maximum target tracking lag: {max_tracking_error:.3f} rad")
    print(f"100 Hz commands: {len(commands)} x 7")
    print(
        f"right-hand peak speed: {hand_peak_speed:.3f}/{args.max_hand_speed:.3f} normalized/s; "
        f"maximum tracking lag: {hand_tracking_error:.3f}"
    )
    print(f"right-hand commands: {len(hand_commands)} x 6; DDS publish={args.hand_frequency:.1f} Hz")
    print(f"episode {args.episode} input pose: {np.round(initial_pose, 4)}")
    print(f"raw initial state: {initial_parquet}")
    print(f"episode {args.episode} left input pose: {np.round(initial_left_pose, 4)}")
    print(f"episode {args.episode} right/left hand states: {np.round(initial_right_hand_state, 4)} / {np.round(initial_left_hand_state, 4)}")
    print(f"episode {args.episode} right/left hand commands: {np.round(initial_right_hand_command, 4)} / {np.round(initial_left_hand_command, 4)}")
    print(f"final command: {np.round(commands[-1], 4)}")
    print(
        "初始化外环位置补偿: "
        f"{'启用' if args.enable_initial_outer_loop_compensation else '关闭（默认）'}"
    )
    if not args.arm:
        print("[DRY RUN] 最终下发序列检查通过；未连接 DDS，也未发送任何命令")
        return
    if not sys.stdin.isatty():
        raise SystemExit("真机模式必须在交互终端运行，确保 SPACE/Q 急停可用")

    from g1_joint_client import (
        G1DDS,
        LEFT_ARM_MOTORS,
        RIGHT_ARM_MOTORS,
        _update_pose_correction,
    )

    robot = G1DDS(
        args.network_interface,
        args.lower_body_mode,
        gravity_compensation=not args.no_gravity_compensation,
        gravity_urdf=args.gravity_urdf,
        gravity_scale=args.gravity_scale,
        gravity_hand_model=args.gravity_hand_model,
    )
    recorder = None
    if not args.no_record:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        record_path = args.record_dir.expanduser().resolve() / (
            f"{trajectory_path.stem}_{args.aggregation}_{timestamp}.csv"
        )
        recorder = TrajectoryTrackingRecorder(record_path, trajectory_path, args.frequency)
        print(f"[RECORD] 将记录轨迹目标、实际下发目标和实测关节角：{record_path}")

    def read_robot_state() -> np.ndarray:
        return robot.state(
            args.lowstate_timeout,
            warning_age=args.lowstate_warning_age,
            warning_interval=args.lowstate_warning_interval,
        )

    estop = EStop()
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    period = 1.0 / args.frequency
    enabled = False
    phase = "DISARMED"
    init_start = np.zeros(7)
    init_start_left = np.zeros(7)
    init_start_right_hand = np.zeros(6)
    init_start_left_hand = np.zeros(6)
    init_start_time = 0.0
    init_duration = args.initial_duration
    stable_since = None
    last_init_status_time = 0.0
    left_init_correction = np.zeros(7, dtype=np.float64)
    right_init_correction = np.zeros(7, dtype=np.float64)
    left_hold = initial_left_pose.copy()
    command_index = 0
    previous_command = commands[0].copy()
    right_hand_command = initial_right_hand_command.copy()
    left_hand_command = initial_left_hand_command.copy()
    cycle_index = 0

    print("机器人必须由可靠吊架完全承重。ENTER=解锁并初始化，READY 后 L=播放，SPACE/Q=急停。")
    try:
        while not estop.latched:
            cycle_start = time.monotonic()
            key = read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break

            measured = read_robot_state()
            left_q = measured[LEFT_ARM_MOTORS]
            right_q = measured[RIGHT_ARM_MOTORS]
            if key in ("\n", "\r") and phase == "DISARMED":
                init_start_right_hand, init_start_left_hand = robot.hand_state(0.5)
                robot.enter_low_level(measured)
                enabled = True
                init_start = right_q.copy()
                init_start_left = left_q.copy()
                distance = max(
                    float(np.max(np.abs(initial_pose - init_start))),
                    float(np.max(np.abs(initial_left_pose - init_start_left))),
                )
                init_duration = max(args.initial_duration, 1.875 * distance / args.initial_speed)
                init_start_time = time.monotonic()
                stable_since = None
                left_init_correction.fill(0.0)
                right_init_correction.fill(0.0)
                phase = "INITIALIZING"
                print(f"[ARMED] 双臂开始移动到 episode {args.episode} 输入姿态，预计 {init_duration:.1f}s")

            if phase == "INITIALIZING":
                progress = (time.monotonic() - init_start_time) / init_duration
                target = minimum_jerk(init_start, initial_pose, progress)
                left_target = minimum_jerk(init_start_left, initial_left_pose, progress)
                right_hand_command = minimum_jerk(
                    init_start_right_hand, initial_right_hand_command, progress
                )
                left_hand_command = minimum_jerk(
                    init_start_left_hand, initial_left_hand_command, progress
                )
                if progress >= 1.0:
                    left_error = initial_left_pose - left_q
                    right_error = initial_pose - right_q
                    if args.enable_initial_outer_loop_compensation:
                        left_init_correction = _update_pose_correction(
                            left_init_correction,
                            left_error,
                            period,
                            args.initial_correction_rate,
                            args.initial_correction_speed,
                            args.initial_correction_limit,
                            args.initial_correction_deadband,
                        )
                        right_init_correction = _update_pose_correction(
                            right_init_correction,
                            right_error,
                            period,
                            args.initial_correction_rate,
                            args.initial_correction_speed,
                            args.initial_correction_limit,
                            args.initial_correction_deadband,
                        )
                    left_target = np.clip(
                        initial_left_pose + left_init_correction, LEFT_ARM_LOWER, LEFT_ARM_UPPER
                    )
                    target = np.clip(
                        initial_pose + right_init_correction, RIGHT_ARM_LOWER, RIGHT_ARM_UPPER
                    )
                    error = max(
                        float(np.max(np.abs(right_error))),
                        float(np.max(np.abs(left_error))),
                    )
                    right_hand_measured, left_hand_measured = robot.hand_state(0.5)
                    hand_error = max(
                        float(np.max(np.abs(initial_right_hand_state - right_hand_measured))),
                        float(np.max(np.abs(initial_left_hand_state - left_hand_measured))),
                    )
                    if error <= args.initial_tolerance and hand_error <= args.initial_hand_tolerance:
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= args.initial_stable_duration:
                            phase = "READY"
                            print(f"[READY] max_error={error:.4f} rad；检查现场后按 L 连续播放完整轨迹")
                    else:
                        stable_since = None
                    now = time.monotonic()
                    if now - last_init_status_time >= 1.0:
                        stable_duration = 0.0 if stable_since is None else now - stable_since
                        print(
                            f"[INITIALIZING] max_error={error:.4f}rad | "
                            f"hand_error={hand_error:.4f} | "
                            f"L-wrist err={np.round(left_error[4:7], 4)} | "
                            f"R-wrist err={np.round(right_error[4:7], 4)} | "
                            f"correction_max={max(np.max(np.abs(left_init_correction)), np.max(np.abs(right_init_correction))):.4f}rad | "
                            f"stable={stable_duration:.1f}/{args.initial_stable_duration:.1f}s"
                        )
                        last_init_status_time = now
                robot.send_arms(left_target, target)
                if recorder is not None:
                    recorder.record(
                        "INITIALIZING", cycle_index, -1, 0,
                        initial_pose, target, right_q, right_init_correction,
                    )
                if cycle_index % hand_stride == 0:
                    robot.send_inspire_hands(right_hand_command, left_hand_command)

            elif phase == "READY":
                left_error = initial_left_pose - left_q
                right_error = initial_pose - right_q
                if args.enable_initial_outer_loop_compensation:
                    left_init_correction = _update_pose_correction(
                        left_init_correction, left_error, period, args.initial_correction_rate,
                        args.initial_correction_speed, args.initial_correction_limit,
                        args.initial_correction_deadband,
                    )
                    right_init_correction = _update_pose_correction(
                        right_init_correction, right_error, period, args.initial_correction_rate,
                        args.initial_correction_speed, args.initial_correction_limit,
                        args.initial_correction_deadband,
                    )
                left_hold = np.clip(
                    initial_left_pose + left_init_correction, LEFT_ARM_LOWER, LEFT_ARM_UPPER
                )
                right_hold = np.clip(
                    initial_pose + right_init_correction, RIGHT_ARM_LOWER, RIGHT_ARM_UPPER
                )
                robot.send_arms(left_hold, right_hold)
                if recorder is not None:
                    recorder.record(
                        "READY", cycle_index, -1, 0,
                        initial_pose, right_hold, right_q, right_init_correction,
                    )
                if cycle_index % hand_stride == 0:
                    robot.send_inspire_hands(right_hand_command, left_hand_command)
                if key == "l":
                    command_index = 0
                    previous_command = commands[0].copy()
                    right_hand_command = hand_commands[0].copy()
                    phase = "PLAYING"
                    print(f"[PLAYING] 开始连续播放 {playback_duration:.2f}s 的完整右臂和右手轨迹")

            elif phase == "PLAYING":
                played_command_index = command_index
                command = commands[played_command_index]
                right_hand_command = hand_commands[played_command_index]
                elapsed = played_command_index / args.frequency
                trajectory_index = min(int(elapsed / target_dt), len(targets) - 1)
                trajectory_target = targets[trajectory_index]
                runtime_speed = float(np.max(np.abs(command - previous_command)) * args.frequency)
                if runtime_speed > args.max_speed + 1e-9:
                    raise RuntimeError(f"runtime command speed guard triggered: {runtime_speed:.6f} rad/s")
                robot.send_arms(left_hold, command)
                if recorder is not None:
                    recorder.record(
                        "PLAYING", cycle_index, played_command_index, trajectory_index,
                        trajectory_target, command, right_q, right_init_correction,
                    )
                if cycle_index % hand_stride == 0:
                    robot.send_inspire_hands(right_hand_command, left_hand_command)
                previous_command = command.copy()
                command_index += 1
                if command_index >= len(commands):
                    phase = "HOLDING"
                    print("[DONE] 完整右臂和右手轨迹播放完成；保持末姿态，按 SPACE/Q 停止")

            elif phase == "HOLDING":
                robot.send_arms(left_hold, commands[-1])
                if recorder is not None:
                    recorder.record(
                        "HOLDING", cycle_index, len(commands) - 1, len(targets) - 1,
                        targets[-1], commands[-1], right_q, right_init_correction,
                    )
                if cycle_index % hand_stride == 0:
                    robot.send_inspire_hands(hand_commands[-1], left_hand_command)

            cycle_index += 1
            time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            if enabled:
                state_is_stale = "lowstate missing or stale" in estop.reason.lower()
                measured = None
                if not state_is_stale:
                    try:
                        measured = read_robot_state()
                    except RuntimeError:
                        state_is_stale = True
                if state_is_stale:
                    robot.enter_emergency_damping()
                    print(
                        f"[E-STOP] LowState 不可用：持续发送双臂阻尼命令 "
                        f"{args.estop_damping_duration:.1f}s 后退出",
                        flush=True,
                    )
                    deadline = time.monotonic() + args.estop_damping_duration
                    while time.monotonic() < deadline:
                        time.sleep(0.05)
                else:
                    held_left = measured[LEFT_ARM_MOTORS].copy()
                    held_right = measured[RIGHT_ARM_MOTORS].copy()
                    for _ in range(20):
                        robot.send_arms(held_left, held_right)
                        robot.send_inspire_hands(right_hand_command, left_hand_command)
                        time.sleep(0.01)
        finally:
            robot.stop_low_level_publisher()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
            if recorder is not None:
                recorder.close()


if __name__ == "__main__":
    main()
