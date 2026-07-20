"""Safely replay a saved DiT4DiT right-arm trajectory on a suspended G1.

The Inspire hand is intentionally not commanded here.  Run the existing
inspire_modbus_hand.py DDS bridge independently when hand motion is required.
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

DEFAULT_TRAJECTORY = (
    Path(__file__).resolve().parents[3]
    / "inference_records"
    / "joints_steps_52000_episode_000000_result.json"
)
# Unitree G1 29-DoF right arm: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
# These are the same URDF limits enforced by g1_joint_client.py.
RIGHT_ARM_LOWER = np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
RIGHT_ARM_UPPER = np.array([2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])


class EStop:
    def __init__(self) -> None:
        self.latched = False

    def trigger(self, reason: str) -> None:
        if not self.latched:
            self.latched = True
            print(f"\n[E-STOP] 已锁存：{reason}；保持当前实测关节角", flush=True)


def read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1).lower()
    return None


def minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)

    actions = np.asarray(record.get("actions"), dtype=np.float64)
    state = np.asarray(record.get("state"), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 7:
        raise ValueError(f"actions must have shape [T, >=7], got {actions.shape}")
    if state.ndim != 1 or state.size < 7:
        raise ValueError(f"state must contain at least 7 values, got {state.shape}")
    right_arm = actions[:, :7]
    initial_right_arm = state[:7]
    if not np.isfinite(right_arm).all() or not np.isfinite(initial_right_arm).all():
        raise ValueError("trajectory contains NaN or Inf")
    if np.any(right_arm < RIGHT_ARM_LOWER) or np.any(right_arm > RIGHT_ARM_UPPER):
        raise ValueError("trajectory exceeds G1 right-arm URDF joint limits")
    if np.any(initial_right_arm < RIGHT_ARM_LOWER) or np.any(initial_right_arm > RIGHT_ARM_UPPER):
        raise ValueError("episode0 input state exceeds G1 right-arm URDF joint limits")
    return initial_right_arm, right_arm, record


def check_trajectory_speed(
    initial_right_arm: np.ndarray,
    trajectory: np.ndarray,
    action_dt: float,
    max_speed: float,
) -> float:
    segments = np.diff(np.vstack((initial_right_arm, trajectory)), axis=0)
    peak_speed = float(np.max(np.abs(segments)) / action_dt)
    if peak_speed > max_speed + 1e-9:
        raise ValueError(
            f"trajectory requires {peak_speed:.3f} rad/s, above --max-speed "
            f"{max_speed:.3f} rad/s; increase --action-dt rather than bypassing the limit"
        )
    return peak_speed


def interpolate_segment(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    """Linearly interpolate so segment velocity equals the preflight-checked velocity."""
    return start + np.clip(progress, 0.0, 1.0) * (goal - start)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--network-interface", default="enp7s0")
    parser.add_argument("--frequency", type=float, default=100.0, help="DDS control frequency in Hz")
    parser.add_argument("--action-dt", type=float, default=0.4, help="seconds between saved action targets")
    parser.add_argument("--max-speed", type=float, default=0.25, help="hard per-joint playback limit in rad/s")
    parser.add_argument("--initial-duration", type=float, default=5.0)
    parser.add_argument("--initial-speed", type=float, default=0.15, help="initialization limit in rad/s")
    parser.add_argument("--initial-tolerance", type=float, default=0.03, help="READY tolerance in rad")
    parser.add_argument("--stable-duration", type=float, default=1.0)
    parser.add_argument(
        "--lower-body-mode",
        choices=("damping", "zero-torque"),
        default="damping",
        help="damping is the only recommended mode for this suspended playback",
    )
    parser.add_argument("--arm", action="store_true", help="actually release motion mode and publish rt/lowcmd")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if min(
        args.frequency,
        args.action_dt,
        args.max_speed,
        args.initial_duration,
        args.initial_speed,
        args.initial_tolerance,
        args.stable_duration,
    ) <= 0.0:
        raise SystemExit("all timing, speed, and tolerance arguments must be positive")

    trajectory_path = args.trajectory.expanduser().resolve()
    try:
        initial_pose, trajectory, record = load_trajectory(trajectory_path)
        peak_speed = check_trajectory_speed(initial_pose, trajectory, args.action_dt, args.max_speed)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"trajectory validation failed: {exc}") from exc

    print(f"trajectory: {trajectory_path}")
    print(f"checkpoint: {record.get('checkpoint', 'unknown')}")
    print(f"right-arm targets: {len(trajectory)} x 7, action_dt={args.action_dt:.3f}s")
    print(f"preflight peak segment speed: {peak_speed:.3f} rad/s")
    print(f"episode0 input pose: {np.round(initial_pose, 4)}")
    print(f"first target: {np.round(trajectory[0], 4)}")
    print(f"last target:  {np.round(trajectory[-1], 4)}")
    if not args.arm:
        print("[DRY RUN] validation passed; no DDS connection was opened and no command was sent")
        return
    if not sys.stdin.isatty():
        raise SystemExit("真机模式必须在交互终端运行，确保 SPACE/Q 急停可用")

    # Keep offline validation independent of robot-only dependencies such as pyzmq.
    from g1_joint_client import G1DDS, RIGHT_ARM_MOTORS

    robot = G1DDS(args.network_interface, args.lower_body_mode)
    estop = EStop()
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    period = 1.0 / args.frequency
    enabled = False
    phase = "DISARMED"
    init_start = np.zeros(7)
    init_start_time = 0.0
    init_duration = args.initial_duration
    stable_since = None
    segment_index = 0
    segment_start = initial_pose.copy()
    segment_start_time = 0.0
    final_target = initial_pose.copy()

    print("机器人必须由可靠吊架完全承重。ENTER=解锁并初始化，READY 后 L=播放，SPACE/Q=急停。")
    try:
        while not estop.latched:
            cycle_start = time.monotonic()
            key = read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break

            measured = robot.state(0.2)
            right_q = measured[RIGHT_ARM_MOTORS]
            if key in ("\n", "\r") and phase == "DISARMED":
                robot.enter_low_level(measured)
                enabled = True
                init_start = right_q.copy()
                distance = float(np.max(np.abs(initial_pose - init_start)))
                # Minimum-jerk peak speed is 1.875 * distance / duration.
                init_duration = max(args.initial_duration, 1.875 * distance / args.initial_speed)
                init_start_time = time.monotonic()
                phase = "INITIALIZING"
                print(f"[ARMED] 开始移动到 episode0 输入姿态，预计 {init_duration:.1f}s")

            if phase == "INITIALIZING":
                progress = (time.monotonic() - init_start_time) / init_duration
                target = minimum_jerk(init_start, initial_pose, progress)
                robot.send_right_arm(target, measured)
                if progress >= 1.0:
                    error = float(np.max(np.abs(initial_pose - right_q)))
                    if error <= args.initial_tolerance:
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= args.stable_duration:
                            phase = "READY"
                            print(f"[READY] max_error={error:.4f} rad；检查现场后按 L 播放")
                    else:
                        stable_since = None

            elif phase == "READY":
                robot.send_right_arm(initial_pose, measured)
                if key == "l":
                    segment_index = 0
                    segment_start = initial_pose.copy()
                    segment_start_time = time.monotonic()
                    phase = "PLAYING"
                    print("[PLAYING] 开始播放右臂轨迹")

            elif phase == "PLAYING":
                goal = trajectory[segment_index]
                progress = (time.monotonic() - segment_start_time) / args.action_dt
                target = interpolate_segment(segment_start, goal, progress)
                # Runtime guard remains active even after the complete preflight check.
                max_step = args.max_speed * period * 1.05
                if np.max(np.abs(target - final_target)) > max_step:
                    raise RuntimeError("runtime right-arm speed guard triggered")
                robot.send_right_arm(target, measured)
                final_target = target.copy()
                if progress >= 1.0:
                    robot.send_right_arm(goal, measured)
                    final_target = goal.copy()
                    segment_index += 1
                    if segment_index >= len(trajectory):
                        phase = "HOLDING"
                        print("[DONE] 轨迹播放完成；保持末姿态，按 SPACE/Q 停止")
                    else:
                        segment_start = goal.copy()
                        segment_start_time += args.action_dt

            elif phase == "HOLDING":
                robot.send_right_arm(trajectory[-1], measured)

            time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            if enabled:
                measured = robot.state(0.2)
                held = measured[RIGHT_ARM_MOTORS].copy()
                for _ in range(20):
                    robot.send_right_arm(held, measured)
                    time.sleep(0.01)
        finally:
            robot.stop_low_level_publisher()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)


if __name__ == "__main__":
    main()
