"""Safely collect single-joint G1 arm data for offline friction identification."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from g1_joint_client import (
    ARM_JOINT_NAMES,
    ARM_KD,
    ARM_KP,
    EStop,
    G1DDS,
    LEFT_ARM_LOWER,
    LEFT_ARM_MOTORS,
    LEFT_ARM_UPPER,
    RIGHT_ARM_LOWER,
    RIGHT_ARM_MOTORS,
    RIGHT_ARM_UPPER,
    _read_key,
)


CONTROL_PERIOD = 0.01
ARM_MOTORS = np.concatenate((LEFT_ARM_MOTORS, RIGHT_ARM_MOTORS))


@dataclass(frozen=True)
class Sample:
    phase: str
    repeat: int
    speed: float
    valid: bool
    q: float
    dq: float


def _quintic(start: float, end: float, duration: float, dt: float):
    count = max(1, int(np.ceil(duration / dt)))
    for index in range(count):
        t = min(index * dt, duration)
        u = t / duration
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
        yield start + (end - start) * blend, (end - start) * blend_dot
    yield end, 0.0


def _constant_velocity_move(start: float, end: float, speed: float, ramp: float, dt: float):
    direction = 1.0 if end > start else -1.0
    distance = abs(end - start)
    velocity = abs(speed)
    ramp_time = min(ramp, distance / velocity)
    ramp_distance = 0.5 * velocity * ramp_time
    constant_distance = max(0.0, distance - 2.0 * ramp_distance)
    constant_time = constant_distance / velocity
    total = 2.0 * ramp_time + constant_time
    count = max(1, int(np.ceil(total / dt)))
    for index in range(count):
        t = min(index * dt, total)
        if t < ramp_time:
            phase = np.pi * t / ramp_time
            local_dq = 0.5 * velocity * (1.0 - np.cos(phase))
            position = 0.5 * velocity * (t - ramp_time * np.sin(phase) / np.pi)
            steady = False
        elif t < ramp_time + constant_time:
            local_dq = velocity
            position = ramp_distance + velocity * (t - ramp_time)
            steady = True
        else:
            decel_t = min(t - ramp_time - constant_time, ramp_time)
            phase = np.pi * decel_t / ramp_time
            local_dq = 0.5 * velocity * (1.0 + np.cos(phase))
            position = ramp_distance + constant_distance + 0.5 * velocity * (
                decel_t + ramp_time * np.sin(phase) / np.pi
            )
            steady = False
        yield start + direction * position, direction * local_dq, steady
    yield end, 0.0, False


def build_trajectory(
    center: float,
    amplitude: float,
    speeds: list[float],
    repeats: int,
    ramp_duration: float,
    dwell_duration: float,
    prepare_duration: float,
    dt: float = CONTROL_PERIOD,
):
    low, high = center - amplitude, center + amplitude
    for q, dq in _quintic(center, low, prepare_duration, dt):
        yield Sample("prepare", -1, 0.0, False, q, dq)
    dwell_samples = max(1, int(np.ceil(dwell_duration / dt)))
    for speed in speeds:
        for repeat in range(repeats):
            for q, dq, valid in _constant_velocity_move(low, high, speed, ramp_duration, dt):
                yield Sample("positive", repeat, speed, valid, q, dq)
            for _ in range(dwell_samples):
                yield Sample("high_dwell", repeat, speed, False, high, 0.0)
            for q, dq, valid in _constant_velocity_move(high, low, speed, ramp_duration, dt):
                yield Sample("negative", repeat, speed, valid, q, dq)
            for _ in range(dwell_samples):
                yield Sample("low_dwell", repeat, speed, False, low, 0.0)
    for q, dq in _quintic(low, center, prepare_duration, dt):
        yield Sample("return", -1, 0.0, False, q, dq)


def _parse_speeds(text: str) -> list[float]:
    try:
        speeds = [float(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("speeds must be comma-separated numbers") from exc
    if not speeds or any(not np.isfinite(value) or value <= 0.0 for value in speeds):
        raise argparse.ArgumentTypeError("all speeds must be finite and positive")
    return speeds


def _csv_header() -> list[str]:
    header = [
        "monotonic_time", "elapsed", "phase", "repeat", "nominal_speed", "valid_steady",
        "torque_measurement_valid",
        "selected_q_cmd", "selected_dq_cmd",
    ]
    for prefix in ("q", "dq", "tau_est"):
        header.extend(f"{prefix}_{index}" for index in range(29))
    header.extend(
        f"temperature_{motor_index}_{sensor_index}"
        for motor_index in range(29)
        for sensor_index in range(2)
    )
    for prefix in ("arm_q_cmd", "arm_dq_cmd", "arm_tau_gravity", "arm_kp", "arm_kd"):
        header.extend(f"{prefix}_{index}" for index in range(14))
    return header


def _dry_run(args) -> None:
    samples = list(
        build_trajectory(0.0, args.amplitude, args.speeds, args.repeats,
                         args.ramp_duration, args.dwell_duration, args.prepare_duration)
    )
    steady = sum(sample.valid for sample in samples)
    print(
        f"[DRY RUN] {len(samples)} samples, {len(samples) * CONTROL_PERIOD:.1f}s, "
        f"steady samples={steady}, q=[{-args.amplitude:.3f}, {args.amplitude:.3f}]rad"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", default="enp7s0")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--joint-index", type=int, choices=range(7), required=True)
    parser.add_argument("--amplitude", type=float, default=0.20, help="half range around initial q [rad]")
    parser.add_argument("--speeds", type=_parse_speeds, default=_parse_speeds("0.03,0.06,0.1,0.2,0.35"))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--ramp-duration", type=float, default=0.8)
    parser.add_argument("--dwell-duration", type=float, default=1.0)
    parser.add_argument("--prepare-duration", type=float, default=4.0)
    parser.add_argument("--kp-scale", type=float, default=0.5)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--gravity-hand-model", choices=("rh56e2", "rh56dftp", "rubber"), default="rh56e2")
    parser.add_argument("--limit-margin", type=float, default=0.10)
    parser.add_argument("--max-tracking-error", type=float, default=0.25)
    parser.add_argument("--max-measured-speed", type=float, default=1.0)
    parser.add_argument("--lowstate-timeout", type=float, default=0.10)
    parser.add_argument("--shutdown-damping-duration", type=float, default=3.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("inference_records/friction_identification"),
    )
    parser.add_argument("--dry-run", action="store_true", help="validate trajectory without connecting to G1")
    args = parser.parse_args()

    positive_values = (
        args.amplitude, args.ramp_duration, args.dwell_duration, args.prepare_duration,
        args.limit_margin, args.max_tracking_error, args.max_measured_speed,
        args.lowstate_timeout, args.shutdown_damping_duration,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values):
        raise SystemExit("trajectory and safety values must be finite and positive")
    if args.repeats < 1 or not 0.0 < args.kp_scale <= 1.0 or not 0.0 < args.kd_scale <= 2.0:
        raise SystemExit("invalid repeats/kp-scale/kd-scale")
    if not 0.0 <= args.gravity_scale <= 1.0:
        raise SystemExit("gravity-scale must be in [0, 1]")
    if args.dry_run:
        _dry_run(args)
        return
    if not sys.stdin.isatty():
        raise SystemExit("必须在交互终端运行，确保 SPACE/Q 急停可用")

    side_offset = 0 if args.side == "left" else 7
    motor_indices = LEFT_ARM_MOTORS if args.side == "left" else RIGHT_ARM_MOTORS
    lower = LEFT_ARM_LOWER if args.side == "left" else RIGHT_ARM_LOWER
    upper = LEFT_ARM_UPPER if args.side == "left" else RIGHT_ARM_UPPER
    selected_motor = int(motor_indices[args.joint_index])
    joint_name = ARM_JOINT_NAMES[side_offset + args.joint_index]
    kp = ARM_KP * args.kp_scale
    kd = ARM_KD * args.kd_scale

    robot = G1DDS(
        args.network_interface,
        lower_body_mode="damping",
        gravity_compensation=True,
        gravity_scale=args.gravity_scale,
        gravity_hand_model=args.gravity_hand_model,
    )
    estop = EStop()
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    enabled = False
    output_path = None
    row_count = 0
    print(
        f"[READY] 辨识 {joint_name} (motor {selected_motor})。机器人必须由吊架可靠承重、"
        "双臂无接触。ENTER=低层接管并保持当前姿态，SPACE/Q/Ctrl-C=急停。"
    )
    try:
        while not enabled and not estop.latched:
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            q, _, _, _ = robot.motor_state(args.lowstate_timeout)
            if key in ("\n", "\r"):
                center = float(q[selected_motor])
                low, high = center - args.amplitude, center + args.amplitude
                safe_low = float(lower[args.joint_index] + args.limit_margin)
                safe_high = float(upper[args.joint_index] - args.limit_margin)
                if low < safe_low or high > safe_high:
                    raise RuntimeError(
                        f"requested [{low:.3f}, {high:.3f}] exceeds safe range "
                        f"[{safe_low:.3f}, {safe_high:.3f}]"
                    )
                robot.enter_low_level(q)
                robot.enter_arm_identification(kp, kd)
                enabled = True
                print(
                    f"[ARMED] center={center:.4f}rad, range=[{low:.4f}, {high:.4f}]rad；"
                    "检查周围空间后按 S 开始自动扫动，SPACE/Q 随时急停。"
                )

        started = False
        while enabled and not started and not estop.latched:
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            robot.motor_state(args.lowstate_timeout)
            if key == "s":
                started = True
            time.sleep(CONTROL_PERIOD)

        if started and not estop.latched:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / f"{args.side}_j{args.joint_index}_{stamp}.csv"
            metadata_path = output_path.with_suffix(".json")
            metadata = vars(args).copy()
            metadata.update(
                joint_name=joint_name, motor_index=selected_motor, center=center,
                lower_target=low, upper_target=high, control_frequency_hz=100.0,
                csv_path=str(output_path),
            )
            metadata["output_dir"] = str(metadata["output_dir"])
            metadata["speeds"] = list(metadata["speeds"])
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
            initial_q, _, _, _ = robot.motor_state(args.lowstate_timeout)
            left_target = initial_q[LEFT_ARM_MOTORS].copy()
            right_target = initial_q[RIGHT_ARM_MOTORS].copy()
            start_time = time.monotonic()
            next_tick = start_time
            with output_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(_csv_header())
                for sample in build_trajectory(
                    center, args.amplitude, args.speeds, args.repeats,
                    args.ramp_duration, args.dwell_duration, args.prepare_duration,
                ):
                    key = _read_key()
                    if key in (" ", "q"):
                        estop.trigger("keyboard")
                        break
                    q, dq, tau_est, temperature = robot.motor_state(args.lowstate_timeout)
                    side_target = left_target if args.side == "left" else right_target
                    side_velocity = np.zeros(7, dtype=np.float64)
                    side_target[args.joint_index] = sample.q
                    side_velocity[args.joint_index] = sample.dq
                    left_velocity = side_velocity if args.side == "left" else np.zeros(7)
                    right_velocity = side_velocity if args.side == "right" else np.zeros(7)
                    robot.send_identification_arms(
                        left_target, right_target, left_velocity, right_velocity
                    )
                    arm_error = np.concatenate((left_target, right_target)) - q[ARM_MOTORS]
                    if np.max(np.abs(arm_error)) > args.max_tracking_error:
                        estop.trigger(
                            f"arm tracking error {np.max(np.abs(arm_error)):.3f}rad exceeds limit"
                        )
                        break
                    if np.max(np.abs(dq[ARM_MOTORS])) > args.max_measured_speed:
                        estop.trigger(
                            f"arm speed {np.max(np.abs(dq[ARM_MOTORS])):.3f}rad/s exceeds limit"
                        )
                        break
                    if q[selected_motor] < safe_low or q[selected_motor] > safe_high:
                        estop.trigger("selected joint crossed software position limit")
                        break
                    command = robot.last_arm_command()
                    if command is None:
                        raise RuntimeError("publisher did not produce an arm command snapshot")
                    now = time.monotonic()
                    writer.writerow([
                        now, now - start_time, sample.phase, sample.repeat, sample.speed,
                        int(sample.valid and np.isfinite(tau_est[selected_motor])),
                        int(np.isfinite(tau_est[selected_motor])), sample.q, sample.dq,
                        *q, *dq, *tau_est, *temperature,
                        *command["q"], *command["dq"], *command["tau_ff"],
                        *np.tile(command["kp"], 2), *np.tile(command["kd"], 2),
                    ])
                    row_count += 1
                    if row_count % 100 == 0:
                        handle.flush()
                        print(
                            f"\r[COLLECTING] phase={sample.phase:<10} speed={sample.speed:.3f} "
                            f"repeat={sample.repeat + 1}/{args.repeats} "
                            f"error={arm_error[side_offset + args.joint_index]:+.4f}rad",
                            end="", flush=True,
                        )
                    next_tick += CONTROL_PERIOD
                    delay = next_tick - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                    elif delay < -0.10:
                        raise RuntimeError(f"control loop overrun {-delay * 1000.0:.1f}ms")
            if not estop.latched:
                print(f"\n[DONE] 已完成采集，保持中心姿态；按 SPACE/Q 进入阻尼并退出。")
                while not estop.latched:
                    key = _read_key()
                    if key in (" ", "q"):
                        estop.trigger("normal completion")
                        break
                    robot.motor_state(args.lowstate_timeout)
                    time.sleep(CONTROL_PERIOD)
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            if enabled:
                robot.enter_emergency_damping()
                print(
                    f"\n[STOP] 重力和位置前馈已清零，发送双臂阻尼命令 "
                    f"{args.shutdown_damping_duration:.1f}s；请托住双臂。"
                )
                deadline = time.monotonic() + args.shutdown_damping_duration
                while time.monotonic() < deadline:
                    time.sleep(0.05)
        finally:
            robot.stop_low_level_publisher()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
    if output_path is not None:
        print(f"[RECORD] {row_count} rows: {output_path}")
    print(f"[EXIT] {estop.reason or 'not armed'}")


if __name__ == "__main__":
    main()
