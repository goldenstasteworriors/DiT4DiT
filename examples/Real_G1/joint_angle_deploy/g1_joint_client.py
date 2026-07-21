"""Run a right-arm joint-angle or wrist-delta DiT4DiT policy on a real Unitree G1.

The model runs on a remote ZMQ server.  This process must run on the computer
connected to the G1 DDS network.  Publishing is disabled unless --arm is set.
SPACE, Q, Ctrl-C, inference timeout, stale lowstate, or a joint safety violation
latches an e-stop which holds the latest measured right-arm pose.
"""

from __future__ import annotations

import argparse
import csv
import os
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    # Running this file directly otherwise exposes only joint_angle_deploy/.
    sys.path.insert(0, str(PROJECT_ROOT))

RIGHT_ARM_MOTORS = np.arange(22, 29)
LEFT_ARM_MOTORS = np.arange(15, 22)
LOWER_BODY_MOTORS = np.arange(0, 15)
POS_STOP_F = 2146000000.0
VEL_STOP_F = 16000.0
# SONICMJ g1_29dof_gear_wbc.yaml gains used by the data-collection control loop.
ARM_KP = np.array([100.0, 100.0, 40.0, 40.0, 20.0, 20.0, 20.0])
ARM_KD = np.array([5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0])
LOWER_BODY_KD = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0])
LEFT_ARM_LOWER = np.array([-3.0892, -1.5882, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
LEFT_ARM_UPPER = np.array([2.6704, 2.2515, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
RIGHT_ARM_LOWER = np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
RIGHT_ARM_UPPER = np.array([2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
# Exact timestamp=3.0 s (frame 150 at 50 Hz) observation.state from training episode 0.
DEFAULT_INITIAL_LEFT_ARM = np.array(
    [0.13316871, 0.16294952, 0.43247473, -0.27756664, -0.15438078, 0.03934421, -0.23063642]
)
DEFAULT_INITIAL_RIGHT_ARM = np.array(
    [-0.36188766, -0.19208317, 0.33666086, -0.45916361, 0.39308259, 0.59385431, -0.44077981]
)
DEFAULT_INITIAL_LEFT_HAND_STATE = np.array([0.99900001, 0.99800003, 0.99800003, 0.99800003, 0.99900001, 0.98299998])
DEFAULT_INITIAL_RIGHT_HAND_STATE = np.array([0.99800003, 1.0, 0.99800003, 0.99800003, 0.99900001, 0.98400003])
DEFAULT_INITIAL_HAND_COMMAND = np.ones(6, dtype=np.float64)
DEFAULT_GRAVITY_URDF = (
    PROJECT_ROOT
    / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/g1_29dof.urdf"
)
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# Lumped RH56DFTP hand properties derived from Unitree's
# g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf with all finger joints neutral.
# The base G1 URDF already contains a 0.170 kg rubber hand on each wrist, so
# adding these positive point-mass equivalents makes its total hand mass and
# first moment exactly match the full 0.8783 kg FTP model. Static RNEA gravity
# depends on mass and first moment, not the rotational inertia about the COM.
RH56DFTP_ADDED_MASS = 0.7083
RH56DFTP_ADDED_COM = {
    "left_wrist_yaw_joint": np.array([0.15747250, -0.00100931, 0.00611444]),
    "right_wrist_yaw_joint": np.array([0.14823332, 0.00155904, 0.00618849]),
}


class ArmGravityCompensator:
    """Pinocchio/RNEA gravity feed-forward matching xrteleoperate's dual-arm control."""

    def __init__(self, urdf_path: Path, scale: float = 1.0, hand_model: str = "rh56dftp"):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "gravity compensation requires pinocchio; run in the documented "
                "decoupled_vla_collection conda environment"
            ) from exc
        path = urdf_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"gravity-compensation URDF not found: {path}")
        if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
            raise ValueError("gravity-compensation scale must be in [0, 1]")
        if hand_model not in ("rh56dftp", "rubber"):
            raise ValueError(f"unsupported gravity hand model: {hand_model}")
        full_model = pin.buildModelFromUrdf(str(path))
        missing = [name for name in ARM_JOINT_NAMES if not full_model.existJointName(name)]
        if missing:
            raise ValueError(f"gravity-compensation URDF is missing arm joints: {missing}")
        if hand_model == "rh56dftp":
            zero_inertia = np.zeros((3, 3), dtype=np.float64)
            for joint_name, com in RH56DFTP_ADDED_COM.items():
                joint_id = full_model.getJointId(joint_name)
                full_model.inertias[joint_id] += pin.Inertia(
                    RH56DFTP_ADDED_MASS, com, zero_inertia
                )
        arm_ids = {full_model.getJointId(name) for name in ARM_JOINT_NAMES}
        locked_ids = [joint_id for joint_id in range(1, full_model.njoints) if joint_id not in arm_ids]
        self._pin = pin
        self._model = pin.buildReducedModel(
            full_model, locked_ids, pin.neutral(full_model)
        )
        self._data = self._model.createData()
        reduced_names = tuple(self._model.names[1:])
        if self._model.nq != 14 or self._model.nv != 14 or reduced_names != ARM_JOINT_NAMES:
            raise ValueError(
                "unexpected reduced G1 arm model: "
                f"nq={self._model.nq}, nv={self._model.nv}, joints={reduced_names}"
            )
        self._scale = float(scale)
        self._effort_limits = np.asarray(self._model.effortLimit, dtype=np.float64)
        print(
            "[GRAVITY] 双臂 Pinocchio/RNEA 重力补偿已启用，"
            f"scale={self._scale:.3f}，hand_model={hand_model}，URDF={path}"
        )

    def compute(self, left_target: np.ndarray, right_target: np.ndarray) -> np.ndarray:
        q = np.concatenate((left_target, right_target)).astype(np.float64, copy=False)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("gravity compensation requires 14 finite arm targets")
        tau = np.asarray(
            self._pin.rnea(self._model, self._data, q, np.zeros(14), np.zeros(14)),
            dtype=np.float64,
        ) * self._scale
        if tau.shape != (14,) or not np.isfinite(tau).all():
            raise RuntimeError("Pinocchio returned invalid gravity-compensation torques")
        return np.clip(tau, -self._effort_limits, self._effort_limits)


def _relative_wrist_state(
    previous_position: np.ndarray,
    previous_rotation: np.ndarray,
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    """Build the exact 16-D (15 values + padding) state used in wrist-delta training."""
    relative_position = previous_rotation.T @ (current_position - previous_position)
    relative_rotation = previous_rotation.T @ current_rotation
    state = np.concatenate(
        [relative_position, relative_rotation[:2].reshape(6), right_hand], axis=0
    ).astype(np.float32)
    return np.pad(state, (0, 1))


def _wrist_actions_to_joint_actions(
    predicted: np.ndarray,
    ik,
    start_position: np.ndarray,
    start_rotation: np.ndarray,
    pelvis_position: np.ndarray,
    pelvis_rotation: np.ndarray,
    execution_horizon: int,
    max_delta_position: float,
    max_delta_orientation: float,
    max_ik_position_residual: float,
    max_ik_orientation_residual: float,
) -> np.ndarray:
    """Integrate local SE(3) deltas and solve the corresponding right-arm joint targets."""
    from examples.PipetteRightOnly.convert_wrist_delta_to_joint_chunks import (
        rotation_6d_to_matrix,
        rotation_error,
    )

    if predicted.ndim != 2 or predicted.shape[1] != 15 or not np.isfinite(predicted).all():
        raise RuntimeError(f"invalid wrist-delta policy output {predicted.shape}")
    horizon = min(execution_horizon, len(predicted))
    if horizon <= 0:
        raise RuntimeError("wrist-delta policy returned an empty action chunk")
    converted = np.empty((horizon, 13), dtype=np.float64)
    position = start_position.copy()
    rotation = start_rotation.copy()
    for index, wrist_action in enumerate(predicted[:horizon]):
        delta_position = wrist_action[:3]
        delta_rotation = rotation_6d_to_matrix(wrist_action[3:9])
        position_norm = float(np.linalg.norm(delta_position))
        orientation_norm = float(np.linalg.norm(rotation_error(delta_rotation, np.eye(3))))
        if position_norm > max_delta_position:
            raise RuntimeError(
                f"wrist delta position {position_norm:.4f}m exceeds {max_delta_position:.4f}m"
            )
        if orientation_norm > max_delta_orientation:
            raise RuntimeError(
                f"wrist delta orientation {orientation_norm:.4f}rad exceeds "
                f"{max_delta_orientation:.4f}rad"
            )
        position = position + rotation @ delta_position
        rotation = rotation @ delta_rotation
        target_position = pelvis_position + pelvis_rotation @ position
        target_rotation = pelvis_rotation @ rotation
        joints, position_residual, orientation_residual, _ = ik.solve(
            target_position, target_rotation
        )
        if position_residual > max_ik_position_residual:
            raise RuntimeError(
                f"IK position residual {position_residual:.5f}m exceeds "
                f"{max_ik_position_residual:.5f}m"
            )
        if orientation_residual > max_ik_orientation_residual:
            raise RuntimeError(
                f"IK orientation residual {orientation_residual:.5f}rad exceeds "
                f"{max_ik_orientation_residual:.5f}rad"
            )
        converted[index, :7] = joints
        converted[index, 7:13] = wrist_action[9:15]
    return converted


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(type(obj).__name__)


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _update_pose_correction(
    correction: np.ndarray,
    error: np.ndarray,
    dt: float,
    integral_gain: float,
    speed_limit: float,
    position_limit: float,
    deadband: float,
) -> np.ndarray:
    """Integrate measured pose error with per-cycle speed and total-offset limits."""
    active_error = np.where(np.abs(error) > deadband, error, 0.0)
    max_step = speed_limit * dt
    delta = np.clip(integral_gain * active_error * dt, -max_step, max_step)
    return np.clip(correction + delta, -position_limit, position_limit)


class PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        data = {"examples": [{"image": [image], "lang": instruction, "state": state[None]}]}
        request = {"endpoint": "get_action", "data": data}
        try:
            self.socket.send(msgpack.packb(request, default=_pack_array))
            response = msgpack.unpackb(self.socket.recv(), object_hook=_unpack_array)
        except zmq.Again as exc:
            raise TimeoutError(
                "ZMQ inference timed out; check A800 server logs, SSH tunnel, and --timeout-ms"
            ) from exc
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error")))
        return np.asarray(response["data"]["unnormalized_actions"], dtype=np.float64)


class ModelIORecorder:
    """Persist exactly one record for each policy request, not each control tick."""

    def __init__(self, root: Path | None, action_space: str):
        self.root = root
        self.action_space = action_space
        self.session_dir = None
        self.step = 0
        self._tracking_file = None
        self._tracking_writer = None

    def _ensure_session(self) -> Path:
        if self.session_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.root.expanduser().resolve() / f"{timestamp}_{os.getpid()}"
            self.session_dir.mkdir(parents=True, exist_ok=False)
            print(f"[RECORD] 模型输入输出保存到 {self.session_dir}")
        return self.session_dir

    def record(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        raw_output: np.ndarray,
        joint_output: np.ndarray | None,
        conversion_error: str = "",
    ) -> None:
        if self.root is None:
            return
        path = self._ensure_session() / f"inference_{self.step:06d}.npz"
        np.savez_compressed(
            path,
            inference_index=np.int64(self.step),
            wall_time_unix=np.float64(time.time()),
            action_space=np.asarray(self.action_space),
            instruction=np.asarray(instruction),
            input_image_rgb=np.array(image, copy=True),
            input_state=np.array(state, copy=True),
            raw_model_output=np.array(raw_output, copy=True),
            joint_action_output=(
                np.empty((0, 13), dtype=np.float64)
                if joint_output is None
                else np.array(joint_output, copy=True)
            ),
            conversion_error=np.asarray(conversion_error),
        )
        self.step += 1

    def record_execution(
        self,
        inference_index: int,
        action_index: int,
        desired_arm: np.ndarray,
        commanded_arm: np.ndarray,
        measured_arm: np.ndarray,
        desired_hand: np.ndarray,
        commanded_hand: np.ndarray,
        measured_hand: np.ndarray,
    ) -> None:
        """Append one row per consumed model action for target/tracking diagnosis."""
        if self.root is None:
            return
        if self._tracking_writer is None:
            path = self._ensure_session() / "execution_tracking.csv"
            self._tracking_file = path.open("w", newline="", encoding="utf-8")
            fields = ["wall_time_unix", "monotonic_time", "inference_index", "action_index"]
            for prefix, width in (
                ("desired_arm", 7),
                ("commanded_arm", 7),
                ("measured_arm", 7),
                ("desired_hand", 6),
                ("commanded_hand", 6),
                ("measured_hand", 6),
            ):
                fields.extend(f"{prefix}_{index}" for index in range(width))
            self._tracking_writer = csv.DictWriter(self._tracking_file, fieldnames=fields)
            self._tracking_writer.writeheader()
            print(f"[RECORD] 控制跟踪保存到 {path}")
        row = {
            "wall_time_unix": f"{time.time():.9f}",
            "monotonic_time": f"{time.monotonic():.9f}",
            "inference_index": inference_index,
            "action_index": action_index,
        }
        for prefix, values in (
            ("desired_arm", desired_arm),
            ("commanded_arm", commanded_arm),
            ("measured_arm", measured_arm),
            ("desired_hand", desired_hand),
            ("commanded_hand", commanded_hand),
            ("measured_hand", measured_hand),
        ):
            row.update(
                {f"{prefix}_{index}": f"{float(value):.9f}" for index, value in enumerate(values)}
            )
        self._tracking_writer.writerow(row)
        self._tracking_file.flush()

    def close(self) -> None:
        if self._tracking_file is not None:
            self._tracking_file.close()
            self._tracking_file = None
            self._tracking_writer = None


class _QuietCameraFrameCounter(int):
    """Keep the SONIC client frame counter while disabling its hard-coded 10-frame print."""

    def __add__(self, other):
        return _QuietCameraFrameCounter(int(self) + other)

    def __mod__(self, other):
        if other == 10:
            return 1
        return int(self) % other


class CameraStream:
    """Continuously capture and optionally display camera frames independently of inference."""

    def __init__(
        self,
        source: str,
        show: bool,
        camera_host: str,
        camera_port: int,
        camera_name: str,
        stale_warning: float,
        stale_log_interval: float,
    ):
        self._capture = None
        self._robot_client = None
        self._camera_name = camera_name
        if camera_host:
            from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

            self._robot_client = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)
            # gear_sonic has hard-coded 10-frame latency and 100-ms stale logs.
            # Replace them with deployment-level diagnostics below.
            self._robot_client.idx = _QuietCameraFrameCounter(self._robot_client.idx)
            self._robot_client._staleness_warning_interval = float("inf")
            print(f"[CAMERA] 使用机器人相机服务 {camera_host}:{camera_port}, stream={camera_name}")
        else:
            self._capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not self._capture.isOpened():
                raise RuntimeError(f"无法打开本地相机 {source}")
            print(f"[CAMERA] 使用 PC 本地相机 {source}")
        self._show = show
        self._stale_warning = stale_warning
        self._stale_log_interval = stale_log_interval
        self._last_robot_timestamp = None
        self._last_new_frame_time = time.monotonic()
        self._last_stale_log_time = 0.0
        self._lock = threading.Lock()
        self._frame = None
        self._phase = "STARTING"
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
                error = self._error
            if error is not None:
                raise RuntimeError(error)
            time.sleep(0.01)
        raise RuntimeError("camera first frame timed out")

    def _run(self):
        try:
            while not self._stop.is_set():
                if self._robot_client is not None:
                    sample = self._robot_client.read(blocking=False)
                    images = None if sample is None else sample.get("images")
                    if not images:
                        time.sleep(0.005)
                        continue
                    if self._camera_name not in images:
                        available = ", ".join(sorted(images))
                        raise RuntimeError(
                            f"robot camera stream {self._camera_name!r} missing; available: {available}"
                        )
                    timestamp = (sample.get("timestamps") or {}).get(self._camera_name)
                    now = time.monotonic()
                    if timestamp is not None and timestamp == self._last_robot_timestamp:
                        stale_for = now - self._last_new_frame_time
                        if (
                            stale_for >= self._stale_warning
                            and now - self._last_stale_log_time >= self._stale_log_interval
                        ):
                            print(
                                f"[CAMERA WARNING] ego_view has not updated for {stale_for:.2f}s; "
                                "reusing the latest frame",
                                flush=True,
                            )
                            self._last_stale_log_time = now
                        time.sleep(0.005)
                        continue
                    self._last_robot_timestamp = timestamp
                    self._last_new_frame_time = now
                    # SONIC camera messages are RGB; keep the internal/public frame in OpenCV BGR.
                    frame = cv2.cvtColor(np.asarray(images[self._camera_name]), cv2.COLOR_RGB2BGR)
                else:
                    ok, frame = self._capture.read()
                    if not ok:
                        raise RuntimeError("local camera frame unavailable")
                with self._lock:
                    self._frame = frame
        except Exception as exc:
            if not self._stop.is_set():
                message = f"camera capture thread failed: {exc}"
                print(f"\n[CAMERA ERROR] {message}", flush=True)
                with self._lock:
                    self._error = message

    def _raise_camera_error(self):
        if self._error is not None:
            raise RuntimeError(self._error)

    def frame(self) -> np.ndarray:
        with self._lock:
            self._raise_camera_error()
            if self._frame is None:
                raise RuntimeError("camera frame unavailable")
            return self._frame.copy()

    def read_key(self) -> str | None:
        """Render HighGUI on the main thread and return a camera-window key."""
        with self._lock:
            self._raise_camera_error()
            frame = None if self._frame is None else self._frame.copy()
            phase = self._phase
        if not self._show or phase not in {"DRY_RUN", "READY", "INFERENCE"} or frame is None:
            return None
        try:
            # Display the exact unobstructed robot image. Status and key hints
            # stay in the terminal so the preview matches the model input.
            cv2.imshow("G1 Deployment Camera", frame)
            code = cv2.waitKey(1) & 0xFF
        except Exception as exc:
            message = f"camera display failed: {exc}"
            with self._lock:
                self._error = message
            raise RuntimeError(message) from exc
        return None if code == 0xFF else chr(code).lower()

    @property
    def preview_enabled(self) -> bool:
        return self._show

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()
        if self._robot_client is not None:
            self._robot_client.close()
        if self._show:
            cv2.destroyAllWindows()


class G1DDS:
    def __init__(
        self,
        network_interface: str,
        lower_body_mode: str = "damping",
        gravity_compensation: bool = True,
        gravity_urdf: Path = DEFAULT_GRAVITY_URDF,
        gravity_scale: float = 1.0,
        gravity_hand_model: str = "rh56dftp",
    ):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        available_interfaces = {name for _, name in socket.if_nameindex()}
        if network_interface not in available_interfaces:
            available = ", ".join(sorted(available_interfaces))
            raise ValueError(
                f"network interface '{network_interface}' does not exist; available: {available}"
            )
        ChannelFactoryInitialize(0, network_interface)
        if lower_body_mode not in ("damping", "zero-torque"):
            raise ValueError(f"unsupported lower-body mode: {lower_body_mode}")
        self._lower_body_mode = lower_body_mode
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._q = None
        self._mode_machine = 0
        self._stamp = 0.0
        self._last_lowstate_warning_time = 0.0
        self._hand_q = None
        self._hand_stamp = 0.0
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._low_level_enabled = False
        self._emergency_damping = False
        self._left_arm_target = None
        self._right_arm_target = None
        self._gravity = (
            ArmGravityCompensator(gravity_urdf, gravity_scale, gravity_hand_model)
            if gravity_compensation
            else None
        )
        self._publisher_stop = threading.Event()
        self._publisher_thread = None
        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()
        # Match SONICMJ BodyCommandSender: direct 29-DoF low-level command topic.
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._hand_cmd = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        self._hand_publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._hand_publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_state, 10)
        self._hand_subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self._hand_subscriber.Init(self._on_hand_state, 10)
        self._wait_for_first_state(timeout=5.0)

    def _on_state(self, msg):
        with self._lock:
            self._q = np.array([motor.q for motor in msg.motor_state[:29]], dtype=np.float64)
            self._mode_machine = int(msg.mode_machine)
            self._stamp = time.monotonic()

    def _on_hand_state(self, msg):
        if len(msg.states) < 12:
            return
        with self._lock:
            self._hand_q = np.array([state.q for state in msg.states[:12]], dtype=np.float64)
            self._hand_stamp = time.monotonic()

    def _wait_for_first_state(self, timeout: float):
        """Allow CycloneDDS discovery to finish before enabling the runtime watchdog."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._q is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError(
            f"no rt/lowstate received within {timeout:.1f}s; "
            "check G1 power/mode, DDS domain, and robot network interface"
        )

    def state(
        self,
        max_age: float,
        warning_age: float | None = None,
        warning_interval: float = 2.0,
    ) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            if self._q is None:
                raise RuntimeError("lowstate missing or stale: no message received")
            age = now - self._stamp
            state = self._q.copy()
        if age > max_age:
            raise RuntimeError(
                f"lowstate missing or stale: age={age * 1000.0:.1f}ms, "
                f"timeout={max_age * 1000.0:.1f}ms"
            )
        if (
            warning_age is not None
            and age > warning_age
            and now - self._last_lowstate_warning_time >= warning_interval
        ):
            print(
                f"[LOWSTATE WARNING] 最新状态已过去 {age * 1000.0:.1f}ms；"
                f"达到 {max_age * 1000.0:.1f}ms 才触发急停",
                flush=True,
            )
            self._last_lowstate_warning_time = now
        return state

    def hand_state(self, max_age: float) -> tuple[np.ndarray, np.ndarray]:
        """Return measured right/left Inspire state from the DDS-Modbus bridge."""
        with self._lock:
            if self._hand_q is None or time.monotonic() - self._hand_stamp > max_age:
                raise RuntimeError(
                    "rt/inspire/state missing or stale; start inspire_modbus_hand.py with --mode dds"
                )
            state = self._hand_q.copy()
        return state[:6], state[6:12]

    def enter_low_level(self, measured: np.ndarray):
        """Release Unitree's motion service, matching SONICMJ real-robot debug mode."""
        if self._low_level_enabled:
            return
        status, result = self._motion_switcher.CheckMode()
        if status != 0:
            raise RuntimeError(f"MotionSwitcher CheckMode failed: status={status}, result={result}")
        for _ in range(10):
            if not result.get("name"):
                break
            status, result = self._motion_switcher.ReleaseMode()
            if status != 0:
                raise RuntimeError(f"MotionSwitcher ReleaseMode failed: status={status}, result={result}")
            time.sleep(1.0)
            status, result = self._motion_switcher.CheckMode()
        if result.get("name"):
            raise RuntimeError(f"failed to release active motion mode: {result}")
        self._left_arm_target = measured[LEFT_ARM_MOTORS].copy()
        self._right_arm_target = measured[RIGHT_ARM_MOTORS].copy()
        self._low_level_enabled = True
        self._publisher_stop.clear()
        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()
        print(f"[LOW LEVEL] rt/lowcmd enabled; lower body={self._lower_body_mode}")

    def send_right_arm(self, target: np.ndarray, measured: np.ndarray):
        if not self._low_level_enabled or self._left_arm_target is None:
            raise RuntimeError("low-level control is not enabled")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("right-arm target must contain 7 finite values")
        with self._command_lock:
            if self._emergency_damping:
                raise RuntimeError("arm command rejected after emergency damping was latched")
            self._right_arm_target = target.copy()

    def send_arms(self, left_target: np.ndarray, right_target: np.ndarray):
        if not self._low_level_enabled:
            raise RuntimeError("low-level control is not enabled")
        left = np.asarray(left_target, dtype=np.float64)
        right = np.asarray(right_target, dtype=np.float64)
        if left.shape != (7,) or right.shape != (7,) or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("left/right arm targets must each contain 7 finite values")
        with self._command_lock:
            if self._emergency_damping:
                raise RuntimeError("arm command rejected after emergency damping was latched")
            self._left_arm_target = left.copy()
            self._right_arm_target = right.copy()

    def enter_emergency_damping(self):
        """Latch zero-position-gain arm damping without requiring fresh LowState."""
        if not self._low_level_enabled:
            return
        with self._command_lock:
            self._emergency_damping = True

    def _publish_loop(self):
        """Keep rt/lowcmd alive at 100 Hz while inference runs asynchronously on the main thread."""
        period = 0.01
        while not self._publisher_stop.is_set():
            start = time.monotonic()
            try:
                self._write_lowcmd()
            except Exception as exc:
                print(f"\n[LOWCMD ERROR] {exc}", flush=True)
                self._publisher_stop.set()
                return
            self._publisher_stop.wait(max(0.0, period - (time.monotonic() - start)))

    def _write_lowcmd(self):
        with self._command_lock:
            if self._right_arm_target is None:
                return
            right_target = self._right_arm_target.copy()
            left_target = self._left_arm_target.copy()
            emergency_damping = self._emergency_damping
        arm_tauff = (
            np.zeros(14, dtype=np.float64)
            if emergency_damping or self._gravity is None
            else self._gravity.compute(left_target, right_target)
        )
        self._cmd.level_flag = 0xFF
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = self._mode_machine
        for i in range(29):
            motor = self._cmd.motor_cmd[i]
            motor.mode = 0x01
            motor.dq = 0.0
            motor.tau = 0.0
            if i in LOWER_BODY_MOTORS:
                motor.q = POS_STOP_F
                motor.kp = 0.0
                if self._lower_body_mode == "damping":
                    motor.kd = float(LOWER_BODY_KD[i])
                else:
                    motor.dq = VEL_STOP_F
                    motor.kd = 0.0
            elif i in LEFT_ARM_MOTORS:
                arm_index = i - LEFT_ARM_MOTORS[0]
                motor.tau = float(arm_tauff[arm_index])
                motor.q = POS_STOP_F if emergency_damping else float(left_target[arm_index])
                motor.kp = 0.0 if emergency_damping else float(ARM_KP[arm_index])
                motor.kd = float(ARM_KD[arm_index])
            else:
                arm_index = i - RIGHT_ARM_MOTORS[0]
                motor.tau = float(arm_tauff[7 + arm_index])
                motor.q = POS_STOP_F if emergency_damping else float(right_target[arm_index])
                motor.kp = 0.0 if emergency_damping else float(ARM_KP[arm_index])
                motor.kd = float(ARM_KD[arm_index])
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._publisher.Write(self._cmd)

    def stop_low_level_publisher(self):
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)

    def send_inspire_hands(self, right_target: np.ndarray, left_target: np.ndarray):
        """Publish normalized Inspire commands; bridge order is right 6 then left 6."""
        right = np.clip(np.asarray(right_target, dtype=np.float64), 0.0, 1.0)
        left = np.clip(np.asarray(left_target, dtype=np.float64), 0.0, 1.0)
        if right.shape != (6,) or left.shape != (6,):
            raise ValueError("Inspire hand targets must each contain 6 values")
        for i, value in enumerate(np.concatenate((right, left))):
            self._hand_cmd.cmds[i].q = float(value)
        self._hand_publisher.Write(self._hand_cmd)


class EStop:
    def __init__(self):
        self.latched = False
        self.reason = ""

    def trigger(self, reason: str):
        if not self.latched:
            self.latched, self.reason = True, reason
            print(
                f"\n[E-STOP] 已锁存：{reason}；状态有效时保持实测角，"
                "状态失联时双臂切换零位置增益阻尼",
                flush=True,
            )


def _read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1).lower()
    return None


def _minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    """Zero-velocity/acceleration interpolation at both endpoints."""
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument(
        "--lower-body-mode",
        choices=("damping", "zero-torque"),
        default="damping",
        help="legs/waist receive no position target; damping is the safer default",
    )
    parser.add_argument(
        "--no-gravity-compensation", action="store_true",
        help="disable the default full-time dual-arm Pinocchio/RNEA gravity feed-forward",
    )
    parser.add_argument("--gravity-urdf", type=Path, default=DEFAULT_GRAVITY_URDF)
    parser.add_argument(
        "--gravity-scale", type=float, default=1.0,
        help="gravity feed-forward scale in [0,1]",
    )
    parser.add_argument(
        "--gravity-hand-model", choices=("rh56dftp", "rubber"), default="rh56dftp",
        help="end-effector inertial model; defaults to the installed RH56DFTP hand",
    )
    parser.add_argument("--camera", default="0", help="local OpenCV camera index/URL; used only when --camera-host is empty")
    parser.add_argument("--camera-host", default="192.168.123.164", help="SONIC robot camera server; pass an empty string for local camera")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-name", default="ego_view", help="robot camera stream used by the training dataset")
    parser.add_argument("--camera-stale-warning", type=float, default=0.5, help="warn after this many seconds without a genuinely new robot frame")
    parser.add_argument("--camera-stale-log-interval", type=float, default=2.0, help="minimum interval between stale-camera warnings")
    parser.add_argument(
        "--view-camera",
        action="store_true",
        help="show the selected robot/local camera after initialization reaches READY",
    )
    parser.add_argument(
        "--view-simulation",
        action="store_true",
        help="open a synchronized MuJoCo shadow that assumes every nominal arm target is reached exactly",
    )
    parser.add_argument(
        "--simulation-model",
        type=Path,
        default=PROJECT_ROOT
        / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml",
    )
    parser.add_argument("--instruction", default="pick up the pipette")
    parser.add_argument(
        "--action-space",
        choices=("joint", "wrist-delta"),
        default="joint",
        help="checkpoint action representation; wrist-delta runs online MuJoCo FK/IK",
    )
    parser.add_argument(
        "--ik-model",
        type=Path,
        default=PROJECT_ROOT
        / "decoupled_wbc/gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml",
    )
    parser.add_argument("--ik-max-iterations", type=int, default=80)
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-max-step", type=float, default=0.2, help="maximum IK iteration step in rad")
    parser.add_argument("--ik-position-tolerance", type=float, default=2e-5)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=2e-4)
    parser.add_argument("--max-wrist-delta-position", type=float, default=0.03, help="per predicted step, metres")
    parser.add_argument("--max-wrist-delta-orientation", type=float, default=0.35, help="per predicted step, radians")
    parser.add_argument("--max-ik-position-residual", type=float, default=0.01, help="metres")
    parser.add_argument("--max-ik-orientation-residual", type=float, default=0.1, help="radians")
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--max-speed", type=float, default=0.25, help="rad/s")
    parser.add_argument("--max-hand-speed", type=float, default=0.5, help="normalized Inspire units/s")
    parser.add_argument(
        "--lowstate-warning-age",
        type=float,
        default=0.2,
        help="warn when no new LowState has arrived for this many seconds",
    )
    parser.add_argument(
        "--lowstate-timeout",
        type=float,
        default=0.5,
        help="latch e-stop after this continuous LowState outage in seconds",
    )
    parser.add_argument(
        "--lowstate-warning-interval",
        type=float,
        default=2.0,
        help="minimum interval between delayed-LowState warnings",
    )
    parser.add_argument(
        "--estop-damping-duration",
        type=float,
        default=10.0,
        help="keep publishing arm damping this long after stale LowState, allowing it to reach a reconnecting robot",
    )
    parser.add_argument("--initial-speed", type=float, default=0.15, help="initialization speed limit in rad/s")
    parser.add_argument("--initial-duration", type=float, default=5.0, help="minimum initialization duration in seconds")
    parser.add_argument("--initial-tolerance", type=float, default=0.1, help="per-joint READY tolerance in rad")
    parser.add_argument("--initial-hand-tolerance", type=float, default=0.02, help="per-finger READY tolerance")
    parser.add_argument(
        "--enable-initial-outer-loop-compensation",
        action="store_true",
        help="enable initialization/READY integral position-offset compensation; disabled by default",
    )
    parser.add_argument(
        "--initial-correction-rate",
        type=float,
        default=2.0,
        help="post-interpolation outer-loop correction rate in 1/s",
    )
    parser.add_argument(
        "--initial-correction-speed",
        type=float,
        default=0.03,
        help="maximum correction-offset change speed in rad/s",
    )
    parser.add_argument(
        "--initial-correction-limit",
        type=float,
        default=0.15,
        help="maximum outer-loop position offset per joint in rad",
    )
    parser.add_argument(
        "--initial-correction-deadband",
        type=float,
        default=0.003,
        help="do not integrate joint errors smaller than this value in rad",
    )
    parser.add_argument(
        "--initial-stable-duration",
        type=float,
        default=1.0,
        help="all joints must remain within tolerance for this many seconds before READY",
    )
    parser.add_argument(
        "--initial-left-arm",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_LEFT_ARM.tolist(),
        metavar=("SP", "SR", "SY", "E", "WR", "WP", "WY"),
        help="left-arm deployment pose; default is episode 0 observation.state at 3.0 s",
    )
    parser.add_argument(
        "--initial-right-arm",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_RIGHT_ARM.tolist(),
        metavar=("SP", "SR", "SY", "E", "WR", "WP", "WY"),
        help="right-arm deployment pose; default is episode 0 observation.state at 3.0 s",
    )
    parser.add_argument(
        "--initial-right-hand-state",
        "--right-hand-state",
        dest="initial_right_hand_state",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_RIGHT_HAND_STATE.tolist(),
        help="episode-0 measured right Inspire state used for READY and model input",
    )
    parser.add_argument(
        "--initial-left-hand-state",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_LEFT_HAND_STATE.tolist(),
        help="episode-0 measured left Inspire state used for READY",
    )
    parser.add_argument(
        "--initial-hand-command",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_HAND_COMMAND.tolist(),
        help="episode-0 action.wbc command sent to both Inspire hands",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="ZMQ send/receive timeout; includes first-inference CUDA warm-up",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=PROJECT_ROOT / "inference_records/deployment_model_io",
        help="one NPZ per model request; relative paths are resolved from the current directory",
    )
    parser.add_argument(
        "--no-record-model-io",
        action="store_true",
        help="disable recording model inputs and outputs",
    )
    parser.add_argument("--arm", action="store_true", help="actually publish rt/lowcmd")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("必须在交互终端运行，确保键盘急停可用")
    if args.camera_stale_warning <= 0.0 or args.camera_stale_log_interval <= 0.0:
        raise SystemExit("camera stale warning/log intervals must be positive")
    if args.estop_damping_duration <= 0.0:
        raise SystemExit("estop-damping-duration must be positive")
    if not 0.0 <= args.gravity_scale <= 1.0:
        raise SystemExit("gravity-scale must be in [0, 1]")
    if min(args.lowstate_warning_age, args.lowstate_timeout, args.lowstate_warning_interval) <= 0.0:
        raise SystemExit("LowState watchdog parameters must be positive")
    if args.lowstate_warning_age >= args.lowstate_timeout:
        raise SystemExit("lowstate-warning-age must be smaller than lowstate-timeout")
    if min(
        args.ik_max_iterations,
        args.ik_damping,
        args.ik_max_step,
        args.ik_position_tolerance,
        args.ik_orientation_tolerance,
        args.max_wrist_delta_position,
        args.max_wrist_delta_orientation,
        args.max_ik_position_residual,
        args.max_ik_orientation_residual,
    ) <= 0:
        raise SystemExit("IK and wrist-delta safety parameters must be positive")

    robot = G1DDS(
        args.network_interface,
        args.lower_body_mode,
        gravity_compensation=not args.no_gravity_compensation,
        gravity_urdf=args.gravity_urdf,
        gravity_scale=args.gravity_scale,
        gravity_hand_model=args.gravity_hand_model,
    )

    def read_robot_state() -> np.ndarray:
        return robot.state(
            args.lowstate_timeout,
            warning_age=args.lowstate_warning_age,
            warning_interval=args.lowstate_warning_interval,
        )

    policy = PolicyClient(args.server, args.port, args.timeout_ms)
    recorder = ModelIORecorder(
        None if args.no_record_model_io else args.record_dir,
        args.action_space,
    )
    camera = CameraStream(
        args.camera,
        args.view_camera,
        args.camera_host,
        args.camera_port,
        args.camera_name,
        args.camera_stale_warning,
        args.camera_stale_log_interval,
    )
    shadow = None
    if args.view_simulation:
        from examples.Real_G1.joint_angle_deploy.live_shadow_mujoco import LiveShadowMujoco

        shadow = LiveShadowMujoco(args.simulation_model)
    wrist_ik = None
    if args.action_space == "wrist-delta":
        from examples.PipetteRightOnly.convert_wrist_delta_to_joint_chunks import RightArmIK

        ik_args = SimpleNamespace(
            max_iterations=args.ik_max_iterations,
            damping=args.ik_damping,
            max_step=args.ik_max_step,
            position_tolerance=args.ik_position_tolerance,
            orientation_tolerance=args.ik_orientation_tolerance,
        )
        wrist_ik = RightArmIK(args.ik_model.expanduser().resolve(), ik_args)
        print(f"[IK] wrist-delta 在线 FK/IK 已启用，模型: {args.ik_model}")
    estop, period = EStop(), 1.0 / args.frequency
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    initial_left_pose = np.asarray(args.initial_left_arm, dtype=np.float64)
    initial_pose = np.asarray(args.initial_right_arm, dtype=np.float64)
    initial_right_hand_state = np.asarray(args.initial_right_hand_state, dtype=np.float64)
    initial_left_hand_state = np.asarray(args.initial_left_hand_state, dtype=np.float64)
    initial_hand_command = np.asarray(args.initial_hand_command, dtype=np.float64)
    if np.any(initial_left_pose < LEFT_ARM_LOWER) or np.any(initial_left_pose > LEFT_ARM_UPPER):
        raise SystemExit("initial-left-arm exceeds URDF joint limits")
    if np.any(initial_pose < RIGHT_ARM_LOWER) or np.any(initial_pose > RIGHT_ARM_UPPER):
        raise SystemExit("initial-right-arm exceeds URDF joint limits")
    if min(
        args.initial_correction_rate,
        args.initial_correction_speed,
        args.initial_correction_limit,
        args.initial_correction_deadband,
        args.initial_stable_duration,
    ) < 0.0:
        raise SystemExit("initial correction and stable-duration parameters must be non-negative")
    if args.initial_tolerance <= 0.0 or args.initial_hand_tolerance <= 0.0:
        raise SystemExit("initial arm/hand tolerances must be positive")
    if args.initial_correction_deadband >= args.initial_tolerance:
        raise SystemExit("initial-correction-deadband must be smaller than initial-tolerance")
    for name, values in (
        ("initial-right-hand-state", initial_right_hand_state),
        ("initial-left-hand-state", initial_left_hand_state),
        ("initial-hand-command", initial_hand_command),
    ):
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise SystemExit(f"{name} must be normalized to [0, 1]")
    print(f"episode 0 t=3.0s/frame 150 左臂目标: {np.round(initial_left_pose, 4)}")
    print(f"episode 0 t=3.0s/frame 150 右臂目标: {np.round(initial_pose, 4)}")
    print(f"episode 0 t=3.0s/frame 150 左手实测: {np.round(initial_left_hand_state, 4)}")
    print(f"episode 0 t=3.0s/frame 150 右手实测: {np.round(initial_right_hand_state, 4)}")
    print(f"episode 0 t=3.0s/frame 150 双手命令: {np.round(initial_hand_command, 4)}")
    print(
        "初始化外环位置补偿: "
        f"{'启用' if args.enable_initial_outer_loop_compensation else '关闭（默认）'}"
    )
    print("SPACE/Q=急停；ENTER=开始抬臂初始化；到达 READY 后按 L 才启动模型。")
    enabled = False
    phase = "DRY_RUN" if not args.arm else "DISARMED"
    camera.set_phase(phase)
    init_start_q = None
    init_start_left_q = None
    init_start_right_hand = None
    init_start_left_hand = None
    init_start_time = 0.0
    init_duration = args.initial_duration
    last_init_status_time = 0.0
    within_tolerance_since = None
    left_init_correction = np.zeros(7, dtype=np.float64)
    right_init_correction = np.zeros(7, dtype=np.float64)
    cache = np.empty((0, 13))
    cache_inference_index = -1
    cache_action_index = 0
    previous_wrist_pose = None
    right_hand_command = initial_hand_command.copy()
    left_hand_command = initial_hand_command.copy()
    try:
        while not estop.latched:
            start = time.monotonic()
            key = _read_key() or camera.read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            if key in ("\n", "\r") and args.arm and phase == "DISARMED":
                measured = read_robot_state()
                init_start_right_hand, init_start_left_hand = robot.hand_state(0.5)
                robot.enter_low_level(measured)
                enabled = True
                phase = "START_INITIALIZATION"
                camera.set_phase(phase)
                print("[ARMED] 已启用 rt/lowcmd，开始移动到初始化姿态；机器人必须可靠吊挂")
            if key == "l" and phase == "READY":
                phase = "INFERENCE"
                camera.set_phase(phase)
                cache = np.empty((0, 13))
                previous_wrist_pose = None
                print(f"[INFERENCE] 模型已接管右臂，action-space={args.action_space}")
            measured = read_robot_state()
            left_arm_q = measured[LEFT_ARM_MOTORS]
            arm_q = measured[RIGHT_ARM_MOTORS]
            if shadow is not None and phase in ("DISARMED", "DRY_RUN"):
                shadow.update(measured, left_arm_q, arm_q)
            if phase not in ("DISARMED", "DRY_RUN"):
                right_hand_measured, left_hand_measured = robot.hand_state(0.5)
            if phase == "START_INITIALIZATION":
                init_start_left_q = measured[LEFT_ARM_MOTORS].copy()
                init_start_q = arm_q.copy()
                init_start_time = time.monotonic()
                # Minimum-jerk peak velocity is 1.875 * distance / duration.
                max_distance = max(
                    float(np.max(np.abs(initial_pose - init_start_q))),
                    float(np.max(np.abs(initial_left_pose - init_start_left_q))),
                )
                speed_duration = 1.875 * max_distance / args.initial_speed
                init_duration = max(args.initial_duration, speed_duration)
                phase = "INITIALIZING"
                camera.set_phase(phase)
                print(f"[INITIALIZING] 预计 {init_duration:.1f}s；插值期间 SPACE/Q 同样有效")
            if phase == "INITIALIZING":
                progress = (time.monotonic() - init_start_time) / init_duration
                target = _minimum_jerk(init_start_q, initial_pose, progress)
                left_target = _minimum_jerk(init_start_left_q, initial_left_pose, progress)
                right_hand_command = _minimum_jerk(init_start_right_hand, initial_hand_command, progress)
                left_hand_command = _minimum_jerk(init_start_left_hand, initial_hand_command, progress)
                if progress >= 1.0:
                    left_error = initial_left_pose - left_arm_q
                    right_error = initial_pose - arm_q
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
                    left_target = np.clip(initial_left_pose + left_init_correction, LEFT_ARM_LOWER, LEFT_ARM_UPPER)
                    target = np.clip(initial_pose + right_init_correction, RIGHT_ARM_LOWER, RIGHT_ARM_UPPER)
                robot.send_arms(left_target, target)
                robot.send_inspire_hands(right_hand_command, left_hand_command)
                if shadow is not None:
                    shadow.update(measured, left_target, target)
                if progress >= 1.0:
                    right_hand_error = initial_right_hand_state - right_hand_measured
                    left_hand_error = initial_left_hand_state - left_hand_measured
                    max_error = max(float(np.max(np.abs(left_error))), float(np.max(np.abs(right_error))))
                    max_hand_error = max(
                        float(np.max(np.abs(left_hand_error))),
                        float(np.max(np.abs(right_hand_error))),
                    )
                    now = time.monotonic()
                    if max_error <= args.initial_tolerance and max_hand_error <= args.initial_hand_tolerance:
                        if within_tolerance_since is None:
                            within_tolerance_since = now
                    else:
                        within_tolerance_since = None
                    stable_duration = 0.0 if within_tolerance_since is None else now - within_tolerance_since
                    if within_tolerance_since is not None and stable_duration >= args.initial_stable_duration:
                        phase = "READY"
                        camera.set_phase(phase)
                        print("[READY] 双臂和双手初始化姿态已到位。检查现场安全后按 L 启动模型")
                        if camera.preview_enabled:
                            print("[CAMERA] 正在 PC 主线程打开机器人 ego_view 预览窗口")
                        else:
                            print("[CAMERA] 未启用预览；重新启动时请添加 --view-camera")
                        print(
                            "  left wrist  target/measured/error: "
                            f"{np.round(initial_left_pose[4:7], 4)} / "
                            f"{np.round(left_arm_q[4:7], 4)} / {np.round(left_error[4:7], 4)}"
                        )
                        print(
                            "  right wrist target/measured/error: "
                            f"{np.round(initial_pose[4:7], 4)} / "
                            f"{np.round(arm_q[4:7], 4)} / {np.round(right_error[4:7], 4)}"
                        )
                        print(
                            "  right hand  target/measured/error: "
                            f"{np.round(initial_right_hand_state, 4)} / "
                            f"{np.round(right_hand_measured, 4)} / {np.round(right_hand_error, 4)}"
                        )
                    elif now - last_init_status_time >= 1.0:
                        print(
                            f"[INITIALIZING] arms={max_error:.4f}rad hands={max_hand_error:.4f} | "
                            f"L-wrist err={np.round(left_error[4:7], 4)} | "
                            f"R-wrist err={np.round(right_error[4:7], 4)} | "
                            f"correction_max={max(np.max(np.abs(left_init_correction)), np.max(np.abs(right_init_correction))):.4f} rad | "
                            f"L-wrist corr={np.round(left_init_correction[4:7], 4)} | "
                            f"R-wrist corr={np.round(right_init_correction[4:7], 4)} | "
                            f"stable={stable_duration:.1f}/{args.initial_stable_duration:.1f}s"
                        )
                        last_init_status_time = now
                time.sleep(max(0.0, period - (time.monotonic() - start)))
                continue
            if phase in ("DISARMED", "READY"):
                if phase == "READY":
                    left_error = initial_left_pose - left_arm_q
                    right_error = initial_pose - arm_q
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
                    left_hold = np.clip(initial_left_pose + left_init_correction, LEFT_ARM_LOWER, LEFT_ARM_UPPER)
                    right_hold = np.clip(initial_pose + right_init_correction, RIGHT_ARM_LOWER, RIGHT_ARM_UPPER)
                    robot.send_arms(left_hold, right_hold)
                    robot.send_inspire_hands(right_hand_command, left_hand_command)
                    if shadow is not None:
                        shadow.update(measured, initial_left_pose, initial_pose)
                time.sleep(max(0.0, period - (time.monotonic() - start)))
                continue
            frame = camera.frame()
            current_wrist_pose = None
            pelvis_pose = None
            if wrist_ik is not None:
                wrist_ik.set_source_state(measured[:29])
                current_wrist_pose = wrist_ik.wrist_pose_in_pelvis()
                pelvis_pose = wrist_ik.pelvis_pose()
            if len(cache) == 0:
                image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
                if wrist_ik is None:
                    model_state = np.concatenate((arm_q, right_hand_measured))
                else:
                    current_position, current_rotation = current_wrist_pose
                    if previous_wrist_pose is None:
                        previous_position, previous_rotation = current_position, current_rotation
                    else:
                        previous_position, previous_rotation = previous_wrist_pose
                    model_state = _relative_wrist_state(
                        previous_position,
                        previous_rotation,
                        current_position,
                        current_rotation,
                        right_hand_measured,
                    )
                predicted = policy.predict(image, model_state, args.instruction)
                if wrist_ik is None:
                    if predicted.ndim != 2 or predicted.shape[1] != 13 or not np.isfinite(predicted).all():
                        raise RuntimeError(f"invalid joint policy output {predicted.shape}")
                    cache = predicted[: args.execution_horizon]
                    recorder.record(image, model_state, args.instruction, predicted, cache)
                else:
                    pelvis_position, pelvis_rotation = pelvis_pose
                    try:
                        cache = _wrist_actions_to_joint_actions(
                            predicted,
                            wrist_ik,
                            current_position,
                            current_rotation,
                            pelvis_position,
                            pelvis_rotation,
                            args.execution_horizon,
                            args.max_wrist_delta_position,
                            args.max_wrist_delta_orientation,
                            args.max_ik_position_residual,
                            args.max_ik_orientation_residual,
                        )
                    except Exception as exc:
                        recorder.record(
                            image,
                            model_state,
                            args.instruction,
                            predicted,
                            None,
                            str(exc),
                        )
                        raise
                    recorder.record(image, model_state, args.instruction, predicted, cache)
                cache_inference_index = recorder.step - 1
                cache_action_index = 0
            if current_wrist_pose is not None:
                previous_wrist_pose = (
                    current_wrist_pose[0].copy(),
                    current_wrist_pose[1].copy(),
                )
            desired_arm, desired_hand, cache = cache[0, :7], cache[0, 7:13], cache[1:]
            if np.any(desired_arm < RIGHT_ARM_LOWER) or np.any(desired_arm > RIGHT_ARM_UPPER):
                raise RuntimeError("policy target exceeds URDF joint limits")
            if np.any(desired_hand < 0.0) or np.any(desired_hand > 1.0):
                raise RuntimeError("policy Inspire target exceeds normalized limits")
            max_step = args.max_speed * period
            target = arm_q + np.clip(desired_arm - arm_q, -max_step, max_step)
            hand_max_step = args.max_hand_speed * period
            right_hand_command += np.clip(desired_hand - right_hand_command, -hand_max_step, hand_max_step)
            recorder.record_execution(
                cache_inference_index,
                cache_action_index,
                desired_arm,
                target,
                arm_q,
                desired_hand,
                right_hand_command,
                right_hand_measured,
            )
            cache_action_index += 1
            if shadow is not None:
                # Use the un-clipped online IK target: this is the ideal pose when
                # every relative wrist step is reached with zero tracking error.
                shadow.update(measured, initial_left_pose, desired_arm)
            if enabled:
                robot.send_right_arm(target, measured)
                robot.send_inspire_hands(right_hand_command, left_hand_command)
            else:
                print(f"\rdry-run q_target={np.round(target, 3)}", end="", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        recorder.close()
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
            camera.close()
            if shadow is not None:
                shadow.close()


if __name__ == "__main__":
    main()
