import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    HandState_,
    IMUState_,
    LowState_ as LowState_hg,
    OdoState_,
)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Vector3_

from gr00t_wbc.control.utils.signal_handler import is_shutdown_requested

MOTION_SWITCHER_TIMEOUT = 15.0  # timeout for motion switcher request

class BodyStateProcessor:
    def _release_motion_mode(self):
        """Release motion mode using MotionSwitcherClient.

        This is REQUIRED before sending low-level commands to the robot.
        The robot won't accept rt/lowcmd unless the motion mode is released.

        Note: API request frequency should not exceed 50Hz to avoid error 3104.
        If CheckMode fails, we continue anyway as the robot may already be in debug mode.
        """
        print("[MotionSwitcher] Initializing motion switcher client...")

        try:
            msc = MotionSwitcherClient()
            msc.SetTimeout(MOTION_SWITCHER_TIMEOUT)
            msc.Init()

            # Wait before first request (avoid rapid requests, stay under 50Hz)
            time.sleep(0.2)

            status, result = msc.CheckMode()
            if status != 0:
                print(f"[MotionSwitcher] CheckMode failed with status {status}, continuing anyway - robot may already be in debug mode")
                return

            print(f"[MotionSwitcher] Current mode: {result}")
            while result and result.get("name"):
                print(f"[MotionSwitcher] Releasing mode: {result['name']}")

                # Wait between API calls (50Hz max = 20ms minimum between calls)
                time.sleep(0.1)
                msc.ReleaseMode()

                # Wait after release before checking again
                time.sleep(0.5)

                time.sleep(0.1)  # Additional delay before next CheckMode
                status, result = msc.CheckMode()
                if status != 0:
                    print(f"[MotionSwitcher] CheckMode after release failed with status {status}")
                    break

            print("[MotionSwitcher] Robot is now in debug mode")

        except Exception as e:
            print(f"[MotionSwitcher] Error: {e}, continuing anyway - robot may already be in debug mode")

    def __init__(self, config):
        self.config = config

        # Enter debug mode for real robot
        if self.config["ENV_TYPE"] == "real":
            self._release_motion_mode()

        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(None, 10)  # Queue depth 10 for non-blocking reads
        elif (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
        ):
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(None, 10)  # Queue depth 10 for non-blocking reads

            self.secondary_imu_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
            self.secondary_imu_subscriber.Init(None, 10)  # Queue depth 10 for non-blocking reads

            # Subscribe to odo state (only available in simulation)
            if self.config["ENV_TYPE"] == "sim":
                self.odo_state_subscriber = ChannelSubscriber("rt/odostate", OdoState_)
                self.odo_state_subscriber.Init(None, 10)  # Queue depth 10 for non-blocking reads
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported")

        self.num_dof = self.config["NUM_JOINTS"]
        # 3 + 4 + 19
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.ddq = np.zeros(3 + 3 + self.num_dof)
        self.tau_est = np.zeros(3 + 3 + self.num_dof)
        self.torso_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w, x, y, z)
        self.torso_ang_vel = np.zeros(3)
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)
        self.robot_low_state = None
        self.secondary_imu_state = None
        self.odo_state = None
        
        # Wait for robot to start sending low state (real robot only)
        if self.config["ENV_TYPE"] == "real":
            print("Waiting for robot low state... (Ctrl+C to cancel)")
            timeout = 10.0  # seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check for shutdown request (Ctrl+C)
                if is_shutdown_requested():
                    print("[BodyStateProcessor] Shutdown requested, aborting wait for robot state")
                    raise KeyboardInterrupt("Shutdown requested during robot state wait")

                self.robot_low_state = self.robot_lowstate_subscriber.Read()
                if self.robot_low_state:
                    print(f"Robot low state received! Mode machine: {self.robot_low_state.mode_machine}")
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Timeout waiting for robot low state. Check robot connection and ensure robot is in debug mode.")

    def _prepare_low_state(self) -> np.ndarray:
        # Use short timeout to prevent blocking and allow responsive Ctrl+C handling.
        # Without timeout, Read() blocks for ~33ms per call waiting for new data,
        # which prevents the signal handler from processing Ctrl+C interrupts.
        # The timeout makes Read() non-blocking while still capturing available data.
        self.robot_low_state = self.robot_lowstate_subscriber.Read(timeout=0.01)
        self.secondary_imu_state = self.secondary_imu_subscriber.Read(timeout=0.01)

        if not self.robot_low_state:
            print("No low state received")
            return
        imu_state = self.robot_low_state.imu_state

        # Use odo_state for position and velocity if available, otherwise set to zero
        if self.config["ENV_TYPE"] == "sim":
            self.odo_state = self.odo_state_subscriber.Read(timeout=0.01)
            if self.odo_state:
                self.q[0:3] = self.odo_state.position
                self.dq[0:3] = self.odo_state.linear_velocity
            else:
                self.q[0:3] = [0.0, 0.0, 0.0]
                self.dq[0:3] = [0.0, 0.0, 0.0]
        else:
            self.q[0:3] = [0.0, 0.0, 0.0]
            self.dq[0:3] = [0.0, 0.0, 0.0]

        self.q[3:7] = imu_state.quaternion  # w, x, y, z
        # Note: IMU bias correction moved to g1_gear_wbc_policy.py
        # Applied conditionally only for forward/backward movement

        self.dq[3:6] = imu_state.gyroscope
        self.ddq[0:3] = imu_state.accelerometer
        unitree_joint_state = self.robot_low_state.motor_state

        # Handle secondary IMU (might be None if timeout or not available)
        if self.secondary_imu_state:
            self.torso_quat = self.secondary_imu_state.quaternion
            self.torso_ang_vel = self.secondary_imu_state.gyroscope
        # else: keep previous values (initialized to identity quaternion and zeros in __init__)

        for i in range(self.num_dof):
            self.q[7 + i] = unitree_joint_state[self.config["JOINT2MOTOR"][i]].q
            self.dq[6 + i] = unitree_joint_state[self.config["JOINT2MOTOR"][i]].dq
            self.tau_est[6 + i] = unitree_joint_state[self.config["JOINT2MOTOR"][i]].tau_est

        robot_state_data = np.concatenate(
            [self.q, self.dq, self.tau_est, self.ddq, self.torso_quat, self.torso_ang_vel], axis=0
        ).reshape(1, -1)
        # (7 + 29) + (6 + 29) + (6 + 29) + (6 + 29) = 141 dim

        return robot_state_data

    def close(self):
        """Close DDS subscribers."""
        try:
            if hasattr(self, 'robot_lowstate_subscriber') and self.robot_lowstate_subscriber:
                self.robot_lowstate_subscriber.Close()
        except Exception as e:
            print(f"[BodyStateProcessor] Error closing lowstate subscriber: {e}")

        try:
            if hasattr(self, 'secondary_imu_subscriber') and self.secondary_imu_subscriber:
                self.secondary_imu_subscriber.Close()
        except Exception as e:
            print(f"[BodyStateProcessor] Error closing IMU subscriber: {e}")

        try:
            if hasattr(self, 'odo_state_subscriber') and self.odo_state_subscriber:
                self.odo_state_subscriber.Close()
        except Exception as e:
            print(f"[BodyStateProcessor] Error closing odo subscriber: {e}")


class HandStateProcessor:
    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        if self.is_left:
            self.state_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        else:
            self.state_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)

        self.state_sub.Init(None, 10)  # Queue depth 10 for non-blocking reads
        self.state = None
        self.num_dof = 7  # for single hand

    def _prepare_low_state(self) -> np.ndarray:
        # Use short timeout to avoid blocking when topic unavailable
        self.state = self.state_sub.Read(timeout=0.001)

        if not self.state:
            print("No state received")
            return

        state_data = (
            np.concatenate(
                [
                    [self.state.motor_state[i].q for i in range(self.num_dof)],
                    [self.state.motor_state[i].dq for i in range(self.num_dof)],
                    [self.state.motor_state[i].tau_est for i in range(self.num_dof)],
                    [self.state.motor_state[i].ddq for i in range(self.num_dof)],
                ],
                axis=0,
            )
            .astype(np.float64)
            .reshape(1, -1)
        )
        return state_data

    def close(self):
        """Close DDS subscriber."""
        try:
            if hasattr(self, 'state_sub') and self.state_sub:
                self.state_sub.Close()
        except Exception as e:
            print(f"[HandStateProcessor] Error closing subscriber: {e}")


class AlohaSharedStateProcessor:
    """Shared state processor for ALOHA-style grippers (single DDS subscriber for both hands).

    ALOHA feedback messages contain BOTH left and right gripper positions in a single Vector3_:
    - x: left gripper position
    - y: right gripper position
    - z: load/reserved

    To avoid duplicate subscriber issues, use ONE shared instance for both hands.
    Each hand then uses get_state_for_hand() to extract its specific data.
    """

    def __init__(self, state_topic: str = None, use_feedback: bool = False):
        """Initialize shared ALOHA state processor.

        Args:
            state_topic: DDS topic for gripper state feedback (e.g., "rt/aloha_hand/state")
            use_feedback: If True, subscribe to feedback topic; if False, use command echo
        """
        self.use_feedback = use_feedback
        self.state_topic = state_topic
        self.num_dof = 1  # ALOHA has 1-DOF gripper per hand

        # State variables for BOTH grippers in hardware convention
        # Hardware range: 0.0 (open) to 0.065 (closed)
        self.left_gripper_pos = 0.0
        self.left_gripper_vel = 0.0
        self.left_gripper_acc = 0.0
        self.left_gripper_load = 0.0

        self.right_gripper_pos = 0.0
        self.right_gripper_vel = 0.0
        self.right_gripper_acc = 0.0
        self.right_gripper_load = 0.0

        self.state_sub = None
        if self.use_feedback and state_topic:
            # Single subscriber for both hands
            self.state_sub = ChannelSubscriber(state_topic, Vector3_)
            self.state_sub.Init(None, 10)  # Queue depth 10 for non-blocking reads
            print(f"AlohaSharedStateProcessor: Subscribing to feedback topic '{state_topic}'")
        else:
            print(f"AlohaSharedStateProcessor: Using command echo (no feedback subscription)")

    def read_feedback(self):
        """Read feedback from DDS topic and update both gripper states."""
        if self.use_feedback and self.state_sub:
            # Use short timeout to avoid blocking and prevent "[Reader] take sample error"
            # when topic doesn't exist (e.g., running outside Docker without ALOHA bridge)
            state_msg = self.state_sub.Read(timeout=0.001)
            if state_msg:
                # Feedback format: x=left_pos, y=right_pos, z=load
                self.left_gripper_pos = state_msg.x
                self.right_gripper_pos = state_msg.y
                # Load is shared (could be split in future)
                self.left_gripper_load = state_msg.z
                self.right_gripper_load = state_msg.z

    def get_state_for_hand(self, is_left: bool) -> np.ndarray:
        """Get state array for a specific hand.

        Args:
            is_left: True for left hand, False for right hand

        Returns:
            State array with shape (1, 4): [position, velocity, torque_est, acceleration]
        """
        # Read latest feedback if available
        self.read_feedback()

        if is_left:
            state_data = np.array([
                self.left_gripper_pos,
                self.left_gripper_vel,
                self.left_gripper_load,
                self.left_gripper_acc,
            ])
        else:
            state_data = np.array([
                self.right_gripper_pos,
                self.right_gripper_vel,
                self.right_gripper_load,
                self.right_gripper_acc,
            ])

        return state_data.astype(np.float64).reshape(1, -1)

    def update_state_from_command(self, is_left: bool, commanded_pos: float):
        """Update internal state estimate from commanded position.

        Used when feedback is not available.

        Args:
            is_left: True for left hand, False for right hand
            commanded_pos: Commanded gripper position (0.0 to 0.065)
        """
        pos = np.clip(commanded_pos, 0.0, 0.065)
        if is_left:
            self.left_gripper_pos = pos
            self.left_gripper_vel = 0.0
            self.left_gripper_acc = 0.0
            self.left_gripper_load = 0.0
        else:
            self.right_gripper_pos = pos
            self.right_gripper_vel = 0.0
            self.right_gripper_acc = 0.0
            self.right_gripper_load = 0.0

    def close(self):
        """Close DDS subscriber if using feedback."""
        try:
            if self.state_sub:
                self.state_sub.Close()
        except Exception as e:
            print(f"[AlohaSharedStateProcessor] Error closing subscriber: {e}")


class AlohaHandStateProcessor:
    """State processor for a single ALOHA gripper hand.

    This is a lightweight wrapper that references a shared AlohaSharedStateProcessor.
    This design avoids duplicate DDS subscribers to the same topic.

    If no shared processor is provided, creates its own subscriber (legacy mode).
    """

    def __init__(self, is_left: bool = True, state_topic: str = None, use_feedback: bool = False,
                 shared_processor: AlohaSharedStateProcessor = None):
        """Initialize ALOHA hand state processor.

        Args:
            is_left: True for left hand, False for right hand
            state_topic: Optional DDS topic for gripper state feedback
            use_feedback: If True, subscribe to feedback topic; if False, use command echo
            shared_processor: Optional shared processor to avoid duplicate subscribers
        """
        self.is_left = is_left
        self.use_feedback = use_feedback
        self.num_dof = 1  # ALOHA has 1-DOF gripper

        # Use shared processor if provided (recommended)
        self.shared_processor = shared_processor

        # Legacy mode: own state variables when no shared processor
        self.gripper_pos = 0.0
        self.gripper_vel = 0.0
        self.gripper_acc = 0.0
        self.gripper_load = 0.0

        # Legacy mode: own subscriber (only if no shared processor AND use_feedback)
        self.state_sub = None
        if self.shared_processor is None and self.use_feedback and state_topic:
            self.state_sub = ChannelSubscriber(state_topic, Vector3_)
            self.state_sub.Init(None, 10)
            print(f"AlohaHandStateProcessor: Subscribing to feedback topic '{state_topic}' (legacy mode)")
        elif self.shared_processor is None:
            print(f"AlohaHandStateProcessor: Using command echo (no feedback subscription)")
    
    def _prepare_low_state(self) -> np.ndarray:
        """Prepare ALOHA gripper state.

        Returns:
            State array with shape (1, 4) containing:
            [position, velocity, torque_est, acceleration]
        """
        # Use shared processor if available (recommended path)
        if self.shared_processor is not None:
            return self.shared_processor.get_state_for_hand(self.is_left)

        # Legacy mode: own subscriber
        if self.use_feedback and self.state_sub:
            # Use short timeout to avoid blocking and exceptions when topic unavailable
            state_msg = self.state_sub.Read(timeout=0.001)
            if state_msg:
                if self.is_left:
                    self.gripper_pos = state_msg.x
                else:
                    self.gripper_pos = state_msg.y
                self.gripper_load = state_msg.z
                self.gripper_vel = 0.0
                self.gripper_acc = 0.0

        # Return state in same format as HandStateProcessor for compatibility
        state_data = np.array([
            self.gripper_pos,
            self.gripper_vel,
            self.gripper_load,
            self.gripper_acc,
        ]).astype(np.float64).reshape(1, -1)

        return state_data

    def update_state_from_command(self, commanded_pos: float):
        """Update internal state estimate from commanded position.

        Used when feedback is not available.

        Args:
            commanded_pos: Commanded gripper position (0.0 to 0.065)
        """
        # Use shared processor if available
        if self.shared_processor is not None:
            self.shared_processor.update_state_from_command(self.is_left, commanded_pos)
            return

        # Legacy mode
        self.gripper_pos = np.clip(commanded_pos, 0.0, 0.065)
        self.gripper_vel = 0.0
        self.gripper_acc = 0.0
        self.gripper_load = 0.0

    def close(self):
        """Close DDS subscriber if using feedback (legacy mode only).

        Note: If using shared_processor, close that separately.
        """
        try:
            if self.state_sub:
                self.state_sub.Close()
        except Exception as e:
            print(f"[AlohaHandStateProcessor] Error closing subscriber: {e}")
