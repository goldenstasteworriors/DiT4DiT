"""Keyboard-based arm control streamer for simulation teleop.

Controls:
    ARM POSITION:
        T/G: Forward/Backward (X-axis)
        R/Y: Left/Right (Y-axis)
        F/H: Up/Down (Z-axis)

    ARM ROTATION (numpad):
        7/9: Roll left/right
        4/6: Pitch up/down
        1/3: Yaw left/right

    GRIPPER:
        Space: Toggle gripper open/close

    ARM SELECTION:
        Tab: Switch between left/right arm

    BASE NAVIGATION:
        W/S: Forward/backward velocity
        A/D: Strafe left/right
        Q/E: Rotate left/right
        Z: Stop all movement

    DATA COLLECTION:
        C: Start/stop recording episode
        X: Discard current episode
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput

# Finger indices in the 25-element fingertip array (matching hand tracking format)
_THUMB_IDX = 4
_INDEX_IDX = 9
_MIDDLE_IDX = 14
_RING_IDX = 19
_PINKY_IDX = 24

# Gripper threshold distance (meters) - fingers closer than this are "closed"
_GRIPPER_CLOSE_DIST = 0.01
_GRIPPER_OPEN_DIST = 0.10


class KeyboardArmStreamer(BaseStreamer):
    """Keyboard streamer providing full arm teleop control for simulation."""

    # Movement parameters
    POS_STEP = 0.02  # 2cm per keypress
    ROT_STEP = np.deg2rad(10)  # 10 degrees per keypress
    VEL_STEP = 0.05  # velocity increment
    MAX_LIN_VEL = 0.3
    MAX_ANG_VEL = 0.5
    DEFAULT_BASE_HEIGHT = 0.74

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_streaming = False
        self._pending_keys: list[str] = []
        self._activated = False

        # Active arm selection (True = left, False = right)
        self._use_left_arm = True

        # Wrist poses (position + quaternion wxyz)
        self._left_wrist_pos = np.array([0.25, 0.25, 0.15])
        self._right_wrist_pos = np.array([0.25, -0.25, 0.15])
        self._left_wrist_quat = R.from_euler("xyz", [-90, 0, -90], degrees=True).as_quat(
            scalar_first=True
        )
        self._right_wrist_quat = R.from_euler("xyz", [-90, 0, 90], degrees=True).as_quat(
            scalar_first=True
        )

        # Gripper state (True = closed)
        self._left_gripper_closed = False
        self._right_gripper_closed = False

        # Navigation state
        self._lin_vel = np.zeros(2)  # [vx, vy]
        self._ang_vel = 0.0
        self._base_height = self.DEFAULT_BASE_HEIGHT

        # Data collection flags (edge-triggered, reset after read)
        self._toggle_data_collection = False
        self._toggle_data_abort = False

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    def start_streaming(self):
        self._is_streaming = True
        print("\n[Keyboard Teleop] T/G R/Y F/H=move  Space=grip  Tab=switch arm  C=record X=discard")

    def stop_streaming(self):
        self._is_streaming = False
        print("[Keyboard Teleop] Stopped")

    def handle_keyboard_button(self, key: str):
        """Receive keyboard input from KeyboardDispatcher."""
        self._pending_keys.append(key)

    def get(self) -> StreamerOutput:
        """Process pending keyboard input and return current teleop state."""
        if not self._is_streaming:
            return StreamerOutput()

        # Process all pending keys
        while self._pending_keys:
            self._process_key(self._pending_keys.pop(0))

        # Build output
        left_wrist_T = self._build_transform(self._left_wrist_pos, self._left_wrist_quat)
        right_wrist_T = self._build_transform(self._right_wrist_pos, self._right_wrist_quat)

        # Consume edge-triggered flags
        toggle_collection = self._toggle_data_collection
        toggle_abort = self._toggle_data_abort
        self._toggle_data_collection = False
        self._toggle_data_abort = False

        # Auto-activate on first call
        should_activate = not self._activated
        if should_activate:
            self._activated = True

        return StreamerOutput(
            ik_data={
                "left_wrist": left_wrist_T,
                "right_wrist": right_wrist_T,
                "left_fingers": {"position": self._generate_finger_data(self._left_gripper_closed)},
                "right_fingers": {"position": self._generate_finger_data(self._right_gripper_closed)},
            },
            control_data={
                "base_height_command": self._base_height,
                "navigate_cmd": [self._lin_vel[0], self._lin_vel[1], self._ang_vel],
                "toggle_policy_action": False,
                "torso_orientation_rpy": [0.0, 0.0, 0.0],
            },
            teleop_data={
                "toggle_activation": should_activate,
            },
            data_collection_data={
                "toggle_data_collection": toggle_collection,
                "toggle_data_abort": toggle_abort,
            },
            source="keyboard",
        )

    def _process_key(self, key: str):
        """Process a single key press and update internal state."""
        key = key.lower()

        # Arm switching (Tab)
        if key in ("\t", "tab"):
            self._use_left_arm = not self._use_left_arm
            print(f"\n[{'LEFT' if self._use_left_arm else 'RIGHT'} arm]")
            return

        # Get active arm state
        pos = self._left_wrist_pos if self._use_left_arm else self._right_wrist_pos

        # Position control: T/G (X), R/Y (Y), F/H (Z)
        if key == "t":
            pos[0] += self.POS_STEP
        elif key == "g":
            pos[0] -= self.POS_STEP
        elif key == "r":
            pos[1] += self.POS_STEP
        elif key == "y":
            pos[1] -= self.POS_STEP
        elif key == "f":
            pos[2] += self.POS_STEP
        elif key == "h":
            pos[2] -= self.POS_STEP

        # Rotation control: numpad 7/9 (roll), 4/6 (pitch), 1/3 (yaw)
        elif key == "7":
            self._rotate_active_wrist("x", self.ROT_STEP)
        elif key == "9":
            self._rotate_active_wrist("x", -self.ROT_STEP)
        elif key == "4":
            self._rotate_active_wrist("y", self.ROT_STEP)
        elif key == "6":
            self._rotate_active_wrist("y", -self.ROT_STEP)
        elif key == "1":
            self._rotate_active_wrist("z", self.ROT_STEP)
        elif key == "3":
            self._rotate_active_wrist("z", -self.ROT_STEP)

        # Gripper toggle (Space)
        elif key in (" ", "space"):
            if self._use_left_arm:
                self._left_gripper_closed = not self._left_gripper_closed
                state = "CLOSED" if self._left_gripper_closed else "OPEN"
            else:
                self._right_gripper_closed = not self._right_gripper_closed
                state = "CLOSED" if self._right_gripper_closed else "OPEN"
            print(f"\n[Gripper {state}]")

        # Base navigation: WASD (linear), QE (angular), Z (stop)
        elif key == "w":
            self._lin_vel[0] = min(self._lin_vel[0] + self.VEL_STEP, self.MAX_LIN_VEL)
        elif key == "s":
            self._lin_vel[0] = max(self._lin_vel[0] - self.VEL_STEP, -self.MAX_LIN_VEL)
        elif key == "a":
            self._lin_vel[1] = min(self._lin_vel[1] + self.VEL_STEP, self.MAX_LIN_VEL)
        elif key == "d":
            self._lin_vel[1] = max(self._lin_vel[1] - self.VEL_STEP, -self.MAX_LIN_VEL)
        elif key == "q":
            self._ang_vel = min(self._ang_vel + self.VEL_STEP, self.MAX_ANG_VEL)
        elif key == "e":
            self._ang_vel = max(self._ang_vel - self.VEL_STEP, -self.MAX_ANG_VEL)
        elif key == "z":
            self._lin_vel[:] = 0
            self._ang_vel = 0
            print("\n[Navigation stopped]")

        # Data collection: C (record), X (discard)
        elif key == "c":
            self._toggle_data_collection = True
            print("\n[Recording toggled]")
        elif key == "x":
            self._toggle_data_abort = True
            print("\n[Episode discarded]")

    def _rotate_active_wrist(self, axis: str, angle: float):
        """Apply incremental rotation to the active wrist."""
        if self._use_left_arm:
            current = R.from_quat(self._left_wrist_quat, scalar_first=True)
            self._left_wrist_quat = (R.from_euler(axis, angle) * current).as_quat(scalar_first=True)
        else:
            current = R.from_quat(self._right_wrist_quat, scalar_first=True)
            self._right_wrist_quat = (R.from_euler(axis, angle) * current).as_quat(scalar_first=True)

    @staticmethod
    def _build_transform(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Build 4x4 homogeneous transform from position and quaternion (wxyz)."""
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
        return T

    @staticmethod
    def _generate_finger_data(gripper_closed: bool) -> np.ndarray:
        """Generate finger position data for G1 gripper IK.

        The G1 gripper IK solver calculates thumb-to-finger distances.
        Distance < 0.05m triggers finger closure.
        """
        # Initialize all 25 finger joints as identity transforms
        fingertips = np.tile(np.eye(4), (25, 1, 1))

        # Set fingertip positions relative to thumb
        dist = _GRIPPER_CLOSE_DIST if gripper_closed else _GRIPPER_OPEN_DIST
        fingertips[_THUMB_IDX, :3, 3] = [0.0, 0.0, 0.0]
        for idx in (_INDEX_IDX, _MIDDLE_IDX, _RING_IDX, _PINKY_IDX):
            fingertips[idx, :3, 3] = [dist, 0.0, 0.0]

        return fingertips
