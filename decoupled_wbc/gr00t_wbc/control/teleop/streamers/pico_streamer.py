import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.teleop.device.pico.xr_client import XrClient
from gr00t_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


class PicoStreamer(BaseStreamer):
    def __init__(self):
        self.xr_client = XrClient()
        self.run_pico_service()

        self.reset_status()

    def run_pico_service(self):
        # Run the pico service
        self.pico_service_pid = subprocess.Popen(
            ["bash", "/opt/apps/roboticsservice/runService.sh"]
        )
        print(f"Pico service running with pid {self.pico_service_pid.pid}")

    def stop_pico_service(self):
        # find pid and kill it
        if self.pico_service_pid:
            subprocess.Popen(["kill", "-9", str(self.pico_service_pid.pid)])
            print(f"Pico service killed with pid {self.pico_service_pid.pid}")
        else:
            print("Pico service not running")

    def reset_status(self):
        self.current_base_height = 0.74  # Initial base height, 0.74m (standing height)
        self.target_standing_height = 0.74  # Base standing height for head-tracking offset
        self.toggle_policy_action_last = False
        self.toggle_activation_last = False
        self.toggle_data_collection_last = False
        self.toggle_delete_episode_last = False
        # Head tracking for torso orientation (calibrated on first frame after reset)
        self.head_calibrated = False
        self.head_init_rotation = None   # Headset rotation at calibration (z-up world frame)
        self.head_init_height_y = None   # Headset Y position at calibration (Y-up raw frame)

    def start_streaming(self):
        pass

    def stop_streaming(self):
        self.xr_client.close()

    def get(self) -> StreamerOutput:
        pico_data = self._get_pico_data()

        raw_data = self._generate_unified_raw_data(pico_data)
        
        # Print only non-default values
        out = []
        nav = raw_data.control_data["navigate_cmd"]
        h = raw_data.control_data["base_height_command"]
        rpy = raw_data.control_data['torso_orientation_rpy']
        
        if np.any(np.abs(nav) > 0.01): out.append(f"Nav:{nav}")
        if abs(h - 0.74) > 0.01: out.append(f"H:{h:.2f}")
        if np.any(np.abs(rpy) > 0.01): out.append(f"RPY:{np.rad2deg(rpy)}")
        
        btns = [k[0] for k in ['A', 'X', 'Y'] if pico_data.get(k)]
        if pico_data.get('left_menu_button'): btns.append('LM')
        if pico_data.get('right_menu_button'): btns.append('RM')
        if btns: out.append(f"Btn:{btns}")
        
        if max(pico_data['left_trigger'], pico_data['right_trigger']) > 0.01:
            out.append(f"Trig:{pico_data['left_trigger']:.1f}/{pico_data['right_trigger']:.1f}")
        if max(pico_data['left_grip'], pico_data['right_grip']) > 0.01:
            out.append(f"Grip:{pico_data['left_grip']:.1f}/{pico_data['right_grip']:.1f}")
        
        if raw_data.control_data['toggle_policy_action'] or raw_data.teleop_data['toggle_activation']:
            out.append(f"Tog:{int(raw_data.control_data['toggle_policy_action'])}/{int(raw_data.teleop_data['toggle_activation'])}")
        if raw_data.data_collection_data['toggle_data_collection']:
            out.append(f"Rec:{int(raw_data.data_collection_data['toggle_data_collection'])}")
        
        if out: print(" ".join(out))
        
        return raw_data

    def __del__(self):
        pass

    def _get_pico_data(self):
        pico_data = {}

        # Get the pose of the left and right controllers and the headset
        pico_data["left_pose"] = self.xr_client.get_pose_by_name("left_controller")
        pico_data["right_pose"] = self.xr_client.get_pose_by_name("right_controller")
        pico_data["head_pose"] = self.xr_client.get_pose_by_name("headset")

        # Get key value of the left and right controllers
        pico_data["left_trigger"] = self.xr_client.get_key_value_by_name("left_trigger")
        pico_data["right_trigger"] = self.xr_client.get_key_value_by_name("right_trigger")
        pico_data["left_grip"] = self.xr_client.get_key_value_by_name("left_grip")
        pico_data["right_grip"] = self.xr_client.get_key_value_by_name("right_grip")

        # Get button state of the left and right controllers
        pico_data["A"] = self.xr_client.get_button_state_by_name("A")
        pico_data["X"] = self.xr_client.get_button_state_by_name("X")
        pico_data["Y"] = self.xr_client.get_button_state_by_name("Y")
        pico_data["left_menu_button"] = self.xr_client.get_button_state_by_name("left_menu_button")
        pico_data["right_menu_button"] = self.xr_client.get_button_state_by_name(
            "right_menu_button"
        )
        pico_data["left_axis_click"] = self.xr_client.get_button_state_by_name("left_axis_click")
        pico_data["right_axis_click"] = self.xr_client.get_button_state_by_name("right_axis_click")

        # Get the timestamp of the left and right controllers
        pico_data["timestamp"] = self.xr_client.get_timestamp_ns()

        # Get the hand tracking state of the left and right controllers
        pico_data["left_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("left")
        pico_data["right_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("right")

        # Get the joystick state of the left and right controllers
        pico_data["left_joystick"] = self.xr_client.get_joystick_state("left")
        pico_data["right_joystick"] = self.xr_client.get_joystick_state("right")

        # Get the motion tracker data
        pico_data["motion_tracker_data"] = self.xr_client.get_motion_tracker_data()

        # Get the body tracking data
        pico_data["body_tracking_data"] = self.xr_client.get_body_tracking_data()

        return pico_data

    def _generate_unified_raw_data(self, pico_data):
        # Get controller position and orientation in z up world frame
        left_controller_T = self._process_xr_pose(pico_data["left_pose"], pico_data["head_pose"])
        right_controller_T = self._process_xr_pose(pico_data["right_pose"], pico_data["head_pose"])

        # Get navigation commands
        DEAD_ZONE = 0.1
        MAX_LINEAR_VEL_X = 0.1 # m/s
        MAX_LINEAR_VEL_Y = 0.1  # m/s
        MAX_ANGULAR_VEL = 0.5   # rad/s

        fwd_bwd_input = pico_data["left_joystick"][1]
        strafe_input = -pico_data["left_joystick"][0]
        
        # === Torso orientation from headset tracking ===
        # Extract headset rotation in z-up world frame
        headset_quat = np.array(pico_data["head_pose"])[3:]  # x, y, z, w
        if np.allclose(headset_quat, 0):
            headset_quat = np.array([0, 0, 0, 1])
        headset_rot = R_HEADSET_TO_WORLD @ R.from_quat(headset_quat).as_matrix() @ R_HEADSET_TO_WORLD.T

        # Calibrate on first frame: record initial headset orientation and height
        if not self.head_calibrated:
            self.head_init_rotation = headset_rot
            self.head_init_height_y = float(np.array(pico_data["head_pose"])[1])  # Y-up raw
            self.head_calibrated = True
            print(f"[PicoStreamer] Head calibrated! Init height Y: {self.head_init_height_y:.3f}")

        # Compute delta RPY relative to calibration pose
        delta_rot = self.head_init_rotation.T @ headset_rot
        delta_rpy = R.from_matrix(delta_rot).as_euler("xyz")  # [roll, pitch, yaw]
        MAX_TORSO_ANGLE = np.deg2rad(30)
        torso_rpy = np.clip(delta_rpy, -MAX_TORSO_ANGLE, MAX_TORSO_ANGLE)

        # === Base height from headset vertical position ===
        current_height_y = float(np.array(pico_data["head_pose"])[1])  # Y-up raw
        delta_height = current_height_y - self.head_init_height_y
        self.current_base_height = np.clip(
            self.target_standing_height + delta_height, 0.24, 0.74
        )

        # === Navigation commands ===
        if not pico_data["left_menu_button"]:
            yaw_input = -pico_data["right_joystick"][0]
            ang_vel_z = self._apply_dead_zone(yaw_input, DEAD_ZONE) * MAX_ANGULAR_VEL
        else:
            ang_vel_z = 0.0  # Left Menu held: disable rotation to avoid accidental turns

        lin_vel_x = self._apply_dead_zone(fwd_bwd_input, DEAD_ZONE) * MAX_LINEAR_VEL_X
        lin_vel_y = self._apply_dead_zone(strafe_input, DEAD_ZONE) * MAX_LINEAR_VEL_Y

        # X/Y buttons adjust the base standing height (offsets head-tracked height)
        height_increment = 0.01
        if pico_data["Y"] and self.target_standing_height < 0.74:
            self.target_standing_height = min(self.target_standing_height + height_increment, 0.74)
        elif pico_data["X"] and self.target_standing_height > 0.24:
            self.target_standing_height = max(self.target_standing_height - height_increment, 0.24)

        # Get gripper commands
        left_fingers = self._generate_finger_data(pico_data, "left")
        right_fingers = self._generate_finger_data(pico_data, "right")

        # Get activation commands
        # SAFETY: Require left_menu + grip combos to toggle state
        # (prevents accidental toggles; right menu is reserved by Pico system)
        # Toggle lower-body policy action: left_menu + left_trigger + left_grip
        # Toggle upper-body teleop activation: left_menu + right_grip
        left_menu = pico_data["left_menu_button"]
        toggle_policy_action_tmp = (
            left_menu and
            pico_data["left_trigger"] > 0.98 and
            pico_data["left_grip"] > 0.98
        )
        toggle_activation_tmp = (
            left_menu and
            pico_data["right_grip"] > 0.98
        )

        if self.toggle_policy_action_last != toggle_policy_action_tmp:
            toggle_policy_action = toggle_policy_action_tmp
        else:
            toggle_policy_action = False
        self.toggle_policy_action_last = toggle_policy_action_tmp

        # Re-calibrate head tracking on policy activation to zero out any yaw drift
        if toggle_policy_action:
            self.head_calibrated = False

        if self.toggle_activation_last != toggle_activation_tmp:
            toggle_activation = toggle_activation_tmp
        else:
            toggle_activation = False
        self.toggle_activation_last = toggle_activation_tmp

        # Get data collection commands (rising edge detection).
        # Button A starts/stops recording; the actual signal is only emitted after
        # the operator confirms by re-activating teleop with left_menu + right_grip
        # (handled in TeleopPolicy). See README §7.
        toggle_data_collection_tmp = pico_data["A"]

        toggle_data_collection = False
        if toggle_data_collection_tmp and not self.toggle_data_collection_last:
            toggle_data_collection = True
            print(f"[PicoStreamer] Button A pressed - toggle_data_collection=True")
        self.toggle_data_collection_last = toggle_data_collection_tmp

        # Episode deletion: Y + Left Menu (rising edge detection)
        delete_episode_tmp = pico_data["Y"] and pico_data["left_menu_button"]

        toggle_delete_episode = False
        if delete_episode_tmp and not self.toggle_delete_episode_last:
            toggle_delete_episode = True
            print(f"[PicoStreamer] Y + Left Menu pressed - toggle_delete_episode=True")
        self.toggle_delete_episode_last = delete_episode_tmp

        return StreamerOutput(
            ik_data={
                "left_wrist": left_controller_T,
                "right_wrist": right_controller_T,
                "left_fingers": left_fingers,  # Already a dict with "position" and "trigger"
                "right_fingers": right_fingers,  # Already a dict with "position" and "trigger"
            },
            control_data={
                "base_height_command": self.current_base_height,
                "navigate_cmd": [lin_vel_x, lin_vel_y, ang_vel_z],
                "toggle_policy_action": toggle_policy_action,
                "torso_orientation_rpy": [float(torso_rpy[0]), float(torso_rpy[1]), float(torso_rpy[2])],
            },
            teleop_data={
                "toggle_activation": toggle_activation,
            },
            data_collection_data={
                "toggle_data_collection": toggle_data_collection,
                "toggle_data_abort": False,
                "toggle_delete_episode": toggle_delete_episode,
            },
            source="pico",
        )

    def _process_xr_pose(self, controller_pose, headset_pose):
        # Convert controller pose to x, y, z, w quaternion
        xr_pose_xyz = np.array(controller_pose)[:3]  # x, y, z
        xr_pose_quat = np.array(controller_pose)[3:]  # x, y, z, w

        # Handle all-zero quaternion case by using identity quaternion
        if np.allclose(xr_pose_quat, 0):
            xr_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        xr_pose_xyz = R_HEADSET_TO_WORLD @ xr_pose_xyz
        xr_pose_rotation = R.from_quat(xr_pose_quat).as_matrix()
        xr_pose_rotation = R_HEADSET_TO_WORLD @ xr_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Convert headset pose to x, y, z, w quaternion
        headset_pose_xyz = np.array(headset_pose)[:3]
        headset_pose_quat = np.array(headset_pose)[3:]

        if np.allclose(headset_pose_quat, 0):
            headset_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        headset_pose_xyz = R_HEADSET_TO_WORLD @ headset_pose_xyz
        headset_pose_rotation = R.from_quat(headset_pose_quat).as_matrix()
        headset_pose_rotation = R_HEADSET_TO_WORLD @ headset_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Calculate the delta between the controller and headset positions
        xr_pose_xyz_delta = xr_pose_xyz - headset_pose_xyz

        # Calculate the yaw of the headset
        R_headset_to_world = R.from_matrix(headset_pose_rotation)
        headset_pose_yaw = R_headset_to_world.as_euler("xyz")[2]  # Extract yaw (Z-axis rotation)
        inverse_yaw_rotation = R.from_euler("z", -headset_pose_yaw).as_matrix()

        # Align with headset yaw to controller position delta and rotation
        xr_pose_xyz_delta_compensated = inverse_yaw_rotation @ xr_pose_xyz_delta
        xr_pose_rotation_compensated = inverse_yaw_rotation @ xr_pose_rotation

        xr_pose_T = np.eye(4)
        xr_pose_T[:3, :3] = xr_pose_rotation_compensated
        xr_pose_T[:3, 3] = xr_pose_xyz_delta_compensated
        return xr_pose_T

    def _apply_dead_zone(self, value, dead_zone):
        """Apply dead zone and normalize."""
        if abs(value) < dead_zone:
            return 0.0
        sign = 1 if value > 0 else -1
        # Normalize the output to be between -1 and 1 after dead zone
        return sign * (abs(value) - dead_zone) / (1.0 - dead_zone)

    def _generate_finger_data(self, pico_data, hand):
        """Generate finger position data.
        
        Returns a dictionary with:
        - 'position': 25x4x4 array for three-finger hand (legacy)
        - 'trigger': continuous trigger value 0.0-1.0 for ALOHA grippers
        """
        fingertips = np.zeros([25, 4, 4])

        thumb = 0
        index = 5
        middle = 10
        ring = 15

        # Control thumb based on shoulder button state (index 4 is thumb tip)
        fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
        if not pico_data["left_menu_button"]:
            if pico_data[f"{hand}_trigger"] > 0.5 and not pico_data[f"{hand}_grip"] > 0.5:
                fingertips[4 + index, 0, 3] = 1.0  # close index
            elif pico_data[f"{hand}_trigger"] > 0.5 and pico_data[f"{hand}_grip"] > 0.5:
                fingertips[4 + middle, 0, 3] = 1.0  # close middle
            elif not pico_data[f"{hand}_trigger"] > 0.5 and pico_data[f"{hand}_grip"] > 0.5:
                fingertips[4 + ring, 0, 3] = 1.0  # close ring

        # Include trigger value for ALOHA gripper control
        return {
            "position": fingertips,
            "trigger": pico_data[f"{hand}_trigger"]
        }


if __name__ == "__main__":
    # from gr00t_wbc.control.utils.debugger import wait_for_debugger
    # wait_for_debugger()

    streamer = PicoStreamer()
    streamer.start_streaming()
    while True:
        raw_data = streamer.get()
        print(
            f"left_wrist: {raw_data.ik_data['left_wrist']}, right_wrist: {raw_data.ik_data['right_wrist']}"
        )
        time.sleep(0.1)
