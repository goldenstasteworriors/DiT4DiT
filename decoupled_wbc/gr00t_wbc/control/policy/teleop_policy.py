from contextlib import contextmanager
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.base.policy import Policy
from gr00t_wbc.control.robot_model import RobotModel
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from gr00t_wbc.control.teleop.teleop_streamer import TeleopStreamer


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes for status display."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # State colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"

    # Background colors for emphasis
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_CYAN = "\033[46m"


class TeleopPolicy(Policy):
    """
    Robot-agnostic teleop policy.
    Clean separation: IK processing vs command passing.
    All robot-specific properties are abstracted through robot_model and hand_ik_solvers.
    """

    def __init__(
        self,
        body_control_device: str,
        hand_control_device: str,
        robot_model: RobotModel,
        retargeting_ik: TeleopRetargetingIK,
        body_streamer_ip: str = "192.168.?.?",
        body_streamer_keyword: str = "shoulder",
        enable_real_device: bool = True,
        replay_data_path: Optional[str] = None,
        replay_speed: float = 1.0,
        wait_for_activation: int = 5,
        activate_keyboard_listener: bool = True,
    ):
        if activate_keyboard_listener:
            from gr00t_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber

            self.keyboard_listener = KeyboardListenerSubscriber()
        else:
            self.keyboard_listener = None

        self.wait_for_activation = wait_for_activation

        self.teleop_streamer = TeleopStreamer(
            robot_model=robot_model,
            body_control_device=body_control_device,
            hand_control_device=hand_control_device,
            enable_real_device=enable_real_device,
            body_streamer_ip=body_streamer_ip,
            body_streamer_keyword=body_streamer_keyword,
            replay_data_path=replay_data_path,
            replay_speed=replay_speed,
        )
        self.robot_model = robot_model
        self.retargeting_ik = retargeting_ik
        self.is_active = False

        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

        # Initial pose movement state
        self._commanding_initial_pose = False
        self._initial_upper_body_pose = robot_model.get_initial_upper_body_pose()
        self._initial_pose_start_time = None  # When we started commanding initial pose
        self._waiting_for_manual_activation = (
            False  # Waiting for operator to press Left Menu + Right Trigger
        )
        self._next_toggle_starts_recording = (
            True  # Alternates: True=start recording, False=stop recording
        )
        self._pending_signal_type = None  # "toggle_data_collection" or "toggle_data_abort"
        self._teleop_reactivation_time = (
            None  # When teleop was just re-enabled (for smooth transition)
        )

    def set_goal(self, goal: dict[str, any]):
        # The current teleop policy doesn't take higher level commands yet.
        pass

    def _print_status_banner(self, status_type: str):
        """Print a large, colored status banner for initial pose movement.

        Args:
            status_type: One of "prepare_start", "prepare_save", or "prepare_discard"
        """
        banner_width = 80

        if status_type == "prepare_start":
            color = Colors.CYAN
            bg_color = Colors.BG_CYAN
            status_text = "🤖 MOVING TO INITIAL POSE - PREPARING TO START RECORDING"
        elif status_type == "prepare_save":
            color = Colors.YELLOW
            bg_color = Colors.BG_YELLOW
            status_text = "🤖 MOVING TO INITIAL POSE - PREPARING TO SAVE EPISODE"
        elif status_type == "prepare_discard":
            color = Colors.RED
            bg_color = Colors.BG_RED
            status_text = "🤖 MOVING TO INITIAL POSE - DISCARDING EPISODE"
        else:
            color = Colors.RESET
            bg_color = ""
            status_text = f"Unknown status: {status_type}"

        # Print banner
        print(f"\n{color}{Colors.BOLD}{'=' * banner_width}")
        print(f"{bg_color}{' ' * banner_width}{Colors.RESET}")
        print(f"{bg_color}{status_text.center(banner_width)}{Colors.RESET}")
        print(f"{bg_color}{' ' * banner_width}{Colors.RESET}")
        print(f"{color}{'=' * banner_width}{Colors.RESET}\n")

    def get_action(self) -> dict[str, any]:
        # Get structured data
        streamer_output = self.teleop_streamer.get_streamer_data()

        # Handle activation using teleop_data commands
        self.check_activation(
            streamer_output.teleop_data, wait_for_activation=self.wait_for_activation
        )

        action = {}

        # Process streamer data if active
        if self.is_active and streamer_output.ik_data:
            body_data = streamer_output.ik_data["body_data"]
            left_hand_data = streamer_output.ik_data["left_hand_data"]
            right_hand_data = streamer_output.ik_data["right_hand_data"]

            left_wrist_name = self.robot_model.supplemental_info.hand_frame_names["left"]
            right_wrist_name = self.robot_model.supplemental_info.hand_frame_names["right"]
            self.latest_left_wrist_data = body_data[left_wrist_name]
            self.latest_right_wrist_data = body_data[right_wrist_name]
            self.latest_left_fingers_data = left_hand_data
            self.latest_right_fingers_data = right_hand_data

            # TODO: This stores the same data again
            ik_data = {
                "body_data": body_data,
                "left_hand_data": left_hand_data,
                "right_hand_data": right_hand_data,
            }
            action["ik_data"] = ik_data

        # Wrist poses (pos and quat)
        # TODO: This stores the same wrist poses in two different formats
        left_wrist_matrix = self.latest_left_wrist_data
        right_wrist_matrix = self.latest_right_wrist_data
        left_wrist_pose = np.concatenate(
            [
                left_wrist_matrix[:3, 3],
                R.from_matrix(left_wrist_matrix[:3, :3]).as_quat(scalar_first=True),
            ]
        )
        right_wrist_pose = np.concatenate(
            [
                right_wrist_matrix[:3, 3],
                R.from_matrix(right_wrist_matrix[:3, :3]).as_quat(scalar_first=True),
            ]
        )

        # Combine IK results with control commands (no teleop_data commands)
        action.update(
            {
                "left_wrist": self.latest_left_wrist_data,
                "right_wrist": self.latest_right_wrist_data,
                "left_fingers": self.latest_left_fingers_data,
                "right_fingers": self.latest_right_fingers_data,
                "wrist_pose": np.concatenate([left_wrist_pose, right_wrist_pose]),
                **streamer_output.control_data,  # Only control commands pass through
                # Note: data_collection_data is added conditionally below
            }
        )

        # ========== INITIAL POSE & DATA COLLECTION WORKFLOW ==========
        #
        # START RECORDING (Button A - first press):
        #   1. Turn OFF teleop (self.is_active = False)
        #   2. Robot moves to initial pose autonomously (continuously commanded)
        #   3. Operator manually turns teleop back ON (Left Menu + Right Trigger) when ready
        #   4. Send toggle_data_collection signal to start recording
        #
        # STOP RECORDING (Button A - second press):
        #   1. Turn OFF teleop (self.is_active = False)
        #   2. Robot moves to initial pose autonomously (continuously commanded)
        #   3. Operator manually turns teleop back ON (Left Menu + Right Trigger) to confirm convergence
        #   4. Send toggle_data_collection signal to save episode
        #
        # DISCARD (Button B):
        #   1. Turn OFF teleop (self.is_active = False)
        #   2. Robot moves to initial pose autonomously (continuously commanded)
        #   3. Operator manually turns teleop back ON (Left Menu + Right Trigger) to confirm
        #   4. Send toggle_data_abort signal to discard episode
        #
        # ==============================================================

        button_a_pressed = streamer_output.data_collection_data.get("toggle_data_collection", False)
        button_b_pressed = streamer_output.data_collection_data.get("toggle_data_abort", False)
        button_delete_pressed = streamer_output.data_collection_data.get("toggle_delete_episode", False)

        # Handle Button A press (start/stop recording)
        if (
            button_a_pressed
            and not self._commanding_initial_pose
            and not self._waiting_for_manual_activation
        ):
            # Start the initial pose sequence
            self.is_active = False  # Disable teleop immediately
            self._commanding_initial_pose = True
            self._initial_pose_start_time = (
                time.monotonic()
            )  # Record start time for smooth interpolation
            self._waiting_for_manual_activation = True  # Always wait for manual activation
            self._pending_signal_type = "toggle_data_collection"  # Store which signal to send

            if self._next_toggle_starts_recording:
                self._print_status_banner("prepare_start")
                print(
                    f"{Colors.CYAN}[TeleopPolicy] Teleop DISABLED - Robot moving to initial pose{Colors.RESET}"
                )
                print(
                    f"{Colors.CYAN}[TeleopPolicy] Press {Colors.BOLD}Left Menu + Right Trigger{Colors.RESET}{Colors.CYAN} to START recording{Colors.RESET}"
                )
            else:
                self._print_status_banner("prepare_save")
                print(
                    f"{Colors.YELLOW}[TeleopPolicy] Teleop DISABLED - Robot moving to initial pose{Colors.RESET}"
                )
                print(
                    f"{Colors.YELLOW}[TeleopPolicy] Press {Colors.BOLD}Left Menu + Right Trigger{Colors.RESET}{Colors.YELLOW} to SAVE episode{Colors.RESET}"
                )

            self._next_toggle_starts_recording = (
                not self._next_toggle_starts_recording
            )  # Toggle for next press

        # Handle Button B press (discard)
        if (
            button_b_pressed
            and not self._commanding_initial_pose
            and not self._waiting_for_manual_activation
        ):
            self.is_active = False  # Disable teleop immediately
            self._commanding_initial_pose = True
            self._initial_pose_start_time = (
                time.monotonic()
            )  # Record start time for smooth interpolation
            self._waiting_for_manual_activation = True
            self._pending_signal_type = "toggle_data_abort"  # Store which signal to send
            self._next_toggle_starts_recording = (
                True  # Reset: next Button A should START new recording
            )
            self._print_status_banner("prepare_discard")
            print(
                f"{Colors.RED}[TeleopPolicy] Teleop DISABLED - Robot moving to initial pose{Colors.RESET}"
            )
            print(
                f"{Colors.RED}[TeleopPolicy] Press {Colors.BOLD}Left Menu + Right Trigger{Colors.RESET}{Colors.RED} to DISCARD episode{Colors.RESET}"
            )

        # Check if operator manually re-enabled teleop
        # Note: check_activation() (called earlier) handles Left Menu + Right Trigger and toggles self.is_active
        # When operator presses Left Menu + Right Trigger, self.is_active becomes True and calibration happens
        if self._waiting_for_manual_activation and self.is_active:
            if self._pending_signal_type is None:
                print(
                    f"{Colors.RED}[TeleopPolicy] ERROR: Operator re-enabled teleop but no pending signal stored!{Colors.RESET}"
                )
            else:
                print(f"\n{Colors.GREEN}{Colors.BOLD}{'=' * 80}")
                print(f"✓ TELEOP RE-ENABLED - Transition to operator control (smooth 2s warmup)")
                print(f"{'=' * 80}{Colors.RESET}\n")
                print(
                    f"{Colors.GREEN}[TeleopPolicy] Sending stored signal: {self._pending_signal_type}{Colors.RESET}"
                )

                # Send the stored signal to DataExporter
                if self._pending_signal_type == "toggle_data_collection":
                    action.update({"toggle_data_collection": True, "toggle_data_abort": False})
                elif self._pending_signal_type == "toggle_data_abort":
                    action.update({"toggle_data_collection": False, "toggle_data_abort": True})

            # Record reactivation time for smooth transition from initial pose to teleop
            self._teleop_reactivation_time = time.monotonic()

            # Clear state regardless
            self._waiting_for_manual_activation = False
            self._commanding_initial_pose = False
            self._initial_pose_start_time = None
            self._pending_signal_type = None
        # Otherwise suppress data collection signals while commanding initial pose

        # Run retargeting IK
        if "ik_data" in action:
            self.retargeting_ik.set_goal(action["ik_data"])

        # Override with initial pose if commanding
        if self._commanding_initial_pose:
            action["target_upper_body_pose"] = self._initial_upper_body_pose
            # Mark this as initial pose movement for smooth interpolation in TeleopPolicyLoop
            action["commanding_initial_pose"] = True
            # Set fixed target time for smooth movement (computed from start time)
            # This ensures the target time doesn't keep getting pushed forward
            initial_pose_movement_duration = 3.0  # seconds for safe, slow movement
            action["initial_pose_target_time"] = (
                self._initial_pose_start_time + initial_pose_movement_duration
            )
        else:
            action["target_upper_body_pose"] = self.retargeting_ik.get_action()
            action["commanding_initial_pose"] = False
            # Pass reactivation time for smooth transition
            action["teleop_reactivation_time"] = self._teleop_reactivation_time

        # print(f"[DEBUG teleop_policy] Returning action with left_fingers: {action.get('left_fingers')}")
        # print(f"[DEBUG teleop_policy] Returning action with right_fingers: {action.get('right_fingers')}")

        # Pass through delete episode signal (Y + Left Menu button)
        if button_delete_pressed:
            action.update({"toggle_delete_episode": True})

        return action

    def close(self) -> bool:
        self.teleop_streamer.stop_streaming()
        return True

    def check_activation(self, teleop_data: dict, wait_for_activation: int = 5):
        """Activation logic only looks at teleop data commands"""
        key = self.keyboard_listener.read_msg() if self.keyboard_listener else ""
        toggle_activation_by_keyboard = key == "l"
        reset_teleop_policy_by_keyboard = key == "k"
        toggle_activation_by_teleop = teleop_data.get("toggle_activation", False)

        if reset_teleop_policy_by_keyboard:
            print("Resetting teleop policy")
            self.reset()

        if toggle_activation_by_keyboard or toggle_activation_by_teleop:
            self.is_active = not self.is_active
            if self.is_active:
                print("Starting teleop policy")

                if wait_for_activation > 0 and toggle_activation_by_keyboard:
                    print(f"Sleeping for {wait_for_activation} seconds before starting teleop...")
                    for i in range(wait_for_activation, 0, -1):
                        print(f"Starting in {i}...")
                        time.sleep(1)

                # dda: calibration logic should use current IK data
                self.teleop_streamer.calibrate()
                print("Teleop policy calibrated")
            else:
                print("Stopping teleop policy")

    @contextmanager
    def activate(self):
        try:
            yield self
        finally:
            self.close()

    def handle_keyboard_button(self, keycode):
        """Handle keyboard input with proper state toggle."""
        if keycode == "l":
            self.is_active = not self.is_active
        if keycode == "k":
            print("Resetting teleop policy")
            self.reset()

        # Forward keyboard input to body_streamer
        body_streamer = getattr(self.teleop_streamer, "body_streamer", None)
        if body_streamer and hasattr(body_streamer, "handle_keyboard_button"):
            body_streamer.handle_keyboard_button(keycode)

    def activate_policy(self, wait_for_activation: int = 5):
        """activate the teleop policy"""
        self.is_active = False
        self.check_activation(
            teleop_data={"toggle_activation": True}, wait_for_activation=wait_for_activation
        )

    def reset(self, wait_for_activation: int = 5, auto_activate: bool = False):
        """Reset the teleop policy to the initial state, and re-activate it."""
        self.teleop_streamer.reset()
        self.retargeting_ik.reset()
        self.is_active = False
        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}
        # Reset initial pose state
        self._commanding_initial_pose = False
        self._initial_pose_start_time = None
        self._waiting_for_manual_activation = False
        self._pending_signal_type = None
        self._teleop_reactivation_time = None

        if auto_activate:
            self.activate_policy(wait_for_activation)
