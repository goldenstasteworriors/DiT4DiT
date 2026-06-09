import time

import rclpy
import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from gr00t_wbc.control.main.teleop.configs.configs import TeleopConfig
from gr00t_wbc.control.policy.compact_replay_policy import CompactReplayPolicy
from gr00t_wbc.control.policy.lerobot_replay_policy import LerobotReplayPolicy
from gr00t_wbc.control.policy.teleop_policy import TeleopPolicy, Colors
from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_aloha_hand_ik_instantiation import (
    instantiate_g1_aloha_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from gr00t_wbc.control.utils.signal_handler import SignalHandler, is_shutdown_requested
from gr00t_wbc.control.utils.telemetry import Telemetry

import pdb

TELEOP_NODE_NAME = "TeleopPolicy"


def main(config: TeleopConfig):
    # Install signal handlers for graceful shutdown (Ctrl+C)
    signal_handler = SignalHandler()

    ros_manager = None
    teleop_policy = None

    try:
        ros_manager = ROSManager(node_name=TELEOP_NODE_NAME)
        node = ros_manager.node
    except KeyboardInterrupt:
        print("[TeleopPolicy] Interrupted during ROS initialization")
        return

    # Select robot model based on robot name and hand_type configuration
    robot_name = config.robot.lower()
    left_hand_ik_solver, right_hand_ik_solver = None, None

    if "g1" in robot_name:
        # Only initialize hand IK solvers and use hand-specific model if hands are enabled
        if config.with_hands:
            if config.hand_type == "aloha":
                robot_name = "g1_aloha"
                left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_aloha_hand_ik_solver()
                print(f"[TeleopPolicy] Using ALOHA gripper IK solvers (1-DOF)")
            else:
                # Default to three-finger hands
                left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
                print(f"[TeleopPolicy] Using three-finger hand IK solvers (7-DOF)")
        else:
            # Hands disabled - use default g1 model (no hand-specific variant)
            print(f"[TeleopPolicy] Hands disabled, no hand IK solvers")

        robot_type, robot_model = get_robot_type_and_model(
            robot=robot_name,
            high_elbow_pose=config.high_elbow_pose,
        )
    else:
        raise ValueError(f"Unsupported robot name: {config.robot}")

    if config.compact_replay_path:
        teleop_policy = CompactReplayPolicy(
            robot_model=robot_model, parquet_path=config.compact_replay_path
        )
    elif config.lerobot_replay_path:
        teleop_policy = LerobotReplayPolicy(
            robot_model=robot_model, parquet_path=config.lerobot_replay_path
        )
    else:
        print("running teleop policy, waiting teleop policy to be initialized...")
        retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=config.enable_visualization,
            body_active_joint_groups=["upper_body"],
        )
        teleop_policy = TeleopPolicy(
            robot_model=robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device=config.body_control_device,
            hand_control_device=config.hand_control_device,
            body_streamer_ip=config.body_streamer_ip,  # vive tracker, leap motion does not require
            body_streamer_keyword=config.body_streamer_keyword,
            enable_real_device=config.enable_real_device,
            replay_data_path=config.teleop_replay_path,
        )

    # Create a publisher for the navigation commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    # Create rate controller
    rate = node.create_rate(config.teleop_frequency)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds

    # Get the same fixed initial pose used for data collection
    initial_upper_body_pose = robot_model.get_initial_upper_body_pose()

    telemetry = Telemetry(window_size=100)

    try:
        print("[TeleopPolicy] Starting teleop loop (Ctrl+C to stop)")
        while rclpy.ok() and not is_shutdown_requested():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                # Get the current teleop action
                with telemetry.timer("get_action"):
                    data = teleop_policy.get_action()

                # Add timing information to the message
                t_now = time.monotonic()
                data["timestamp"] = t_now

                # Set target completion time - longer for initial pose movements, then match control frequency
                # Also use longer duration for smooth transition after teleop reactivation
                teleop_reactivation_time = data.get("teleop_reactivation_time", None)
                reactivation_warmup_duration = (
                    0.1  # seconds for smooth transition after reactivation
                )

                if iteration == 0:
                    # First iteration: move to fixed initial pose (same pose used for data collection)
                    # Override the teleop action with fixed initial pose
                    data["target_upper_body_pose"] = initial_upper_body_pose
                    data["target_time"] = t_now + time_to_get_to_initial_pose
                    print(
                        f"{Colors.CYAN}[TeleopPolicyLoop] Moving to initial pose (2 seconds){Colors.RESET}"
                    )
                elif data.get("commanding_initial_pose", False):
                    # During data collection: moving to initial pose for START/STOP/DISCARD
                    # TeleopPolicy has already computed a fixed target_time based on start time
                    # Use this to ensure smooth, consistent interpolation
                    data["target_time"] = data["initial_pose_target_time"]
                elif (
                    teleop_reactivation_time is not None
                    and (t_now - teleop_reactivation_time) < reactivation_warmup_duration
                ):
                    # Just reactivated teleop - use longer duration for smooth transition
                    # from initial pose to operator's current hand position
                    remaining_warmup = reactivation_warmup_duration - (
                        t_now - teleop_reactivation_time
                    )
                    data["target_time"] = t_now + remaining_warmup
                else:
                    # Normal teleoperation: match control frequency
                    data["target_time"] = t_now + (1 / config.teleop_frequency)

                # Publish the teleop command
                with telemetry.timer("publish_teleop_command"):
                    # Debug: Log when publishing toggle_data_collection
                    if data.get("toggle_data_collection", False):
                        print(
                            f"[TeleopPolicyLoop] Publishing data with toggle_data_collection=True to {CONTROL_GOAL_TOPIC}"
                        )
                        print(f"[TeleopPolicyLoop] Keys in data: {list(data.keys())}")
                    control_publisher.publish(data)

                # For the initial pose, wait the full duration before continuing
                if iteration == 0:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                iteration += 1
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.teleop_frequency):
                telemetry.log_timing_info(context="Teleop Policy Loop Missed", threshold=0.001)
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"\n[TeleopPolicy] ROSManager interrupted by user: {e}")
    except KeyboardInterrupt:
        print("\n[TeleopPolicy] Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[TeleopPolicy] Error in teleop loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("[TeleopPolicy] Cleaning up...")
        # Clean up teleop policy and device streamers
        try:
            if teleop_policy is not None:
                print("[TeleopPolicy] Stopping teleop policy...")
                # Stop device streamers if they exist
                if (
                    hasattr(teleop_policy, "body_streamer")
                    and teleop_policy.body_streamer is not None
                ):
                    if hasattr(teleop_policy.body_streamer, "stop_streaming"):
                        teleop_policy.body_streamer.stop_streaming()
                    # Stop Pico service if running
                    if hasattr(teleop_policy.body_streamer, "stop_pico_service"):
                        teleop_policy.body_streamer.stop_pico_service()

                if (
                    hasattr(teleop_policy, "hand_streamer")
                    and teleop_policy.hand_streamer is not None
                ):
                    if hasattr(teleop_policy.hand_streamer, "stop_streaming"):
                        teleop_policy.hand_streamer.stop_streaming()
                    # Stop Pico service if running
                    if hasattr(teleop_policy.hand_streamer, "stop_pico_service"):
                        teleop_policy.hand_streamer.stop_pico_service()
        except Exception as e:
            print(f"[TeleopPolicy] Error stopping teleop policy: {e}")

        try:
            if ros_manager is not None:
                print("[TeleopPolicy] Shutting down ROS...")
                ros_manager.shutdown()
        except Exception as e:
            print(f"[TeleopPolicy] Error shutting down ROS: {e}")

        # Clean up signal handler
        signal_handler.cleanup()
        print("[TeleopPolicy] Cleanup complete")

        # Give DDS threads time to finish before exit
        time.sleep(0.5)

        # Force exit to ensure all threads terminate
        # Use os._exit to avoid C++ destructor issues with DDS
        import os

        os._exit(0)


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
