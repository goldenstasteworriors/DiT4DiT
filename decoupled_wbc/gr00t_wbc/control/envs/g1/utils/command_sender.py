import time
from typing import Dict

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Vector3_
from unitree_sdk2py.utils.crc import CRC


class BodyCommandSender:
    def __init__(self, config: Dict):
        self.config = config
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
        ):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self._LowState_ = LowState_  # Store for subscriber initialization
        else:
            raise NotImplementedError(
                f"Robot type {self.config['ROBOT_TYPE']} is not supported yet"
            )
        # init kp kd
        self.kp_level = 1.0
        self.waist_kp_level = 1.0
        self.robot_kp = np.zeros(self.config["NUM_MOTORS"])
        self.robot_kd = np.zeros(self.config["NUM_MOTORS"])
        # set kp level
        for i in range(len(self.config["MOTOR_KP"])):
            self.robot_kp[i] = self.config["MOTOR_KP"][i] * self.kp_level
        for i in range(len(self.config["MOTOR_KD"])):
            self.robot_kd[i] = self.config["MOTOR_KD"][i] * 1.0
        self.weak_motor_joint_index = []
        for _, value in self.config["WeakMotorJointIndex"].items():
            self.weak_motor_joint_index.append(value)
        # init low cmd publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.low_state = None

        # Wait for first lowstate to set mode_machine (G1/H1-2 robots)
        if (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
        ):
            self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", self._LowState_)
            self._lowstate_subscriber.Init(self._LowStateHandler, 10)
            print("[BodyCommandSender] Waiting for first lowstate to set mode_machine...")
            timeout = 10.0
            start_time = time.time()
            while self.low_state is None:
                if time.time() - start_time > timeout:
                    print("[BodyCommandSender] WARNING: Timeout waiting for lowstate, using config value")
                    break
                time.sleep(0.1)
            if self.low_state is not None:
                print(f"[BodyCommandSender] mode_machine set to {self.low_state.mode_machine}")

        self.InitLowCmd()
        self.crc = CRC()

    def _LowStateHandler(self, msg):
        """Handler for lowstate messages to get mode_machine."""
        self.low_state = msg

    def InitLowCmd(self):
        # h1/go2:
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(self.config["NUM_MOTORS"]):
            # G1 and H1-2 robots use mode=1 (Enable) for all motors
            # H1 (original) uses 0x0A for non-weak motors
            if self.config["ROBOT_TYPE"] in ("g1_29dof", "h1-2_21dof", "h1-2_27dof"):
                self.low_cmd.motor_cmd[i].mode = 0x01  # 1:Enable, 0:Disable
            elif self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = self.config["UNITREE_LEGGED_CONST"]["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.config["UNITREE_LEGGED_CONST"]["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            if (
                self.config["ROBOT_TYPE"] == "g1_29dof"
                or self.config["ROBOT_TYPE"] == "h1-2_21dof"
                or self.config["ROBOT_TYPE"] == "h1-2_27dof"
            ):
                # Use mode_machine from lowstate if available, else fall back to config
                if self.low_state is not None:
                    self.low_cmd.mode_machine = self.low_state.mode_machine
                else:
                    self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
                self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]
            else:
                pass

    def is_weak_motor(self, motor_index: int) -> bool:
        return motor_index in self.weak_motor_joint_index

    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray, mode_machine: int | None = None):
        """Send motor commands to the robot.

        Args:
            cmd_q: Joint position commands
            cmd_dq: Joint velocity commands
            cmd_tau: Joint torque commands
            mode_machine: Mode machine value from robot state (for G1/H1-2 robots).
                         If provided, echoes this value in the command.
                         If None, uses the default from config.
        """
        # Use mode_machine from robot state if available (G1/H1-2 robots)
        if (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
        ):
            if mode_machine is not None:
                self.low_cmd.mode_machine = mode_machine
            elif self.low_state is not None:
                self.low_cmd.mode_machine = self.low_state.mode_machine
            else:
                self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
            self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]
        
        for i in range(self.config["NUM_MOTORS"]):
            motor_index = self.config["JOINT2MOTOR"][i]
            joint_index = self.config["MOTOR2JOINT"][i]
            if joint_index == -1:
                # send default joint position command
                self.low_cmd.motor_cmd[motor_index].q = self.config["DEFAULT_MOTOR_ANGLES"][
                    motor_index
                ]
                self.low_cmd.motor_cmd[motor_index].dq = 0.0
                self.low_cmd.motor_cmd[motor_index].tau = 0.0
            else:
                self.low_cmd.motor_cmd[motor_index].q = cmd_q[joint_index]
                self.low_cmd.motor_cmd[motor_index].dq = cmd_dq[joint_index]
                self.low_cmd.motor_cmd[motor_index].tau = cmd_tau[joint_index]
            # kp kd
            self.low_cmd.motor_cmd[motor_index].kp = self.robot_kp[motor_index]
            self.low_cmd.motor_cmd[motor_index].kd = self.robot_kd[motor_index]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def close(self):
        """Close DDS publisher."""
        try:
            if hasattr(self, 'lowcmd_publisher_') and self.lowcmd_publisher_:
                self.lowcmd_publisher_.Close()
        except Exception as e:
            print(f"[BodyCommandSender] Error closing publisher: {e}")


def make_hand_mode(motor_index: int) -> int:
    status = 0x01
    timeout = 0x01
    mode = motor_index & 0x0F
    mode |= status << 4  # bits [4..6]
    mode |= timeout << 7  # bit 7
    return mode


class HandCommandSender:
    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        if self.is_left:
            self.cmd_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        else:
            self.cmd_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)

        self.cmd_pub.Init()
        self.cmd = unitree_hg_msg_dds__HandCmd_()

        self.hand_dof = 7

        self.kp = [1.0] * self.hand_dof
        self.kd = [0.2] * self.hand_dof
        self.kp[0] = 2.0
        self.kd[0] = 0.5

    def send_command(self, cmd: np.ndarray):
        for i in range(self.hand_dof):
            # Build the bitfield mode (see your C++ example)
            mode_val = make_hand_mode(i)
            self.cmd.motor_cmd[i].mode = mode_val
            self.cmd.motor_cmd[i].q = cmd[i]
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
            self.cmd.motor_cmd[i].kp = self.kp[i]
            self.cmd.motor_cmd[i].kd = self.kd[i]

        self.cmd_pub.Write(self.cmd)

    def close(self):
        """Close DDS publisher."""
        try:
            if hasattr(self, 'cmd_pub') and self.cmd_pub:
                self.cmd_pub.Close()
        except Exception as e:
            print(f"[HandCommandSender] Error closing publisher: {e}")


class AlohaHandCommandSender:
    """DDS command sender for ALOHA-style grippers.
    
    Sends gripper commands to the aloha_gripper_dds_bridge running on the robot.
    The bridge forwards commands to the Arduino/OpenRB controller via serial.
    
    Uses Vector3_ message format:
    - x: left gripper position (0.0-0.065 meters)
    - y: right gripper position (0.0-0.065 meters)
    - z: unused (reserved for future use)
    """
    
    def __init__(self, dds_topic: str = "rt/aloha_hand/cmd"):
        """Initialize ALOHA hand command publisher.
        
        Args:
            dds_topic: DDS topic name for publishing gripper commands
                      Default: "rt/aloha_hand/cmd" (matches aloha_gripper_dds_bridge.py)
        """
        self.dds_topic = dds_topic
        
        # Initialize DDS publisher
        self.cmd_pub = ChannelPublisher(self.dds_topic, Vector3_)
        self.cmd_pub.Init()
        
        # Command message - initialize to open position
        self.cmd = Vector3_(x=0.0, y=0.0, z=0.0)
        
        print(f"AlohaHandCommandSender initialized on topic: {self.dds_topic}")
    
    def send_command(self, gripper_pos: float, is_left: bool):
        """Send single gripper command to Arduino.

        Arduino interprets values inversely:
        - 0.0 = gripper CLOSED
        - 0.065 = gripper OPEN

        Args:
            gripper_pos: Raw command value in meters (0.0-0.065)
            is_left: True for left gripper, False for right gripper
        """
        if is_left:
            self.cmd.x = float(gripper_pos)
        else:
            self.cmd.y = float(gripper_pos)

        # Write command to DDS
        self.cmd_pub.Write(self.cmd)
    
    def send_dual_command(self, left_pos: float, right_pos: float):
        """Send commands to both grippers simultaneously.

        Arduino interprets values inversely (0.0=closed, 0.065=open).

        Args:
            left_pos: Left gripper raw command value (0.0-0.065)
            right_pos: Right gripper raw command value (0.0-0.065)
        """
        self.cmd.x = float(left_pos)
        self.cmd.y = float(right_pos)

        # Write command to DDS
        self.cmd_pub.Write(self.cmd)

    def close(self):
        """Close DDS publisher."""
        try:
            if hasattr(self, 'cmd_pub') and self.cmd_pub:
                self.cmd_pub.Close()
        except Exception as e:
            print(f"[AlohaHandCommandSender] Error closing publisher: {e}")
