#!/usr/bin/env python3

"""
Real G1 Robot Deployment Script

This script manages the complete deployment workflow for teleoperation of a real G1 robot:
1. ALOHA Gripper Server (on Jetson Orin @ 192.168.123.164)
2. RealSense Camera Server (on Jetson Orin @ 192.168.123.164)
3. G1 Control Loop (on local host)
4. Camera Viewer (optional, on local host)
5. Teleop Policy (on local host)
6. Data Recorder (optional, on local host)

Usage:
    # Full deployment with all components
    python scripts/deploy_g1_real_robot.py

    # Minimal deployment (control + teleop only, no camera viewer or data collection)
    python scripts/deploy_g1_real_robot.py --no-view_camera --no-data_collection

    # Custom robot IP
    python scripts/deploy_g1_real_robot.py --robot_ip 192.168.123.100

    # Different interface (e.g., for testing)
    python scripts/deploy_g1_real_robot.py --interface enp4s0
"""

from pathlib import Path
import atexit
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import tyro

from gr00t_wbc.control.main.teleop.configs.configs import DeploymentConfig

import pdb

# Global reference for cleanup
_deployment_instance = None


class G1RealRobotDeployment:
    """
    Deployment manager for real G1 robot with distributed components.

    Architecture:
    - Jetson Orin (robot): ALOHA gripper server + RealSense camera server
    - Local host (operator): Control loop + Teleop policy + Camera viewer + Data collection
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

        # Process directories
        self.project_root = Path(__file__).resolve().parent.parent

        # Tmux session names
        self.session_name_local = "g1_deploy_local"  # For local host processes
        self.session_name_remote = "g1_deploy_remote"  # For Jetson Orin processes

        # SSH connection
        self.robot_ssh = f"unitree@{self.config.robot_ip}"
        self.robot_password = "123"  # Default Unitree password

        # Remote paths
        self.remote_project_root = "/home/unitree/decoupled_wbc"

        # Docker container name
        self.local_docker_container = "gr00t_wbc_deploy"
        self.remote_docker_container = "gr00t_wbc_jetson"

        # Setup X11 display for GUI applications
        self._setup_x11_display()

        # Create tmux sessions
        self._create_tmux_session(self.session_name_local)

    def _setup_x11_display(self):
        """Detect working X11 display and enable access for docker containers."""
        display = os.environ.get("DISPLAY", "")

        # Test if current DISPLAY works
        def test_display(disp: str) -> bool:
            try:
                env = os.environ.copy()
                env["DISPLAY"] = disp
                result = subprocess.run(["xdpyinfo"], capture_output=True, timeout=2, env=env)
                return result.returncode == 0
            except Exception:
                return False

        # If DISPLAY not set or doesn't work, scan for working display
        if not display or not test_display(display):
            for candidate in [":0", ":1", ":2"]:
                if test_display(candidate):
                    os.environ["DISPLAY"] = candidate
                    display = candidate
                    print(f"[X11] Found working display: {display}")
                    break
            else:
                print("[X11] WARNING: No working X11 display found. GUI may not work.")
                return
        else:
            print(f"[X11] Using display: {display}")

        # Enable X11 access for local connections (needed for docker)
        try:
            subprocess.run(
                ["xhost", "+local:root"],
                capture_output=True,
                timeout=5,
                env={"DISPLAY": display, **os.environ},
            )
            print("[X11] Enabled local X11 access for docker containers")
        except Exception as e:
            print(f"[X11] WARNING: Could not run xhost: {e}")

    def _create_tmux_session(self, session_name: str):
        """Create a new tmux session if it doesn't exist"""
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name], capture_output=True, text=True
        )

        if result.returncode != 0:
            subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
            print(f"[Tmux] Created session: {session_name}")

            if session_name == self.session_name_local:
                # Set up windows for local processes
                # Window 0: control_data_teleop (3 panes)
                subprocess.run(
                    ["tmux", "rename-window", "-t", f"{session_name}:0", "control_data_teleop"]
                )
                # Split horizontally
                subprocess.run(["tmux", "split-window", "-t", f"{session_name}:0", "-h"])
                # Split right pane vertically
                subprocess.run(["tmux", "split-window", "-t", f"{session_name}:0.1", "-v"])
                # Select left pane (control)
                subprocess.run(["tmux", "select-pane", "-t", f"{session_name}:0.0"])

    def _run_in_tmux_local(
        self, name: str, cmd: list, wait_time: float = 2, pane_index: Optional[int] = None
    ):
        """Run a command in tmux on the local local host"""
        if pane_index is not None:
            target = f"{self.session_name_local}:0.{pane_index}"
        else:
            subprocess.run(["tmux", "new-window", "-t", self.session_name_local, "-n", name])
            target = f"{self.session_name_local}:{name}"

        # Simple command - just activate venv, cd, and run
        # Assume we're already in the right environment since script is running
        cmd_str = " ".join(str(x) for x in cmd)

        # Pass X11 environment variables for GUI applications (camera viewer, etc.)
        display = os.environ.get("DISPLAY", ":1")
        xauthority = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
        full_cmd = f"export DISPLAY={display} XAUTHORITY={xauthority} && {cmd_str}"

        subprocess.run(["tmux", "send-keys", "-t", target, full_cmd, "C-m"])
        time.sleep(wait_time)

        # Check if process is still running
        result = subprocess.run(
            ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip() == "1":
            print(f"[ERROR] {name} failed to start!")
            return False

        return True

    def _run_ssh_command(self, cmd: str, check_docker: bool = True) -> bool:
        """Execute a command on the Jetson Orin via SSH in tmux

        Args:
            cmd: Command to run on remote
            check_docker: If True, runs command inside Docker container
        """
        if check_docker:
            # Run inside Docker container
            docker_cmd = (
                f"docker exec -it gr00t_wbc_jetson bash -c "
                f"'cd /home/unitree/decoupled_wbc && {cmd}'"
            )
            ssh_cmd = f'ssh -F /dev/null -t {self.robot_ssh} "{docker_cmd}"'
        else:
            # Run directly on host
            ssh_cmd = f'ssh -F /dev/null -t {self.robot_ssh} "{cmd}"'

        print(f"[SSH] Executing on {self.config.robot_ip}: {cmd}")

        try:
            result = subprocess.run(ssh_cmd, shell=True, timeout=10)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"[SSH] Command timed out: {cmd}")
            return False

    def check_local_docker(self) -> bool:
        """Check if local Docker container is running, start if needed"""
        print("\n[Local Docker] Checking Docker environment...")

        # Check if we're already inside a Docker container
        if Path("/.dockerenv").exists():
            print("[Local Docker] ✓ Already running inside Docker container")
            return True

        # Check if Docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)

        if result.returncode != 0:
            print("[ERROR] Docker is not installed or not running")
            print("Please install Docker or run this script from inside the Docker container")
            return False

        # Check if container exists and is running
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        )

        running_containers = result.stdout.strip().split("\n")

        # Look for any gr00t_wbc container
        gr00t_containers = [
            c for c in running_containers if "gr00t_wbc" in c or "gr00t" in c.lower()
        ]

        if gr00t_containers:
            print(f"[Local Docker] ✓ Docker container running: {gr00t_containers[0]}")
            print("[Local Docker] This script should be run INSIDE the Docker container")
            print(f"[Local Docker] Please run: docker exec -it {gr00t_containers[0]} bash")
            print(f"[Local Docker] Then run this script again from inside the container")
            return False

        # No container running - try to start one
        print("[Local Docker] No Docker container running")
        print("[Local Docker] Attempting to start Docker container...")

        docker_script = self.project_root / "docker" / "run_docker.sh"
        if not docker_script.exists():
            print(f"[ERROR] Docker script not found: {docker_script}")
            return False

        print(f"[Local Docker] Starting container with: {docker_script} --root")
        print(
            "[Local Docker] This will launch a new shell. Please run this script again from inside."
        )

        # Launch Docker in a new terminal or tmux window
        try:
            # Try to launch in new tmux window if we're in tmux
            if "TMUX" in os.environ:
                subprocess.run(
                    ["tmux", "new-window", "-n", "docker_setup", str(docker_script), "--root"]
                )
                print("[Local Docker] Docker started in new tmux window 'docker_setup'")
            else:
                # Otherwise just print instructions
                print("[Local Docker] Please run in a new terminal:")
                print(f"  cd {self.project_root}")
                print(f"  ./docker/run_docker.sh --root")
                print(f"  python scripts/deploy_g1_real_robot.py")
        except Exception as e:
            print(f"[ERROR] Failed to start Docker: {e}")
            print("[Local Docker] Please manually run:")
            print(f"  cd {self.project_root}")
            print(f"  ./docker/run_docker.sh --root")

        return False

    def _run_ssh_check_command(
        self, cmd: str, timeout: int = 10, debug: bool = False
    ) -> subprocess.CompletedProcess:
        """Run SSH command for checking, avoiding .bashrc interactive prompts.

        The Jetson has a ROS version prompt in .bashrc that breaks non-interactive SSH.
        We use 'bash --norc --noprofile' to avoid sourcing .bashrc.

        Returns a CompletedProcess. If SSH auth fails and sshpass is not available,
        stderr will contain 'SSH_AUTH_FAILED' marker.
        """
        # Use -F /dev/null to ignore SSH config (avoids permission issues in Docker)
        # Use bash --norc --noprofile to avoid .bashrc ROS prompts
        # Try with sshpass if available for password auth
        ssh_base = (
            f"ssh -F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"
        )
        ssh_cmd = f"{ssh_base} {self.robot_ssh} 'bash --norc --noprofile -c \"{cmd}\"'"

        if debug:
            print(f"[DEBUG] Running SSH command: {ssh_cmd}")

        result = subprocess.run(
            ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )

        if debug:
            print(
                f"[DEBUG] SSH result: returncode={result.returncode}, stdout={repr(result.stdout)}, stderr={repr(result.stderr)}"
            )

        # Check if it failed due to password auth required
        if result.returncode != 0 and "Permission denied" in result.stderr:
            if debug:
                print("[DEBUG] SSH key auth failed, trying with sshpass...")
            # Try with sshpass if available (password: 123)
            sshpass_cmd = f"sshpass -p '123' ssh -F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 {self.robot_ssh} 'bash --norc --noprofile -c \"{cmd}\"'"
            result = subprocess.run(
                sshpass_cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            if debug:
                print(
                    f"[DEBUG] sshpass result: returncode={result.returncode}, stdout={repr(result.stdout)}, stderr={repr(result.stderr)}"
                )

            # If sshpass not found, mark it as auth failure for caller to handle
            if "sshpass: not found" in result.stderr or result.returncode == 127:
                result = subprocess.CompletedProcess(
                    args=ssh_cmd,
                    returncode=255,
                    stdout="",
                    stderr="SSH_AUTH_FAILED: Password authentication required but sshpass not available",
                )

        return result

    def check_jetson_docker(self) -> bool:
        """Check if Jetson Docker container is running, start if needed"""
        print("\n[Jetson Docker] Checking Docker environment...")

        # Check if container is running from the host
        # Use direct path to docker to avoid shell issues with .bashrc ROS prompts
        check_cmd = "/usr/bin/docker ps --format {{.Names}} 2>/dev/null | grep -E gr00t_wbc || true"
        try:
            print(f"[Jetson Docker] Running check command...")
            result = self._run_ssh_check_command(check_cmd, debug=True)

            if result.stdout.strip():
                container_name = result.stdout.strip().split("\n")[0]
                self.remote_docker_container = container_name
                print(f"[Jetson Docker] ✓ Docker container running: {container_name}")
                return True
            elif "SSH_AUTH_FAILED" in result.stderr or (
                "Permission denied" in result.stderr and result.returncode == 255
            ):
                # SSH auth failed - provide helpful message
                print(
                    f"[Jetson Docker] SSH authentication requires password (no SSH keys configured)"
                )
                print(f"[Jetson Docker] To enable automatic checks, either:")
                print(f"  1. Install sshpass: apt-get install -y sshpass")
                print(f"  2. Or set up SSH keys: ssh-copy-id {self.robot_ssh}")
                # Ask user to confirm
                response = input("\nIs Jetson Docker already running? (y/n): ")
                if response.lower() == "y":
                    print(f"[Jetson Docker] ✓ User confirmed Docker is running")
                    return True
            else:
                print(f"[Jetson Docker] No container found, will auto-start...")
        except subprocess.TimeoutExpired:
            print("[Jetson Docker] SSH command timed out")
        except Exception as e:
            print(f"[Jetson Docker] Check failed: {e}")

        # Container not running - auto-start it
        return self._auto_start_jetson_docker()

    def _auto_start_jetson_docker(self) -> bool:
        """Automatically start Jetson Docker container in detached mode"""
        print("\n[Jetson Docker] Auto-starting Docker container...")

        container_name = "gr00t_wbc_jetson"
        image_name = "gr00t_wbc-deploy-jetson:latest"
        home_dir = "/home/unitree"
        project_path = f"{home_dir}/decoupled_wbc"

        # Stop and remove any existing container
        cleanup_cmd = f"docker stop {container_name} 2>/dev/null; docker rm {container_name} 2>/dev/null; sleep 1"
        print(f"[Jetson Docker] Cleaning up old containers...")
        self._run_ssh_with_password(cleanup_cmd, timeout=10)

        # Start container in detached mode with sleep infinity to keep it alive
        docker_run_cmd = (
            f"docker run -d "
            f"--name {container_name} "
            f"--network host "
            f"--privileged "
            f"--runtime nvidia "
            f"--gpus all "
            f"-v /dev:/dev "
            f"-v /tmp/.X11-unix:/tmp/.X11-unix:rw "
            f"-v {project_path}:{project_path} "
            f"-v /etc/passwd:/etc/passwd:ro "
            f"-v /etc/group:/etc/group:ro "
            f"-e DISPLAY=$DISPLAY "
            f"-e NVIDIA_VISIBLE_DEVICES=all "
            f"-e NVIDIA_DRIVER_CAPABILITIES=all "
            f"-e HOME={home_dir} "
            f"-w {project_path} "
            f"{image_name} "
            f"sleep infinity"
        )

        print(f"[Jetson Docker] Starting container '{container_name}'...")
        try:
            result = self._run_ssh_with_password(docker_run_cmd, timeout=20)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start container")
                print(f"[DEBUG] stderr: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("[WARNING] Docker start command timed out, checking if container started...")

        # Verify container started (give it a few seconds to initialize)
        time.sleep(3)
        check_cmd = f"docker ps -q -f name={container_name} && echo OK || echo FAIL"
        result = self._run_ssh_with_password(check_cmd, timeout=5)

        if "OK" not in result.stdout:
            print(f"[ERROR] Container not running after start")
            # Show container logs for debugging
            log_cmd = f"docker logs {container_name} 2>&1 | tail -30"
            log_result = self._run_ssh_with_password(log_cmd, timeout=5)
            print(f"[DEBUG] Container logs:\n{log_result.stdout}")
            return False

        self.remote_docker_container = container_name
        print(f"[Jetson Docker] ✓ Container started: {container_name}")
        print(
            f"[Jetson Docker]   To access: ssh {self.robot_ssh} 'docker exec -it {container_name} bash'"
        )
        return True

    def setup_ssh_keys(self) -> bool:
        """Set up SSH keys for passwordless authentication to robot"""
        print("\n[SSH] Setting up SSH keys for passwordless authentication...")

        ssh_dir = Path.home() / ".ssh"
        ssh_dir.mkdir(mode=0o700, exist_ok=True)

        # Check for existing SSH keys
        key_files = [
            ssh_dir / "id_ed25519",
            ssh_dir / "id_rsa",
            ssh_dir / "id_ecdsa",
        ]

        existing_key = None
        for key_file in key_files:
            if key_file.exists():
                existing_key = key_file
                print(f"[SSH] Found existing SSH key: {key_file}")
                break

        # Generate key if none exists
        if existing_key is None:
            print("[SSH] No SSH key found, generating new ed25519 key...")
            key_file = ssh_dir / "id_ed25519"

            result = subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(key_file), "-N", ""],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"[ERROR] Failed to generate SSH key: {result.stderr}")
                return False

            existing_key = key_file
            print(f"[SSH] ✓ Generated new SSH key: {key_file}")

        # Test if key already works
        print(f"[SSH] Testing SSH connection to {self.robot_ssh}...")
        test_cmd = f"ssh -F /dev/null -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 {self.robot_ssh} 'echo OK'"
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and "OK" in result.stdout:
            print("[SSH] ✓ SSH key authentication already working!")
            return True

        # Copy key to robot
        print(f"[SSH] Copying SSH key to {self.robot_ssh}...")
        print(f"[SSH] You will be prompted for the robot password (default: 123)")

        # Use ssh-copy-id to copy the key
        copy_cmd = [
            "ssh-copy-id",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            str(existing_key.with_suffix(existing_key.suffix + ".pub")),
            self.robot_ssh,
        ]

        try:
            result = subprocess.run(copy_cmd, timeout=60)

            if result.returncode != 0:
                print(f"[ERROR] Failed to copy SSH key")
                return False

        except subprocess.TimeoutExpired:
            print("[ERROR] SSH key copy timed out")
            return False
        except KeyboardInterrupt:
            print("\n[SSH] SSH key setup cancelled by user")
            return False

        # Verify it works
        print(f"[SSH] Verifying SSH key authentication...")
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and "OK" in result.stdout:
            print("[SSH] ✓ SSH key authentication working!")
            return True
        else:
            print(f"[ERROR] SSH key authentication still not working")
            print(f"[DEBUG] Test result: {result.stderr}")
            return False

    def check_ssh_config(self) -> bool:
        """Check SSH configuration and permissions"""
        print("\n[SSH] Checking SSH configuration...")

        ssh_config = Path.home() / ".ssh" / "config"
        if ssh_config.exists():
            # Check permissions
            stat_info = ssh_config.stat()
            perms = oct(stat_info.st_mode)[-3:]

            if perms != "600":
                print(f"[SSH] Warning: SSH config has incorrect permissions: {perms}")
                print(f"[SSH] Fixing permissions to 600...")
                try:
                    ssh_config.chmod(0o600)
                    print(f"[SSH] ✓ Fixed SSH config permissions")
                except Exception as e:
                    print(f"[ERROR] Failed to fix permissions: {e}")
                    print(f"[SSH] Please run: chmod 600 ~/.ssh/config")
                    return False

        print("[SSH] ✓ SSH configuration OK")
        return True

    def check_aloha_gripper_service(self) -> bool:
        """Check if ALOHA gripper DDS bridge is running on Jetson"""
        print("\n[ALOHA] Checking ALOHA gripper service on Jetson...")

        check_cmd = "pgrep -f 'aloha_gripper_dds_bridge.py'"
        result = self._run_ssh_with_password(check_cmd)

        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            print(f"[ALOHA] ✓ ALOHA gripper DDS bridge is running (PID: {', '.join(pids)})")
            return True
        else:
            print("[ALOHA] ⚠ ALOHA gripper DDS bridge is NOT running")
            print("[ALOHA] The gripper bridge should be started before deployment.")
            print("[ALOHA] To start it manually on the Jetson:")
            print(f"  ssh {self.robot_ssh}")
            print("  cd /home/unitree/decoupled_wbc")
            print("  ./aloha_feedback_server/start_aloha_bridge.sh")

            response = input("\nWould you like to start the ALOHA gripper bridge now? (y/n): ")
            if response.lower() == "y":
                return self.start_jetson_gripper_server()
            else:
                print("[ALOHA] Continuing without ALOHA gripper bridge...")
                return True  # Allow continuing without grippers

    def check_robot_connection(self) -> bool:
        """Verify SSH connection to robot"""
        print(f"\n[Setup] Checking connection to robot at {self.config.robot_ip}...")
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", self.config.robot_ip], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"[ERROR] Cannot ping robot at {self.config.robot_ip}")
            print("Please verify:")
            print("  1. Robot is powered on")
            print("  2. Network connection is established")
            print(f"  3. IP address {self.config.robot_ip} is correct")
            return False

        print(f"[Setup] ✓ Robot is reachable at {self.config.robot_ip}")
        return True

    def setup_jetson_camera(self) -> bool:
        """Disable videohub_pc4 service that blocks RealSense camera"""
        print("\n[Jetson] Checking videohub_pc4 service...")

        # Check if service is running
        check_cmd = "pgrep -f videohub_pc4"
        result = self._run_ssh_with_password(check_cmd)

        if result.stdout.strip():
            print("[Jetson] videohub_pc4 is running, needs to be disabled")
            print("[Jetson] You may need to manually run these commands on the Jetson:")
            print(f"  ssh {self.robot_ssh}")
            print("  Select ROS version: 1 (foxy)")
            print(
                "  sudo mv /unitree/module/video_hub_pc4/videohub_pc4 /unitree/module/video_hub_pc4/videohub_pc4.disabled"
            )
            print(f"  sudo kill -9 {result.stdout.strip()}")

            response = input("\nHave you disabled videohub_pc4? (y/n): ")
            if response.lower() != "y":
                print("[Setup] Camera setup cancelled")
                return False

        print("[Jetson] ✓ videohub_pc4 is not blocking camera")
        return True

    def start_jetson_gripper_server(self):
        """Start ALOHA gripper DDS bridge on Jetson Orin (runs on HOST, not in Docker)"""
        print("\n[Jetson] Starting ALOHA gripper server...")

        # Check if already running - if so, don't restart to avoid serial reconnection issues
        check_cmd = "pgrep -f 'aloha_gripper_dds_bridge.py'"
        result = self._run_ssh_with_password(check_cmd, timeout=5)
        if result.stdout.strip():
            print(f"[Jetson] ✓ ALOHA gripper server already running (PID: {result.stdout.strip()})")
            return True

        # Only start if not running - don't kill existing processes
        print("[Jetson] Starting new ALOHA gripper server...")

        # Create a startup script on the Jetson using base64 to avoid quoting issues
        log_file = "/tmp/aloha_gripper.log"
        script_path = "/tmp/start_gripper.sh"

        # Script content
        import base64

        script_content = """#!/bin/bash
cd /home/unitree
exec python3 aloha_feedback_server/aloha_gripper_dds_bridge.py
"""
        script_b64 = base64.b64encode(script_content.encode()).decode()

        # Write script using base64 decode
        create_script_cmd = (
            f"echo {script_b64} | base64 -d > {script_path} && chmod +x {script_path}"
        )
        result = self._run_ssh_with_password(create_script_cmd, timeout=5)
        if result.returncode != 0:
            print(f"[ERROR] Failed to create startup script: {result.stderr}")
            return False

        # Run the script with nohup
        nohup_cmd = f"nohup {script_path} < /dev/null > {log_file} 2>&1 & echo $!"

        try:
            result = self._run_ssh_with_password(nohup_cmd, timeout=5)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start gripper server: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            pass

        # Verify process started
        time.sleep(3)
        check_cmd = "pgrep -f 'aloha_gripper_dds_bridge.py' && echo OK || echo FAIL"
        result = self._run_ssh_with_password(check_cmd, timeout=5)

        if "OK" not in result.stdout:
            print(f"[ERROR] Gripper server process not found after starting")
            # Show log output for debugging
            log_result = self._run_ssh_with_password(f"cat {log_file}", timeout=5)
            print(f"[DEBUG] Log output:\n{log_result.stdout}")
            return False

        pid = result.stdout.strip().split("\n")[0]
        print(f"[Jetson] ✓ ALOHA gripper server started (PID: {pid})")
        print(f"[Jetson]   Log file: {log_file}")
        print(f"[Jetson]   To monitor: ssh unitree@192.168.123.164 'tail -f {log_file}'")

        # Show initial log to check for errors/warnings
        time.sleep(2)
        log_result = self._run_ssh_with_password(f"cat {log_file}", timeout=5)
        if "ERROR" in log_result.stdout or "Error" in log_result.stdout:
            print(f"[WARNING] Gripper server has errors:\n{log_result.stdout}")
        elif log_result.stdout.strip():
            print(f"[INFO] Gripper server output:\n{log_result.stdout}")

        return True

    def _run_ssh_with_password(self, cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
        """Run SSH command with sshpass for password authentication.

        Uses -F /dev/null to avoid SSH config permission issues in Docker.
        """
        ssh_cmd = (
            f"sshpass -p '{self.robot_password}' "
            f"ssh -F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
            f"{self.robot_ssh} '{cmd}'"
        )
        result = subprocess.run(
            ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )

        # If sshpass not found, try without (for key-based auth)
        if "sshpass: not found" in result.stderr:
            ssh_cmd = (
                f"ssh -F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
                f"{self.robot_ssh} '{cmd}'"
            )
            result = subprocess.run(
                ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )

        return result

    def start_jetson_camera_server(self):
        """Start RealSense camera server on Jetson Orin using docker run -d"""
        print("\n[Jetson] Starting RealSense camera server...")

        camera_container = "gr00t_camera_server"

        # Check if camera container already running
        check_cmd = f"docker ps -q -f name={camera_container}"
        result = self._run_ssh_with_password(check_cmd, timeout=5)
        if result.stdout.strip():
            print(f"[Jetson] ✓ Camera container already running")
            return True

        # Stop and remove any existing camera container
        cleanup_cmd = f"docker stop {camera_container} 2>/dev/null; docker rm {camera_container} 2>/dev/null; sleep 1"
        self._run_ssh_with_password(cleanup_cmd, timeout=10)

        # Start new container with camera server using docker run -d
        # This mirrors the run_docker_jetson.sh script but runs detached with the camera command
        image_name = "gr00t_wbc-deploy-jetson:latest"
        home_dir = "/home/unitree"
        project_path = f"{home_dir}/decoupled_wbc"

        docker_run_cmd = (
            f"docker run -d "
            f"--name {camera_container} "
            f"--network host "
            f"--privileged "
            f"--runtime nvidia "
            f"--gpus all "
            f"-v /dev:/dev "
            f"-v /tmp/.X11-unix:/tmp/.X11-unix:rw "
            f"-v {project_path}:{project_path} "
            f"-v /etc/passwd:/etc/passwd:ro "
            f"-v /etc/group:/etc/group:ro "
            f"-e DISPLAY=$DISPLAY "
            f"-e NVIDIA_VISIBLE_DEVICES=all "
            f"-e NVIDIA_DRIVER_CAPABILITIES=all "
            f"-e HOME={home_dir} "
            f"-w {project_path} "
            f"{image_name} "
            f'/bin/bash -c "source {home_dir}/venv/bin/activate && '
            f"python gr00t_wbc/control/sensor/composed_camera.py "
            f'--ego_view_camera realsense --port 5555 --server"'
        )

        try:
            result = self._run_ssh_with_password(docker_run_cmd, timeout=15)
            if result.returncode != 0:
                print(f"[DEBUG] docker run returned: {result.returncode}")
                print(f"[DEBUG] stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            pass

        # Verify container started (give it a few seconds to initialize)
        time.sleep(3)
        check_cmd = f"docker ps -q -f name={camera_container} && echo OK || echo FAIL"
        result = self._run_ssh_with_password(check_cmd, timeout=5)

        if "OK" not in result.stdout:
            print(f"[ERROR] Camera container not running after start")
            # Show container logs for debugging
            log_cmd = f"docker logs {camera_container} 2>&1 | tail -30"
            log_result = self._run_ssh_with_password(log_cmd, timeout=5)
            print(f"[DEBUG] Container logs:\n{log_result.stdout}")

            # Show manual instructions for fallback
            print(f"\n[Jetson] Automatic start failed. Please manually start camera server:")
            print(f"\n  # Terminal: SSH to Jetson")
            print(f"  ssh {self.robot_ssh}")
            print(f"  cd ~/Projects/decoupled_wbc")
            print(f"  ./docker/run_docker_jetson.sh --root")
            print(f"")
            print(f"  # Inside Docker container, run:")
            print(f"  python gr00t_wbc/control/sensor/composed_camera.py \\")
            print(f"      --ego_view_camera realsense \\")
            print(f"      --port 5555 \\")
            print(f"      --server")
            return False

        print(f"[Jetson] ✓ Camera server started in container: {camera_container}")
        print(f"[Jetson]   To monitor: ssh {self.robot_ssh} 'docker logs -f {camera_container}'")
        return True

    def start_control_loop(self):
        """Start G1 control loop on local host"""
        print("\n[local] Starting G1 control loop...")

        # Simple command matching what works manually
        cmd = [
            "python",
            "gr00t_wbc/control/main/teleop/run_g1_control_loop.py",
            "--interface",
            "real",
            "--hand-type",
            "aloha",
            "--with-hands",
        ]
        if not self._run_in_tmux_local("control", cmd, wait_time=1, pane_index=0):
            print("[ERROR] Control loop failed to start!")
            return False

        # Wait for control loop to be fully ready
        if not self._wait_for_control_loop_ready():
            print("[ERROR] Control loop did not become ready in time!")
            return False

        print("[local] ✓ Control loop ready")
        print("[local]   Controls: 'i' for initial pose, ']' to activate locomotion")
        return True

    def _wait_for_control_loop_ready(self, timeout: int = 30) -> bool:
        """Wait for control loop to receive robot low state and ROS to be ready"""
        print("[local] Waiting for control loop to be ready...")
        print("[local]   Looking for: 'Robot low state received' and 'ROS_Manager is running'")
        print(f"[local]   Timeout: {timeout}s")

        target = f"{self.session_name_local}:0.0"
        start_time = time.time()

        low_state_received = False
        ros_ready = False
        last_progress = 0

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # Print progress every 10 seconds
            if elapsed >= last_progress + 10:
                last_progress = elapsed
                status = []
                if low_state_received:
                    status.append("low_state=✓")
                else:
                    status.append("low_state=waiting")
                if ros_ready:
                    status.append("ros=✓")
                else:
                    status.append("ros=waiting")
                print(f"[local]   {elapsed}s elapsed... ({', '.join(status)})")

            # Capture pane contents
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", target, "-p", "-S", "-500"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                output = result.stdout

                # Check for error messages
                error_patterns = [
                    "Error:",
                    "ERROR",
                    "Exception",
                    "Traceback",
                    "Connection refused",
                    "No such device",
                    "Permission denied",
                ]
                for pattern in error_patterns:
                    if pattern in output and "error" not in output.lower()[:50]:
                        # Only print first occurrence of error
                        lines = output.split("\n")
                        for i, line in enumerate(lines):
                            if pattern in line:
                                print(f"[local] ⚠ Detected issue: {line.strip()[:100]}")
                                break
                        break

                # Check for key messages
                if "Robot low state received" in output:
                    if not low_state_received:
                        print("[local] ✓ Robot low state received")
                        low_state_received = True

                if "ROS_Manager is running" in output:
                    if not ros_ready:
                        print("[local] ✓ ROS ready")
                        ros_ready = True

                # Both conditions met
                if low_state_received and ros_ready:
                    return True

                # Check if process died
                pane_result = subprocess.run(
                    ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
                    capture_output=True,
                    text=True,
                )
                if pane_result.stdout.strip() == "1":
                    print("[ERROR] Control loop process died!")
                    # Print last few lines of output
                    lines = output.strip().split("\n")
                    print("[ERROR] Last output lines:")
                    for line in lines[-10:]:
                        print(f"  {line}")
                    return False

            time.sleep(0.5)

        print(f"[ERROR] Timeout after {timeout}s waiting for control loop")
        print("[ERROR] Check tmux output: tmux attach -t g1_deploy_local")

        # Print what we captured to help debug
        try:
            final_result = subprocess.run(
                ["tmux", "capture-pane", "-t", target, "-p", "-S", "-50"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if final_result.returncode == 0:
                lines = final_result.stdout.strip().split("\n")
                print("[DEBUG] Last 20 lines from tmux:")
                for line in lines[-20:]:
                    print(f"  {line}")
        except Exception as e:
            print(f"[DEBUG] Could not capture tmux output: {e}")

        return False

    def start_odin_camera(self):
        """Check that Odin camera (grab_rgb) is running outside the container."""
        print("\n[local] Checking Odin camera (grab_rgb with ZMQ)...")

        odin_port = 5556

        try:
            import zmq
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            sock.connect(f"tcp://localhost:{odin_port}")
            try:
                sock.recv()
                print(f"[local] ✓ Odin camera detected on port {odin_port}")
                sock.close()
                ctx.term()
                return True
            except zmq.Again:
                sock.close()
                ctx.term()
        except Exception as e:
            print(f"[local] ZMQ check error: {e}")

        print(f"[ERROR] Odin camera not detected on port {odin_port}")
        print(f"\n  Please start grab_rgb in a SEPARATE TERMINAL (outside Docker):")
        print(f"")
        print(f"  cd <path-to-odin_ros_driver>")
        print(f"  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
        print(f"  ./examples/build/grab_rgb --undistort --res 640 480 --zmq --zmq_port {odin_port} --save_dir odin_videos")
        print(f"")

        response = input("Have you started grab_rgb? (y/n): ")
        if response.lower() != "y":
            return False

        try:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            sock.connect(f"tcp://localhost:{odin_port}")
            try:
                sock.recv()
                print(f"[local] ✓ Odin camera detected on port {odin_port}")
                sock.close()
                ctx.term()
                return True
            except zmq.Again:
                sock.close()
                ctx.term()
                print(f"[ERROR] Still no data on port {odin_port}. Is grab_rgb running with --zmq?")
                return False
        except Exception as e:
            print(f"[ERROR] ZMQ check failed: {e}")
            return False

    def start_camera_viewer(self):
        """Start camera viewer on local host"""
        if not self.config.view_camera:
            print("\n[local] Camera viewer disabled (--no-view_camera)")
            return True

        if self.config.ego_camera == "odin":
            print("\n[local] Camera viewer skipped (Odin grab_rgb has its own display window)")
            return True

        print("\n[local] Starting camera viewer...")

        camera_host = "192.168.123.164"
        camera_port = "5555"

        # Match manual command exactly
        cmd = [
            "python",
            "gr00t_wbc/control/main/teleop/run_camera_viewer.py",
            "--camera-host",
            camera_host,
            "--camera-port",
            camera_port,
        ]

        if not self._run_in_tmux_local("camera_viewer", cmd, wait_time=2):
            print("[WARNING] Camera viewer failed to start, continuing...")
            return False

        print("[local] ✓ Camera viewer started successfully")
        return True

    def start_teleop_policy(self):
        """Start teleoperation policy on local host"""
        print("\n[local] Starting teleoperation policy...")

        # Match manual command exactly
        cmd = [
            "python",
            "gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py",
            "--hand_control_device",
            "pico",
            "--body_control_device",
            "pico",
            "--hand-type",
            "aloha",
            "--with-hands",
        ]
        if not self._run_in_tmux_local("teleop", cmd, wait_time=2, pane_index=2):
            print("[WARNING] Teleop policy failed to start, continuing...")
            return False

        print("[local] ✓ Teleoperation policy started successfully")
        print("[local]   Press 'l' in control loop to start teleoperation")
        return True

    def start_data_collection(self):
        """Start data collection on local host"""
        if not self.config.data_collection:
            print("\n[local] Data collection disabled (--no-data_collection)")
            return True

        print("\n[local] Starting data collection...")

        # Camera host/port depends on ego_camera type
        if self.config.ego_camera == "odin":
            camera_host = "localhost"
            camera_port = "5556"
        else:
            camera_host = "192.168.123.164"
            camera_port = "5555"

        # Match manual command exactly
        cmd = [
            "python",
            "gr00t_wbc/control/main/teleop/run_g1_data_exporter.py",
            "--camera_host",
            camera_host,
            "--camera_port",
            camera_port,
            "--data_collection_frequency",
            "30",
            "--root_output_dir",
            "./g1_real_data",
            "--no-add_stereo_camera",
        ]
        if not self._run_in_tmux_local("data", cmd, wait_time=2, pane_index=1):
            print("[WARNING] Data collection failed to start, continuing...")
            return False

        print("[local] ✓ Data collection started successfully")
        print("[local]   Press 'c' in control loop or Pico button 'A' to start/stop recording")
        print("[local]   Press 'x' in control loop or Pico button 'B' to abort episode")
        return True

    def deploy(self):
        """Execute full deployment sequence"""
        print("=" * 70)
        print("G1 REAL ROBOT DEPLOYMENT")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Robot IP:              {self.config.robot_ip}")
        print(f"  Interface:             {self.config.interface}")
        print(f"  WBC Version:           {self.config.wbc_version}")
        print(f"  Hand Type:             aloha")
        print(f"  With Hands:            {self.config.with_hands}")
        print(f"  Ego Camera:            {self.config.ego_camera}")
        print(f"  View Camera:           {self.config.view_camera}")
        print(f"  Data Collection:       {self.config.data_collection}")
        print(f"  Body Control Device:   {self.config.body_control_device}")
        print(f"  Hand Control Device:   {self.config.hand_control_device}")
        print(f"  High Elbow Pose:       {self.config.high_elbow_pose}")
        print(f"  Gravity Compensation:  {self.config.enable_gravity_compensation}")
        print("=" * 70)

        # Register signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        # Pre-deployment checks
        print("\n" + "=" * 70)
        print("PRE-DEPLOYMENT CHECKS")
        print("=" * 70)

        # Check if ALOHA gripper service is already running (persists across deployments)
        if not self.check_aloha_gripper_service():
            print("\n[WARNING] ALOHA gripper service not running")
            print("[INFO] It will be started during deployment")

        # Check SSH configuration
        if not self.check_ssh_config():
            sys.exit(1)

        # Set up SSH keys for passwordless authentication
        if not self.setup_ssh_keys():
            print("\n[ERROR] SSH key setup failed")
            print("[ERROR] You can manually set up SSH keys with:")
            print(f"  ssh-copy-id {self.robot_ssh}")
            print("\nOr continue anyway and enter password when prompted")
            response = input("\nContinue without SSH keys? (y/n): ")
            if response.lower() != "y":
                sys.exit(1)

        # Check local Docker environment
        if not self.check_local_docker():
            sys.exit(1)

        # Check robot connectivity
        if not self.check_robot_connection():
            sys.exit(1)

        # Check Jetson Docker environment
        if not self.check_jetson_docker():
            sys.exit(1)

        # Check camera setup (only needed for RealSense on Jetson)
        if self.config.ego_camera == "realsense":
            if not self.setup_jetson_camera():
                sys.exit(1)

        print("\n" + "=" * 70)
        print("STARTING DEPLOYMENT SEQUENCE")
        print("=" * 70)

        # Start Jetson components
        if not self.start_jetson_gripper_server():
            print("[ERROR] Failed to start gripper server")
            sys.exit(1)

        print("[Jetson] Waiting for gripper server to initialize...")
        time.sleep(3)  # Increased wait time for gripper DDS topics

        # Start camera server based on ego_camera type
        if self.config.ego_camera == "realsense":
            if not self.start_jetson_camera_server():
                print("[ERROR] Failed to start camera server")
                sys.exit(1)
            print("[Jetson] Waiting for camera server to initialize...")
            time.sleep(3)  # Wait for camera server to initialize
        elif self.config.ego_camera == "odin":
            if not self.start_odin_camera():
                print("[ERROR] Failed to start Odin camera")
                sys.exit(1)
            print("[local] Waiting for Odin camera to initialize...")
            time.sleep(3)

        # Additional wait to ensure robot DDS topics are available
        print("[Setup] Waiting for robot DDS topics to be ready...")
        time.sleep(5)  # Wait for robot to publish DDS topics

        print("[Setup] DDS topics should now be available")

        # Start local host components
        # Control loop will wait until fully ready (low state + ROS)
        if not self.start_control_loop():
            print("[ERROR] Failed to start control loop")
            print("[DEBUG] Check tmux session for errors: tmux attach -t g1_deploy_local")
            self.cleanup()
            sys.exit(1)

        # Control loop is now ready, safe to start other components
        self.start_camera_viewer()  # Optional, continue on failure

        time.sleep(1)

        if not self.start_teleop_policy():
            print("[WARNING] Teleop policy failed, but continuing...")

        time.sleep(1)

        self.start_data_collection()  # Optional, continue on failure

        print("\n" + "=" * 70)
        print("DEPLOYMENT COMPLETE")
        print("=" * 70)
        print(f"\nLocal processes in tmux session: {self.session_name_local}")
        print(f"Remote processes on {self.config.robot_ip}:")
        print("  - aloha_gripper (host, log: /tmp/aloha_gripper.log)")
        print("  - camera_server (Docker container: gr00t_camera_server)")
        print("\nUseful commands:")
        print(f"  Attach to local:  tmux attach -t {self.session_name_local}")
        print(f"  Monitor gripper:  ssh {self.robot_ssh} 'tail -f /tmp/aloha_gripper.log'")
        print(f"  Monitor camera:   ssh {self.robot_ssh} 'docker logs -f gr00t_camera_server'")
        print("  Detach from tmux: Ctrl+b then d")
        print("  Kill all:         Ctrl+\\ (in any local window)")
        print("\nControl sequence:")
        print("  1. Press 'i' in control loop to set initial pose")
        print("  2. Press ']' in control loop to activate locomotion")
        print("  3. Press 'l' in control loop to start teleoperation")
        print("  4. Press 'c' or Pico button 'A' to start/stop data recording")
        print("=" * 70)

        try:
            # Attach to tmux session
            subprocess.run(
                [
                    "tmux",
                    "attach",
                    "-t",
                    self.session_name_local,
                    ";",
                    "select-window",
                    "-t",
                    "control_data_teleop",
                ]
            )
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            self.cleanup()
            sys.exit(0)

        # Keep alive and monitor
        try:
            while True:
                result = subprocess.run(
                    ["tmux", "has-session", "-t", self.session_name_local],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print("Tmux session terminated. Exiting.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.cleanup()

    def cleanup_jetson(self):
        """Kill all deployment-related processes on Jetson"""
        print(f"\n[Cleanup] Terminating all Jetson processes on {self.config.robot_ip}...")

        # NOTE: ALOHA gripper server is NOT killed here to avoid serial reconnection issues
        # It will persist across multiple deployments. Kill manually if needed:
        # ssh unitree@192.168.123.164 "pkill -f aloha_gripper_dds_bridge.py"

        # Stop and remove camera server container
        try:
            cmd = "docker stop gr00t_camera_server 2>/dev/null; docker rm gr00t_camera_server 2>/dev/null; echo done"
            result = self._run_ssh_with_password(cmd, timeout=10)
            print(f"[Cleanup] ✓ Stopped camera server container")
        except Exception as e:
            print(f"[Cleanup] Warning: Could not kill camera server: {e}")

    def cleanup(self):
        """Clean up all processes (local and remote)"""
        print("\n[Cleanup] Terminating all processes...")

        # Kill local session
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name_local],
                timeout=5,
                capture_output=True,
            )
            print(f"[Cleanup] ✓ Terminated local session: {self.session_name_local}")
        except subprocess.TimeoutExpired:
            subprocess.run(["tmux", "kill-session", "-t", self.session_name_local, "-9"])
        except Exception as e:
            print(f"[Cleanup] Warning: Error cleaning local session: {e}")

        # Kill remote processes (only needed for RealSense mode with Jetson camera server)
        if self.config.ego_camera == "realsense":
            self.cleanup_jetson()

        print("[Cleanup] Complete")

    def signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C) gracefully"""
        print("\n[Signal] Shutdown signal received...")
        self.cleanup()
        sys.exit(0)


def _atexit_cleanup():  #
    """Cleanup handler called on program exit"""
    global _deployment_instance
    if _deployment_instance is not None:
        print("\n[atexit] Cleaning up before exit...")
        try:
            _deployment_instance.cleanup_jetson()
        except Exception as e:
            print(f"[atexit] Cleanup error: {e}")


def main():
    """Main entry point with CLI argument parsing

    Usage:
        # Normal deployment
        python scripts/deploy_g1_real_robot.py

        # Cleanup only (kill all Jetson processes)
        python scripts/deploy_g1_real_robot.py --cleanup
    """
    global _deployment_instance

    # Check for cleanup-only mode
    if "--cleanup" in sys.argv or "--cleanup-only" in sys.argv:
        # Quick cleanup mode - just kill Jetson processes
        print("=" * 70)
        print("G1 DEPLOYMENT CLEANUP")
        print("=" * 70)

        # Create minimal config for cleanup
        class MinimalConfig:
            robot_ip = "192.168.123.164"

        # Check if custom IP provided
        for i, arg in enumerate(sys.argv):
            if arg == "--robot_ip" and i + 1 < len(sys.argv):
                MinimalConfig.robot_ip = sys.argv[i + 1]

        deployment = G1RealRobotDeployment(MinimalConfig())
        deployment.cleanup_jetson()
        print("\n[Cleanup] Done!")
        return

    # Create config with tyro CLI parsing
    config = tyro.cli(DeploymentConfig)

    # Override critical settings for real robot deployment
    config.interface = "real"  # Force real interface
    config.env_type = "real"  # Force real environment
    config.hand_type = "aloha"  # Force ALOHA grippers
    config.with_hands = True  # Must have hands enabled for ALOHA
    config.high_elbow_pose = False  # Disable high elbow pose for real robot safety

    # Set sensible defaults if not specified
    if config.body_control_device == "dummy":
        config.body_control_device = "pico"
    if config.hand_control_device == "dummy":
        config.hand_control_device = "pico"

    # Create deployment and register cleanup
    deployment = G1RealRobotDeployment(config)
    _deployment_instance = deployment
    atexit.register(_atexit_cleanup)

    # Run deployment
    deployment.deploy()


if __name__ == "__main__":
    main()
