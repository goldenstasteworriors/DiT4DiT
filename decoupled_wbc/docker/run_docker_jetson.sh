#!/bin/bash

# Run script for Jetson Orin
set -e

# Get the actual user (even when running with sudo)
if [ -n "$SUDO_USER" ]; then
    USERNAME=$SUDO_USER
    USERID=$(id -u $SUDO_USER)
else
    USERNAME=$(whoami)
    USERID=$(id -u)
fi
HOME_DIR="/home/${USERNAME}"
WORKTREE_NAME="decoupled_wbc"
IMAGE_NAME="gr00t_wbc-deploy-jetson"

# Project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running Docker container: ${IMAGE_NAME}"
echo "Project directory: ${PROJECT_ROOT}"

# Check for --root flag to run as root user
RUN_AS_ROOT=false
for arg in "$@"; do
    if [ "$arg" = "--root" ]; then
        RUN_AS_ROOT=true
    fi
done

# Set user options based on flag
if [ "$RUN_AS_ROOT" = true ]; then
    USER_OPTS=""
    echo "Running as root user"
else
    USER_OPTS="--user ${USERID}"
    echo "Running as user ${USERNAME} (${USERID})"
fi

# Run the container with all necessary privileges for robot control
docker run -it --rm \
    --name gr00t_wbc_jetson \
    --network host \
    --privileged \
    --runtime nvidia \
    --gpus all \
    ${USER_OPTS} \
    -v /dev:/dev \
    -v /opt/ros/foxy:/opt/ros/foxy:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${PROJECT_ROOT}:${HOME_DIR}/Projects/${WORKTREE_NAME} \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -e DISPLAY=${DISPLAY} \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e HOME=${HOME_DIR} \
    -w ${HOME_DIR}/Projects/${WORKTREE_NAME} \
    ${IMAGE_NAME}:latest \
    /bin/bash
