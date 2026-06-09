#!/bin/bash
# Startup script for ALOHA gripper DDS bridge on G1 robot onboard computer
# Save this file on the robot at: ~/scripts/start_aloha_bridge.sh
# Make executable: chmod +x ~/scripts/start_aloha_bridge.sh

# Configuration
DDS_DOMAIN=0
DDS_TOPIC="rt/aloha_hand/cmd"
QUEUE_LENGTH=1
LOG_FILE="/tmp/aloha_gripper_bridge.log"

# Find the aloha_gripper_dds_bridge.py script
SCRIPT_DIR="$HOME/aloha_feedback_server"
BRIDGE_SCRIPT="$SCRIPT_DIR/aloha_gripper_dds_bridge.py"

# Check if script exists
if [ ! -f "$BRIDGE_SCRIPT" ]; then
    echo "Error: aloha_gripper_dds_bridge.py not found at $BRIDGE_SCRIPT"
    echo "Please update SCRIPT_DIR in this startup script"
    exit 1
fi

# Check if Arduino is connected
ARDUINO_PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -n 1)
if [ -z "$ARDUINO_PORT" ]; then
    echo "Warning: No Arduino/OpenRB device found"
    echo "Available serial ports:"
    ls -l /dev/tty* 2>/dev/null | grep -E "USB|ACM"
fi

# Kill any existing bridge processes
echo "Stopping any existing ALOHA bridge processes..."
pkill -f "aloha_gripper_dds_bridge.py"
sleep 1

# Start the bridge
echo "Starting ALOHA gripper DDS bridge..."
echo "  DDS Domain: $DDS_DOMAIN"
echo "  DDS Topic: $DDS_TOPIC"
echo "  Queue Length: $QUEUE_LENGTH"
echo "  Log file: $LOG_FILE"

cd "$SCRIPT_DIR"
python3 aloha_gripper_dds_bridge.py \
    --dds-domain $DDS_DOMAIN \
    --dds-hand-topic "$DDS_TOPIC" \
    --queue-length $QUEUE_LENGTH \
    > "$LOG_FILE" 2>&1 &

BRIDGE_PID=$!

# Wait a moment and check if process is running
sleep 2
if ps -p $BRIDGE_PID > /dev/null; then
    echo "✓ ALOHA gripper bridge started successfully (PID: $BRIDGE_PID)"
    echo "  To view logs: tail -f $LOG_FILE"
    echo "  To stop: kill $BRIDGE_PID"
    
    # Save PID for later reference
    echo $BRIDGE_PID > /tmp/aloha_gripper_bridge.pid
else
    echo "✗ Failed to start ALOHA gripper bridge"
    echo "  Check log file: cat $LOG_FILE"
    exit 1
fi
