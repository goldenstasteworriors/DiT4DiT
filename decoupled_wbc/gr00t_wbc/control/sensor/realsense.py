"""RealSense camera sensor module for Intel RealSense cameras."""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError(
        "pyrealsense2 is not installed. Install it with:\n"
        "  pip install pyrealsense2\n"
        "Or for Jetson: sudo apt-get install python3-pyrealsense2"
    )

from gr00t_wbc.control.base.sensor import Sensor
from gr00t_wbc.control.sensor.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class RealSenseConfig:
    """Configuration for the RealSense camera."""

    color_image_dim: Tuple[int, int] = (640, 480)  # RGB camera resolution
    depth_image_dim: Tuple[int, int] = (640, 480)  # Depth camera resolution
    fps: int = 30
    enable_color: bool = True  # Enable RGB stream
    enable_depth: bool = False  # Enable depth stream (optional)
    mount_position: str = CameraMountPosition.EGO_VIEW.value


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for Intel RealSense cameras (D400 series, L500 series, etc.)."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = None,
        device_id: Optional[str] = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        """Initialize the RealSense camera.
        
        Args:
            run_as_server: Whether to run as a sensor server
            port: Port number for server communication
            config: RealSense configuration
            device_id: Serial number of the RealSense device (optional)
            mount_position: Mount position identifier for the camera
        """
        if config is None:
            config = RealSenseConfig()
        
        self.config = config
        self.mount_position = mount_position
        self._run_as_server = run_as_server
        self.device_id = device_id
        
        # Initialize RealSense context and find devices
        self.context = rs.context()
        devices = self.context.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")
        
        print(f"Found {len(devices)} RealSense device(s):")
        for i, dev in enumerate(devices):
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            print(f"  [{i}] {name} (S/N: {serial})")
        
        # Create pipeline
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        # Select specific device if device_id provided
        if device_id is not None:
            self.rs_config.enable_device(device_id)
            print(f"Using device with serial: {device_id}")
        
        # Configure streams
        width, height = config.color_image_dim
        
        if config.enable_color:
            self.rs_config.enable_stream(
                rs.stream.color, 
                width, height, 
                rs.format.rgb8, 
                config.fps
            )
            print(f"Enabled color stream: {width}x{height} @ {config.fps}fps")
        
        if config.enable_depth:
            depth_width, depth_height = config.depth_image_dim
            self.rs_config.enable_stream(
                rs.stream.depth,
                depth_width, depth_height,
                rs.format.z16,
                config.fps
            )
            print(f"Enabled depth stream: {depth_width}x{depth_height} @ {config.fps}fps")
        
        # Start the pipeline
        try:
            self.profile = self.pipeline.start(self.rs_config)
            device = self.profile.get_device()
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"Connected to RealSense: {name} (S/N: {serial})")
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")
        
        # Get the depth scale for converting depth values to meters (if depth enabled)
        if config.enable_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth scale: {self.depth_scale}")
        
        # Warm up the camera
        for _ in range(10):
            self.pipeline.wait_for_frames()
        
        if run_as_server:
            self.start_server(port)
    
    def read(self) -> Optional[Dict[str, Any]]:
        """Read images from the camera.
        
        Returns:
            Dictionary containing 'images' and 'timestamps' keys, or None on error
        """
        try:
            # Wait for frames with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            if frames is None:
                print(f"[ERROR] No frames received from RealSense {self.mount_position}")
                return None
            
            timestamps = {}
            images = {}
            
            # Get color frame if enabled
            if self.config.enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    # Convert to numpy array (already in RGB format)
                    color_image = np.asanyarray(color_frame.get_data())
                    images[self.mount_position] = color_image
                    # Use frame timestamp
                    frame_timestamp_ms = color_frame.get_timestamp()
                    timestamps[self.mount_position] = frame_timestamp_ms / 1000.0
            
            # Get depth frame if enabled
            if self.config.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    # Convert to meters
                    depth_meters = depth_image * self.depth_scale
                    images[f"{self.mount_position}_depth"] = depth_meters
                    frame_timestamp_ms = depth_frame.get_timestamp()
                    timestamps[f"{self.mount_position}_depth"] = frame_timestamp_ms / 1000.0
            
            if len(images) == 0:
                print(f"[ERROR] No valid frames from RealSense {self.mount_position}")
                return None
            
            return {"images": images, "timestamps": timestamps}
            
        except Exception as e:
            print(f"[ERROR] Failed to read from RealSense {self.mount_position}: {e}")
            return None
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get the observation space for the camera."""
        width, height = self.config.color_image_dim
        spaces = {}
        
        if self.config.enable_color:
            spaces[self.mount_position] = {
                "shape": (height, width, 3),
                "dtype": np.uint8,
                "low": 0,
                "high": 255,
            }
        
        if self.config.enable_depth:
            depth_width, depth_height = self.config.depth_image_dim
            spaces[f"{self.mount_position}_depth"] = {
                "shape": (depth_height, depth_width),
                "dtype": np.float32,
                "low": 0.0,
                "high": 10.0,  # Max depth in meters
            }
        
        return spaces
    
    def close(self):
        """Stop the pipeline and release resources."""
        try:
            self.pipeline.stop()
            print(f"RealSense {self.mount_position} stopped")
        except Exception as e:
            print(f"[WARNING] Error stopping RealSense: {e}")
    
    def __del__(self):
        """Destructor to ensure pipeline is stopped."""
        self.close()


if __name__ == "__main__":
    """Test the RealSense sensor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RealSense camera")
    parser.add_argument("--device_id", type=str, default=None, help="RealSense serial number")
    parser.add_argument("--enable_depth", action="store_true", help="Enable depth stream")
    parser.add_argument("--show", action="store_true", help="Show camera feed")
    args = parser.parse_args()
    
    config = RealSenseConfig()
    config.enable_depth = args.enable_depth
    
    print("Initializing RealSense sensor...")
    sensor = RealSenseSensor(
        config=config,
        device_id=args.device_id,
        mount_position=CameraMountPosition.EGO_VIEW.value,
    )
    
    print("Reading frames...")
    try:
        while True:
            data = sensor.read()
            if data is not None:
                for key, img in data["images"].items():
                    if "depth" not in key:
                        print(f"{key}: shape={img.shape}, dtype={img.dtype}")
                        if args.show:
                            cv2.imshow(key, img[..., ::-1])  # RGB to BGR for display
                    else:
                        print(f"{key}: shape={img.shape}, min={img.min():.2f}m, max={img.max():.2f}m")
                
                if args.show:
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
            else:
                print("No data received")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sensor.close()
        cv2.destroyAllWindows()
