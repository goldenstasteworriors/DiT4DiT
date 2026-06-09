import os
import subprocess
from pathlib import Path


def _test_display(display: str) -> bool:
    """Test if a display is accessible using xdpyinfo."""
    try:
        env = os.environ.copy()
        env["DISPLAY"] = display
        result = subprocess.run(
            ["xdpyinfo"],
            capture_output=True,
            timeout=2,
            env=env
        )
        return result.returncode == 0
    except Exception:
        return False


# Auto-detect display and X authority for current user if not set
display_verified = False

if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    detected = False

    # Method 1: Scan /tmp/.X11-unix for available X sockets and test each one
    try:
        x11_dir = Path("/tmp/.X11-unix")
        if x11_dir.exists():
            # Try each X socket to find one that's accessible
            for socket in sorted(x11_dir.glob("X*")):
                display_num = socket.name[1:]  # Remove 'X' prefix
                test_display = f":{display_num}"
                if _test_display(test_display):
                    os.environ["DISPLAY"] = test_display
                    print(f"[ImageViewer] Found working DISPLAY={test_display} via X11 socket scan")
                    detected = True
                    display_verified = True
                    break
    except Exception:
        pass

    # Method 2: Try loginctl to get the user's graphical session display
    if not detected:
        try:
            username = os.environ.get("USER", "")

            # Get user's sessions
            result = subprocess.run(
                ["loginctl", "list-sessions", "--no-legend"],
                capture_output=True,
                text=True,
                timeout=2
            )

            # Find session for current user
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[2] == username:
                    session_id = parts[0]

                    # Get display from session properties
                    session_result = subprocess.run(
                        ["loginctl", "show-session", session_id, "-p", "Display"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )

                    for prop_line in session_result.stdout.splitlines():
                        if prop_line.startswith("Display="):
                            display = prop_line.split("=", 1)[1].strip()
                            if display and display.startswith(":"):
                                if _test_display(display):
                                    os.environ["DISPLAY"] = display
                                    print(f"[ImageViewer] Auto-detected DISPLAY={display} via loginctl for user '{username}'")
                                    detected = True
                                    display_verified = True
                                    break

                if detected:
                    break
        except Exception:
            pass

    # Method 3: Parse 'who' output if loginctl didn't work
    if not detected:
        try:
            result = subprocess.run(
                ["who", "-u"],
                capture_output=True,
                text=True,
                timeout=2
            )
            username = os.environ.get("USER", "")

            for line in result.stdout.splitlines():
                if username in line and ":" in line:
                    # Extract display from output like "username :N"
                    parts = line.split()
                    for part in parts:
                        if part.startswith(":") and part[1:].split('.')[0].isdigit():
                            if _test_display(part):
                                os.environ["DISPLAY"] = part
                                print(f"[ImageViewer] Auto-detected DISPLAY={part} via 'who' for user '{username}'")
                                detected = True
                                display_verified = True
                                break

                if detected:
                    break
        except Exception:
            pass

    # Final fallback: try :0, :1, :2 in order
    if not detected:
        for fallback_display in [":0", ":1", ":2"]:
            if _test_display(fallback_display):
                os.environ["DISPLAY"] = fallback_display
                print(f"[ImageViewer] Found working DISPLAY={fallback_display} via fallback scan")
                detected = True
                display_verified = True
                break

        if not detected:
            os.environ["DISPLAY"] = ":0"
            print("[ImageViewer] Could not find working display, using fallback DISPLAY=:0")
else:
    # DISPLAY was already set, verify it works
    if _test_display(os.environ["DISPLAY"]):
        display_verified = True
        print(f"[ImageViewer] Verified existing DISPLAY={os.environ['DISPLAY']} is accessible")
    else:
        print(f"[ImageViewer] Existing DISPLAY={os.environ['DISPLAY']} is not accessible, trying to find another...")
        # Try to find a working display
        for fallback_display in [":0", ":1", ":2"]:
            if _test_display(fallback_display):
                os.environ["DISPLAY"] = fallback_display
                print(f"[ImageViewer] Switched to working DISPLAY={fallback_display}")
                display_verified = True
                break

# Set XAUTHORITY if not set (needed for X11 authentication)
if "XAUTHORITY" not in os.environ or not os.environ["XAUTHORITY"]:
    uid = os.getuid()
    xauth_locations = [
        f"/run/user/{uid}/gdm/Xauthority",  # GDM location
        os.path.expanduser("~/.Xauthority"),  # Standard location
    ]
    for xauth_path in xauth_locations:
        if Path(xauth_path).exists():
            os.environ["XAUTHORITY"] = xauth_path
            print(f"[ImageViewer] Using XAUTHORITY={xauth_path}")
            break

import matplotlib

# Check if tkinter is available
tkinter_available = False
try:
    import tkinter
    tkinter_available = True
except ImportError:
    pass

# Choose backend based on display availability and tkinter
if display_verified and tkinter_available:
    # Display is accessible and tkinter is available - use TkAgg
    try:
        matplotlib.use('TkAgg', force=True)
        print("[ImageViewer] Using TkAgg backend (interactive display)")
    except Exception as e:
        matplotlib.use('Agg')
        print(f"[ImageViewer] TkAgg failed ({e}), using Agg backend")
elif tkinter_available:
    # Tkinter available but display not verified - still try TkAgg
    try:
        matplotlib.use('TkAgg')
        print("[ImageViewer] Using TkAgg backend (display not verified, may fail)")
    except Exception as e:
        matplotlib.use('Agg')
        print(f"[ImageViewer] TkAgg failed ({e}), using Agg backend")
else:
    matplotlib.use('Agg')
    print("[ImageViewer] Tkinter not available, using Agg backend (no interactive display)")
    print("[ImageViewer] To enable interactive display, install: python3-tk or conda install tk")

import matplotlib.pyplot as plt
import numpy as np


class ImageViewer:
    def __init__(self, title="Image Viewer", figsize=(8, 6), num_images=1, image_titles=None):
        self.title = title
        self.figsize = figsize
        self.num_images = num_images
        self.image_titles = image_titles or [f"Camera {i+1}" for i in range(num_images)]

        # Enable interactive mode before creating subplots
        plt.ion()

        if num_images == 1:
            self._fig, self._ax = plt.subplots(figsize=self.figsize)
            self._ax.set_title(self.title)
            self._im = self._ax.imshow(np.zeros((100, 100)))
            self._ax.axis("off")
            self._axes = [self._ax]
            self._images = [self._im]
        else:
            # Calculate grid dimensions
            cols = min(num_images, 3)  # Max 3 columns
            rows = (num_images + cols - 1) // cols

            # Adjust figure size based on number of images
            fig_width = self.figsize[0] * min(cols, 2)
            fig_height = self.figsize[1] * rows / 2

            self._fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            self._fig.suptitle(self.title)

            # Flatten axes array for easier access
            if num_images == 2:
                axes = [axes[0], axes[1]]
            elif rows == 1:
                axes = axes if cols > 1 else [axes]
            else:
                axes = axes.flatten()

            self._axes = []
            self._images = []

            for i in range(num_images):
                ax = axes[i]
                ax.set_title(self.image_titles[i])
                im = ax.imshow(np.zeros((100, 100)))
                ax.axis("off")
                self._axes.append(ax)
                self._images.append(im)

            # Hide unused subplots
            for i in range(num_images, len(axes)):
                axes[i].set_visible(False)

        # Show the figure initially to make window appear
        self._fig.show()

    def show(self, image_array):
        """Show a single image (backward compatibility)"""
        if self.num_images == 1:
            self._images[0].set_data(image_array)
        else:
            # If multiple viewers but single image provided, show in first viewer
            self._images[0].set_data(image_array)

        # non-blocking update
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def show_multiple(self, images):
        """Show multiple images"""
        for i, img in enumerate(images):
            if i < len(self._images) and img is not None:
                self._images[i].set_data(img)
                # Auto-adjust aspect ratio
                self._axes[i].set_aspect("auto")

        # non-blocking update
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        plt.close(self._fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
