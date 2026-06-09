import os
import queue
import sys
import threading
import time

import av
import numpy as np


class VideoWriter:
    """Thread-safe video writer using PyAV/ffmpeg.

    Writes video frames to disk using H.264 encoding in a background thread.
    Ensures proper cleanup and graceful shutdown without deadlocks.

    Thread Safety:
        - Uses non-daemon threads to ensure proper cleanup on exit
        - Queue-based communication between main and worker threads
        - STOP_SENTINEL pattern for graceful shutdown

    Performance:
        - Asynchronous encoding via background thread
        - Buffered queue (default 50 frames)
        - H.264 encoding with yuv420p pixel format
    """

    _STOP_SENTINEL = object()  # Unique sentinel to signal thread termination
    _debug_enabled = None  # Cache debug flag check

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
        relative_path: str = None,
    ):
        self.output_path = output_path  # Full path for file operations
        self.relative_path = relative_path  # Relative path to return (optional)
        self._first_frame = True  # Track first frame to suppress x264 info output
        self._stopped = False
        self._error = None  # Store any exception from worker thread

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)

        # PyAV 12.x requires explicit format specification for proper codec resolution
        self.container = av.open(output_path, mode="w", format="mp4")

        # Use libx264 codec name for compatibility with PyAV 12.x
        # PyAV 12+ requires the full codec name (libx264) rather than short name (h264)
        codec_name = "libx264" if codec == "h264" else codec
        self.stream = self.container.add_stream(codec_name, rate=int(fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"

        # Start worker thread (non-daemon for proper cleanup)
        self.thread = threading.Thread(target=self._writer_worker, daemon=False)
        self.thread.start()

        import sys
        if "--debug" in sys.argv:
            print(f"DEBUG [VideoWriter.__init__]: Created thread {self.thread.name} for {output_path}")

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), f"""Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}.
            Expected dimensions: {self.stream.width}x{self.stream.height}"""

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the encoding queue.

        Args:
            frame: RGB frame as numpy array (height, width, 3)

        Raises:
            RuntimeError: If writer has been stopped
            AssertionError: If frame dimensions don't match
        """
        if self._stopped:
            raise RuntimeError("Cannot add frame to stopped VideoWriter")
        self._assert_dimensions(frame)
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        """Background thread worker that encodes frames.

        Runs until _STOP_SENTINEL is received, then exits cleanly.
        Stores any exceptions in self._error for main thread to check.
        """
        import sys
        import threading
        debug = "--debug" in sys.argv

        if debug:
            thread_name = threading.current_thread().name
            print(f"DEBUG [Worker {thread_name}]: Worker thread started for {self.output_path}")

        try:
            while True:
                frame = self.queue.get()

                # Check for stop sentinel
                if frame is self._STOP_SENTINEL:
                    if debug:
                        print(f"DEBUG [Worker {thread_name}]: Received STOP_SENTINEL, exiting")
                    break  # Exit cleanly

                if frame is None:
                    continue  # Skip None frames

                self._assert_dimensions(frame)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                # Suppress stderr for first frame encoding (x264 prints info then)
                if self._first_frame:
                    stderr_fd = sys.stderr.fileno()
                    old_stderr = os.dup(stderr_fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stderr_fd)
                    try:
                        packets = self.stream.encode(frame)
                        for packet in packets:
                            self.container.mux(packet)
                    finally:
                        os.dup2(old_stderr, stderr_fd)
                        os.close(old_stderr)
                        os.close(devnull)
                        self._first_frame = False
                else:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)

        except Exception as e:
            # Store exception for main thread to check
            self._error = e
            print(f"ERROR in video writer thread: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def _flush_stream(self) -> None:
        """Flush any remaining packets in the encoder."""
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self, timeout: float = 30.0) -> str:
        """Gracefully stop the video writer and close the file.

        This is a blocking call that waits for the worker thread to finish.

        Args:
            timeout: Maximum time to wait for thread to finish (seconds)

        Returns:
            Relative path if provided during init, otherwise full path

        Raises:
            RuntimeError: If thread doesn't stop within timeout or worker had an error
        """
        import sys
        debug = "--debug" in sys.argv

        if debug:
            print(f"DEBUG [VideoWriter.stop]: Starting stop for {self.output_path}")
            print(f"  _stopped={self._stopped}, thread.is_alive()={self.thread.is_alive()}")

        if self._stopped:
            if debug:
                print(f"DEBUG [VideoWriter.stop]: Already stopped, returning")
            return self.relative_path if self.relative_path else self.output_path

        # Signal worker thread to exit by sending sentinel
        if debug:
            print(f"DEBUG [VideoWriter.stop]: Putting STOP_SENTINEL on queue")
        self.queue.put(self._STOP_SENTINEL)

        # Wait for worker thread to finish
        if debug:
            print(f"DEBUG [VideoWriter.stop]: Waiting for thread to join (timeout={timeout}s)")
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            if debug:
                print(f"DEBUG [VideoWriter.stop]: Thread still alive after timeout!")
            raise RuntimeError(
                f"Video writer thread did not stop within {timeout}s timeout. "
                "Possible deadlock or slow encoding."
            )

        if debug:
            print(f"DEBUG [VideoWriter.stop]: Thread joined successfully")

        # Check if worker thread had an error
        if self._error is not None:
            raise RuntimeError(f"Video encoding failed: {self._error}")

        # Now safe to flush and close (worker thread has exited)
        self._flush_stream()
        self.container.close()
        self._stopped = True

        # Verify output file exists
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Video file not created: {self.output_path}")
        if os.path.getsize(self.output_path) == 0:
            raise ValueError(f"Video file is empty: {self.output_path}")

        return self.relative_path if self.relative_path else self.output_path

    def cancel(self) -> None:
        """Cancel video writing and delete the output file.

        This immediately stops the worker thread and removes the incomplete video.
        """
        import sys
        debug = "--debug" in sys.argv

        if debug:
            print(f"DEBUG [VideoWriter.cancel]: Starting cancel for {self.output_path}")
            print(f"  _stopped={self._stopped}, thread.is_alive()={self.thread.is_alive()}")

        if not self._stopped:
            # Signal thread to stop
            try:
                if debug:
                    print(f"DEBUG [VideoWriter.cancel]: Putting STOP_SENTINEL on queue")
                self.queue.put(self._STOP_SENTINEL)

                if debug:
                    print(f"DEBUG [VideoWriter.cancel]: Waiting for thread to join (timeout=5.0s)")
                self.thread.join(timeout=5.0)

                if debug:
                    print(f"DEBUG [VideoWriter.cancel]: After join, thread.is_alive()={self.thread.is_alive()}")
            except Exception as e:
                if debug:
                    print(f"DEBUG [VideoWriter.cancel]: Exception during stop: {e}")
                pass  # Best effort

        # Delete incomplete file
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
                if debug:
                    print(f"DEBUG [VideoWriter.cancel]: Deleted file {self.output_path}")
        except:
            pass  # Best effort

        # Close container
        if not self._stopped:
            try:
                self.container.close()
                if debug:
                    print(f"DEBUG [VideoWriter.cancel]: Closed container")
            except:
                pass  # Best effort
            self._stopped = True

        if debug:
            print(f"DEBUG [VideoWriter.cancel]: Finished cancel, _stopped={self._stopped}")

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        if not self._stopped:
            try:
                self.cancel()
            except:
                pass  # Best effort cleanup
