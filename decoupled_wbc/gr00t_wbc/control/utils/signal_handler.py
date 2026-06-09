"""Signal handling utilities for graceful shutdown of control loops.

This module provides a centralized way to handle Ctrl+C (SIGINT) and other
termination signals, allowing blocking operations to be interrupted gracefully.

Usage:
    from gr00t_wbc.control.utils.signal_handler import SignalHandler, is_shutdown_requested

    # Initialize at the start of your main function
    signal_handler = SignalHandler()

    # Check in loops
    while not is_shutdown_requested():
        # Do work...

    # Or use the instance method
    while not signal_handler.should_shutdown():
        # Do work...
"""

import atexit
import signal
import sys
import threading
from typing import Callable, List, Optional

# Global shutdown flag - thread-safe
_shutdown_requested = threading.Event()
_cleanup_callbacks: List[Callable] = []
_handler_initialized = False
_original_sigint_handler = None
_original_sigterm_handler = None


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested (Ctrl+C or SIGTERM).

    Returns:
        True if shutdown was requested, False otherwise.
    """
    return _shutdown_requested.is_set()


def request_shutdown():
    """Request shutdown programmatically."""
    _shutdown_requested.set()


def register_cleanup_callback(callback: Callable):
    """Register a callback to be called during cleanup.

    Args:
        callback: A callable that takes no arguments.
    """
    _cleanup_callbacks.append(callback)


def _run_cleanup_callbacks():
    """Run all registered cleanup callbacks."""
    for callback in _cleanup_callbacks:
        try:
            callback()
        except Exception as e:
            print(f"[SignalHandler] Error in cleanup callback: {e}")


def _signal_handler(signum, frame):
    """Handle termination signals."""
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n[SignalHandler] Received {signal_name}, initiating shutdown...")
    _shutdown_requested.set()

    # Run cleanup callbacks
    _run_cleanup_callbacks()

    # If we receive the signal again, force exit
    if signum == signal.SIGINT:
        signal.signal(signal.SIGINT, _force_exit_handler)
        print("[SignalHandler] Press Ctrl+C again to force exit")


def _force_exit_handler(signum, frame):
    """Force exit on second Ctrl+C."""
    print("\n[SignalHandler] Force exit requested")
    sys.exit(1)


class SignalHandler:
    """Context manager and utility class for signal handling.

    This class sets up signal handlers for SIGINT (Ctrl+C) and SIGTERM
    to allow graceful shutdown of control loops.

    Example:
        signal_handler = SignalHandler()

        try:
            while not signal_handler.should_shutdown():
                # Do work...
        finally:
            signal_handler.cleanup()
    """

    def __init__(self, install_handlers: bool = True):
        """Initialize signal handler.

        Args:
            install_handlers: If True, install signal handlers immediately.
        """
        global _handler_initialized, _original_sigint_handler, _original_sigterm_handler

        if install_handlers and not _handler_initialized:
            # Save original handlers
            _original_sigint_handler = signal.signal(signal.SIGINT, _signal_handler)
            _original_sigterm_handler = signal.signal(signal.SIGTERM, _signal_handler)

            # Register atexit handler
            atexit.register(_run_cleanup_callbacks)

            _handler_initialized = True
            print("[SignalHandler] Installed signal handlers (Ctrl+C to shutdown)")

    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if shutdown was requested, False otherwise.
        """
        return is_shutdown_requested()

    def request_shutdown(self):
        """Request shutdown programmatically."""
        request_shutdown()

    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback.

        Args:
            callback: A callable that takes no arguments.
        """
        register_cleanup_callback(callback)

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to be requested.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait forever.

        Returns:
            True if shutdown was requested, False if timeout occurred.
        """
        return _shutdown_requested.wait(timeout=timeout)

    @staticmethod
    def cleanup():
        """Run cleanup callbacks and restore original signal handlers."""
        global _handler_initialized, _original_sigint_handler, _original_sigterm_handler

        _run_cleanup_callbacks()

        if _handler_initialized:
            if _original_sigint_handler is not None:
                signal.signal(signal.SIGINT, _original_sigint_handler)
            if _original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, _original_sigterm_handler)
            _handler_initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def interruptible_sleep(seconds: float, check_interval: float = 0.1) -> bool:
    """Sleep that can be interrupted by shutdown signal.

    Args:
        seconds: Total time to sleep.
        check_interval: How often to check for shutdown signal.

    Returns:
        True if sleep completed normally, False if interrupted by shutdown.
    """
    import time
    elapsed = 0.0
    while elapsed < seconds:
        if is_shutdown_requested():
            return False
        sleep_time = min(check_interval, seconds - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time
    return True


def cleanup_dds_shared_memory():
    """Clean up stale FastRTPS/DDS shared memory files.

    This helps resolve DDS reconnection issues when the process was terminated
    without proper cleanup. Call this during startup or shutdown.

    The function cleans up /dev/shm/fastrtps_* files which are left behind
    when DDS processes exit without proper cleanup.
    """
    import glob
    import os
    import subprocess

    shm_patterns = ["/dev/shm/fastrtps_*", "/dev/shm/sem.fastrtps_*", "/dev/shm/*fastrtps*"]
    cleaned = 0

    # First try normal cleanup (without sudo)
    for pattern in shm_patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                cleaned += 1
            except (OSError, PermissionError):
                pass  # Will try sudo cleanup next

    # If files remain, try with sudo (for files owned by root)
    remaining = []
    for pattern in shm_patterns:
        remaining.extend(glob.glob(pattern))

    if remaining:
        try:
            subprocess.run(
                ["sudo", "rm", "-f"] + remaining,
                check=False,
                capture_output=True,
                timeout=5
            )
            cleaned += len(remaining)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # sudo not available or timed out

    if cleaned > 0:
        print(f"[SignalHandler] Cleaned up {cleaned} stale DDS shared memory files")
