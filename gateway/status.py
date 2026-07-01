"""Stub: Gateway status helpers (gateway removed)."""
import os

def _pid_exists(pid: int) -> bool:
    """Check if a PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def terminate_pid(pid: int) -> bool:
    """Try to terminate a process by PID."""
    try:
        os.kill(pid, 15)  # SIGTERM
        return True
    except OSError:
        return False

def get_running_pid() -> int | None:
    """No gateway running in trimmed version."""
    return None

def read_runtime_status() -> dict:
    """No gateway status available."""
    return {}

def looks_like_gateway_command_line(cmdline: str) -> bool:
    """Always returns False — no gateway."""
    return False
