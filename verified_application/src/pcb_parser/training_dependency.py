"""Read-only completion gate for a known Windows training controller."""
import ctypes
import json
import os
from pathlib import Path


def windows_process_alive(pid):
    """Query process state without sending a signal (including signal zero)."""
    if os.name != "nt":
        raise RuntimeError("This process query requires Windows")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        # Only ERROR_INVALID_PARAMETER establishes that the PID no longer exists.
        return ctypes.get_last_error() != 87
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def prerequisite_state(job, process_alive=windows_process_alive):
    state = dict(ready=False, job=job["name"], reason="completion_not_confirmed")
    try:
        status = json.loads(Path(job["status_path"]).read_text(encoding="utf-8-sig"))
        training = json.loads(Path(job["training_record_path"]).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return state | dict(reason="completion_record_unavailable", error_type=type(exc).__name__)
    if not isinstance(status, dict) or not isinstance(training, dict):
        return state | dict(reason="invalid_completion_record")
    state.update(job_status=status.get("status"), training_status=training.get("status"),
                 epoch_completed=status.get("epoch_completed", training.get("completed_epochs")),
                 epochs_planned=status.get("epochs_planned"), process_id=status.get("pid"))
    if status.get("run", job["expected_run"]) != job["expected_run"]:
        return state | dict(reason="unexpected_run")
    if training.get("protocol_sha256") != job["training_protocol_sha256"]:
        return state | dict(reason="unexpected_training_protocol")
    if status.get("status") != "completed" or training.get("status") != "trained":
        return state
    completed = training.get("completed_epochs")
    if type(completed) is not int or completed <= 0:
        return state | dict(reason="no_completed_training_epochs")
    pid = status.get("pid")
    if type(pid) is not int or pid != job["expected_process_id"]:
        return state | dict(reason="unexpected_controller_process")
    if process_alive(pid):
        return state | dict(reason="controller_still_running")
    return state | dict(ready=True, reason="controller_completed_and_exited")
