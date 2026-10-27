"""GPU resource inspection for the denoise demo UI."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GpuProcess(BaseModel):
    """A compute process reported by nvidia-smi."""

    model_config = ConfigDict(extra="forbid")

    pid: int
    process_name: str
    command: str
    used_memory_mib: int


class GpuStatus(BaseModel):
    """A point-in-time status snapshot for one GPU."""

    model_config = ConfigDict(extra="forbid")

    gpu_index: int
    name: str | None = None
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    memory_free_mib: int | None = None
    utilization_percent: int | None = None
    processes: list[GpuProcess] = Field(default_factory=list)
    available: bool
    error: str | None = None


def _query_nvidia_smi(query: str, gpu_index: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=3,
    )
    return result.stdout


def _first_csv_row(output: str, description: str) -> list[str]:
    rows = [row for row in csv.reader(io.StringIO(output)) if any(cell.strip() for cell in row)]
    if len(rows) != 1:
        raise ValueError(f"expected one {description} row, got {len(rows)}")
    return [cell.strip() for cell in rows[0]]


def _process_command(pid: int, fallback: str) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return f"{fallback} (command unavailable)"
    command = " ".join(part for part in raw.decode("utf-8", errors="replace").split("\x00") if part)
    return command or fallback


def query_gpu_status(gpu_index: int = 0) -> GpuStatus:
    """Query GPU memory and compute processes using the local nvidia-smi."""
    try:
        gpu_row = _first_csv_row(
            _query_nvidia_smi(
                "gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                gpu_index,
            ),
            "GPU",
        )
        if len(gpu_row) != 6:
            raise ValueError(f"expected six GPU fields, got {len(gpu_row)}")

        process_output = _query_nvidia_smi(
            "compute-apps=pid,process_name,used_memory",
            gpu_index,
        )
        processes: list[GpuProcess] = []
        for row in csv.reader(io.StringIO(process_output)):
            values = [cell.strip() for cell in row]
            if not values or not any(values):
                continue
            if len(values) != 3:
                raise ValueError(f"expected three process fields, got {len(values)}")
            pid = int(values[0])
            processes.append(
                GpuProcess(
                    pid=pid,
                    process_name=values[1],
                    command=_process_command(pid, values[1]),
                    used_memory_mib=int(values[2]),
                )
            )

        return GpuStatus(
            gpu_index=gpu_index,
            name=gpu_row[1],
            memory_total_mib=int(gpu_row[2]),
            memory_used_mib=int(gpu_row[3]),
            memory_free_mib=int(gpu_row[4]),
            utilization_percent=int(gpu_row[5]),
            processes=processes,
            available=True,
        )
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return GpuStatus(
            gpu_index=gpu_index,
            available=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _format_mib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024:.1f} GiB"


def format_gpu_status_markdown(status: GpuStatus) -> str:
    """Render an explicit GPU status panel for Gradio."""
    title = f"## GPU {status.gpu_index} status"
    if not status.available:
        return f"{title}\n\n❌ **Unavailable**\n\n`{status.error}`"

    current_pid = os.getpid()
    own_processes = [process for process in status.processes if process.pid == current_pid]
    other_processes = [process for process in status.processes if process.pid != current_pid]
    if other_processes:
        state = f"⚠️ **Other compute processes are using GPU {status.gpu_index}**"
    else:
        state = f"✅ **No other compute process is using GPU {status.gpu_index}**"

    lines = [
        title,
        "",
        state,
        f"- Device: `{status.name}`",
        f"- Memory: `{_format_mib(status.memory_used_mib)} / {_format_mib(status.memory_total_mib)}` "
        f"used; `{_format_mib(status.memory_free_mib)}` free",
        f"- GPU utilization: `{status.utilization_percent}%`",
    ]
    if own_processes:
        own_memory = sum(process.used_memory_mib for process in own_processes)
        lines.append(f"- This demo process: `{_format_mib(own_memory)}`")
    else:
        lines.append("- This demo process: `not using GPU yet`")

    if other_processes:
        lines.extend(["", "### Other compute processes"])
        for process in other_processes:
            lines.append(f"- PID `{process.pid}` — `{_format_mib(process.used_memory_mib)}` — `{process.command}`")
    return "\n".join(lines)
