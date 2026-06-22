"""Resolve dataset paths from shared DATA_DIR, personal WORK_DIR, and env overrides."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Relative path under /data/<PROJECT_NAME>/ for BasicSR battery smoke defaults.
BATTERY_TIFF_PARTS = (
    "data_0607",
    "260520_1-battery_data",
    "260520_battery image tiff data",
)

CYLINDER_TIFF_PARTS = (
    "data_0607",
    "260520_2-Cylinderblock_data",
    "260520_cylinderblock image tiff data",
)


def project_name() -> str:
    """Project slug for NFS layout: /data/<name> (raw) and /home/<user>/<name> (outputs)."""
    return os.environ.get("PROJECT_NAME", "cst_ai")


def data_root() -> Path:
    """Shared read-only NFS root (bind-mounted from the Docker host)."""
    return Path(os.environ.get("DATA_DIR", "/data"))


def work_root() -> Path | None:
    """Personal writable NFS home (bind-mounted from the Docker host)."""
    value = os.environ.get("WORK_DIR")
    return Path(value) if value else None


def shared_project_root() -> Path:
    """Raw / canonical datasets: ``DATA_DIR/PROJECT_NAME``."""
    return data_root() / project_name()


def personal_project_root() -> Path | None:
    """Personal project workspace: ``WORK_DIR/PROJECT_NAME``."""
    if work := work_root():
        return work / project_name()
    return None


def resolve_dataset_dir(
    *relative_parts: str,
    workspace_fallback: str | None = None,
) -> Path:
    """Return the first existing directory among known candidates.

    Resolution order:
    1. ``DATASET_DIR`` environment variable (absolute path)
    2. ``DATA_DIR/PROJECT_NAME`` / relative_parts  (team raw data)
    3. ``DATA_DIR/PROJECT_NAME/input`` / relative_parts  (legacy layout)
    4. ``datasets/<workspace_fallback>`` under the project root
    """
    if override := os.environ.get("DATASET_DIR"):
        return Path(override)

    rel = Path(*relative_parts)
    shared = shared_project_root()
    candidates: list[Path] = [
        shared / rel,
        shared / "input" / rel,
    ]
    if workspace_fallback:
        candidates.append(ROOT / "datasets" / workspace_fallback)

    for path in candidates:
        if path.is_dir():
            return path

    return candidates[0]


def default_battery_tiff_dir() -> Path:
    return resolve_dataset_dir(
        *BATTERY_TIFF_PARTS,
        workspace_fallback="260520_battery image tiff data",
    )


def default_cylinder_tiff_dir() -> Path:
    return resolve_dataset_dir(
        *CYLINDER_TIFF_PARTS,
        workspace_fallback="260520_cylinderblock image tiff data",
    )


def resolve_output_dir(name: str) -> Path:
    """Return CI/dev output directory under the personal project tree.

    Resolution order:
    1. ``OUTPUT_DIR`` environment variable (absolute path)
    2. ``WORK_DIR/PROJECT_NAME/outputs/<name>``
    3. ``results/<name>`` under the project root (local dev without NFS)
    """
    if override := os.environ.get("OUTPUT_DIR"):
        return Path(override)
    if personal := personal_project_root():
        return personal / "outputs" / name
    return ROOT / "results" / name
