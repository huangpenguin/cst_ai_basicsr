#!/usr/bin/env python3
"""Resolve persistent storage paths for GitLab GPU training jobs.

Runner bind mounts (see config.toml volumes):
  /cache              -> host /cache
  /home               -> host /mnt/home
  /data               -> host /mnt/data (read-only)

GitLab CI ``cache:`` in .gitlab-ci.yml (``.venv/``, ``.cache/uv/``) is separate:
it speeds up dependency install on the same runner; it is NOT the /cache volume.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CiRunPaths:
    storage_root: Path
    run_root: Path
    experiments_root: Path
    tb_logger_root: Path
    dataset_dir: Path | None = None


def resolve_storage_root(
    *,
    login: str | None = None,
    project_dir: Path | None = None,
) -> Path:
    """Pick first writable persistent root: /home, /cache, then repo fallback."""
    user = login or os.environ.get("GITLAB_USER_LOGIN") or os.environ.get("USER") or "ci"
    repo = project_dir or Path(os.environ.get("CI_PROJECT_DIR", Path(__file__).resolve().parents[1]))
    candidates = [
        Path(f"/home/{user}/cst_ai"),
        Path(f"/cache/{user}/cst_ai"),
        repo / ".ci_storage" / user / "cst_ai",
    ]
    for candidate in candidates:
        try:
            (candidate / "datasets").mkdir(parents=True, exist_ok=True)
            (candidate / "ci_outputs").mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except OSError:
            continue
    tried = ", ".join(str(p) for p in candidates)
    raise SystemExit(f"ERROR: no writable storage root. Tried: {tried}")


def prepare_run_outputs(
    storage_root: Path,
    *,
    pipeline_id: str | None = None,
    debug: bool = False,
    dataset_subdir: str = "swinir_cylinderblock_paired",
) -> CiRunPaths:
    pid = pipeline_id or os.environ.get("CI_PIPELINE_ID") or "local"
    suffix = "_debug" if debug else ""
    run_root = storage_root / "ci_outputs" / f"{pid}{suffix}"
    experiments_root = run_root / "experiments"
    tb_logger_root = run_root / "tb_logger"
    experiments_root.mkdir(parents=True, exist_ok=True)
    tb_logger_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = storage_root / "datasets" / dataset_subdir
    return CiRunPaths(
        storage_root=storage_root,
        run_root=run_root,
        experiments_root=experiments_root,
        tb_logger_root=tb_logger_root,
        dataset_dir=dataset_dir,
    )


def write_markers(project_dir: Path, paths: CiRunPaths) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".ci_experiments_root").write_text(str(paths.experiments_root), encoding="utf-8")
    (project_dir / ".ci_run_root").write_text(str(paths.run_root), encoding="utf-8")
    (project_dir / ".ci_tb_logger_root").write_text(str(paths.tb_logger_root), encoding="utf-8")


def apply_tb_logger_env(paths: CiRunPaths) -> None:
    os.environ["BASICSR_TB_LOGGER_ROOT"] = str(paths.tb_logger_root)


def emit_shell_exports(paths: CiRunPaths) -> None:
    exports = {
        "CI_STORAGE_ROOT": paths.storage_root,
        "CI_RUN_ROOT": paths.run_root,
        "TRAIN_EXPERIMENTS_ROOT": paths.experiments_root,
        "BASICSR_TB_LOGGER_ROOT": paths.tb_logger_root,
    }
    if paths.dataset_dir is not None:
        exports["TRAIN_DATASET_DIR"] = paths.dataset_dir
    for key, value in exports.items():
        print(f'export {key}="{value}"')


def print_paths(paths: CiRunPaths, *, host_hint: bool = True) -> None:
    print(f"storage_root={paths.storage_root}")
    print(f"CI_RUN_ROOT={paths.run_root}")
    print(f"TRAIN_EXPERIMENTS_ROOT={paths.experiments_root}")
    print(f"BASICSR_TB_LOGGER_ROOT={paths.tb_logger_root}")
    if paths.dataset_dir is not None:
        print(f"TRAIN_DATASET_DIR={paths.dataset_dir}")
    if host_hint:
        print("--- persistent on GPU host (same path if /cache and /home are bind-mounted) ---")
        print(f"  text logs: {paths.experiments_root}/<exp_name>/train_*.log")
        print(f"  tensorboard: {paths.tb_logger_root}/<exp_name>/")
        print("  serve TB on host: bash scripts/serve_tensorboard.sh", paths.storage_root / "ci_outputs")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI persistent storage helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create run dirs and print paths")
    prepare.add_argument("--debug", action="store_true", help="Use {pipeline_id}_debug run folder")
    prepare.add_argument("--emit-shell", action="store_true", help="Print export statements for bash eval")
    prepare.add_argument(
        "--dataset-subdir",
        default="swinir_cylinderblock_paired",
        help="Dataset folder under storage_root/datasets/",
    )
    prepare.add_argument("--write-markers", action="store_true", help="Write .ci_* marker files in CI_PROJECT_DIR")

    sub.add_parser("resolve", help="Print storage_root only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "resolve":
        print(resolve_storage_root())
        return 0

    storage_root = resolve_storage_root()
    paths = prepare_run_outputs(
        storage_root,
        debug=args.debug,
        dataset_subdir=args.dataset_subdir,
    )
    apply_tb_logger_env(paths)

    project_dir = Path(os.environ.get("CI_PROJECT_DIR", Path(__file__).resolve().parents[1]))
    if args.write_markers:
        write_markers(project_dir, paths)

    if args.emit_shell:
        emit_shell_exports(paths)
    else:
        print_paths(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
