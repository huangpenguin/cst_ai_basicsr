#!/usr/bin/env python3
"""Resolve persistent storage paths for GitLab GPU training jobs.

Runner bind mounts (see config.toml volumes):
  /home               -> host /mnt/home   (NFS — datasets + training outputs)
  /data               -> host /mnt/data   (read-only raw TIFF)
  /cache              -> host /cache      (runner volume; **do not** store project data)

GitLab CI ``cache:`` in .gitlab-ci.yml (``.venv/``, ``.cache/uv/``, pretrained weights)
is unrelated to the ``/cache`` volume: it only speeds up dependency install between jobs.
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


def resolve_storage_user(*, login: str | None = None) -> str:
    """Unix account for NFS layout (/home/<user> in CI, /mnt/home/<user> on host).

    GitLab ``GITLAB_USER_LOGIN`` (e.g. huang.pengbin) often differs from the GPU
    host account (e.g. huang). Set ``CST_STORAGE_USER`` in CI or shell to align paths.
    """
    if login:
        return login
    if user := os.environ.get("CST_STORAGE_USER"):
        return user
    if user := os.environ.get("USER"):
        return user
    return os.environ.get("GITLAB_USER_LOGIN") or "ci"


def container_to_host_path(path: Path) -> Path:
    """Map runner container bind mounts to GPU host paths for SSH / docker-dev."""
    parts = path.parts
    if len(parts) >= 3 and parts[1] == "home":
        return Path("/mnt/home").joinpath(*parts[2:])
    if len(parts) >= 2 and parts[1] == "mnt" and len(parts) >= 4 and parts[2] == "home":
        return path
    return path


def _reject_cache_storage(path: Path, *, context: str) -> Path:
    """Training datasets and outputs must not use the runner ``/cache`` volume."""
    if path.parts[:2] == ("/", "cache"):
        raise SystemExit(
            f"ERROR: {context} resolved to {path}, but project data must not use runner /cache.\n"
            "Use the NFS bind mount /home/<CST_STORAGE_USER>/cst_ai instead "
            "(host /mnt/home/<CST_STORAGE_USER>/cst_ai).\n"
            "GitLab cache: in .gitlab-ci.yml only caches .venv/ and uv wheels — that is separate."
        )
    return path


def resolve_dataset_dir(
    storage_root: Path,
    dataset_subdir: str,
    *,
    login: str | None = None,
) -> Path:
    """Dataset path for CI jobs (always on NFS ``/home/<user>/cst_ai/datasets/``)."""
    user = resolve_storage_user(login=login)
    if os.environ.get("CI"):
        return _reject_cache_storage(
            Path(f"/home/{user}/cst_ai/datasets/{dataset_subdir}"),
            context="TRAIN_DATASET_DIR",
        )
    return storage_root / "datasets" / dataset_subdir


def resolve_storage_root(
    *,
    login: str | None = None,
    project_dir: Path | None = None,
) -> Path:
    """Pick writable persistent root under NFS home — never runner ``/cache``."""
    user = resolve_storage_user(login=login)
    repo = project_dir or Path(os.environ.get("CI_PROJECT_DIR", Path(__file__).resolve().parents[2]))
    fallback = repo / ".ci_storage" / user / "cst_ai"
    if os.environ.get("CI"):
        home_root = Path(f"/home/{user}/cst_ai")
        try:
            (home_root / "ci_outputs").mkdir(parents=True, exist_ok=True)
            if not os.access(home_root, os.W_OK):
                raise PermissionError(f"not writable: {home_root}")
        except OSError as exc:
            raise SystemExit(
                "ERROR: CI training outputs must use NFS "
                f"/home/{user}/cst_ai (host /mnt/home/{user}/cst_ai), not runner /cache.\n"
                "Check runner config.toml volumes include: "
                '"/mnt/home:/home:rw"\n'
                f"And ensure /mnt/home/{user}/cst_ai is writable on the GPU host.\n"
                "If the directory exists but is 'not writable', the job container is probably "
                "running as root while NFS uses root_squash. Fix on the GPU host:\n"
                f"  1. uid/gid for {user}: id -u {user}; id -g {user}\n"
                "  2. In /etc/gitlab-runner/config.toml under [runners.docker]:\n"
                f'       user = "<uid>:<gid>"   # e.g. user = "1003:1003" for huang on gpu01\n'
                "  3. sudo gitlab-runner restart\n"
                f"Details: {exc}"
            ) from exc
        return _reject_cache_storage(home_root, context="CI storage_root")
    candidates = [
        Path(f"/mnt/home/{user}/cst_ai"),  # GPU host SSH (NFS)
        Path(f"/home/{user}/cst_ai"),  # docker-dev container (HOME_MOUNT_SOURCE)
        fallback,
    ]
    selected: Path | None = None
    for candidate in candidates:
        try:
            (candidate / "datasets").mkdir(parents=True, exist_ok=True)
            (candidate / "ci_outputs").mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                selected = candidate
                break
        except OSError:
            continue
    if selected is None:
        tried = ", ".join(str(p) for p in candidates)
        raise SystemExit(f"ERROR: no writable storage root. Tried: {tried}")
    return _reject_cache_storage(selected, context="storage_root")


def prepare_run_outputs(
    storage_root: Path,
    *,
    pipeline_id: str | None = None,
    debug: bool = False,
    dataset_subdir: str = "paired_tiff",
) -> CiRunPaths:
    pid = pipeline_id or os.environ.get("CI_PIPELINE_ID") or "local"
    suffix = "_debug" if debug else ""
    run_root = storage_root / "ci_outputs" / f"{pid}{suffix}"
    experiments_root = run_root / "experiments"
    tb_logger_root = run_root / "tb_logger"
    experiments_root.mkdir(parents=True, exist_ok=True)
    tb_logger_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = resolve_dataset_dir(storage_root, dataset_subdir)
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


def update_latest_symlink(paths: CiRunPaths) -> Path:
    """Point storage_root/ci_outputs/latest at this run (host + CI container)."""
    latest = paths.storage_root / "ci_outputs" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(paths.run_root.name, target_is_directory=True)
    return latest


def print_paths(paths: CiRunPaths, *, host_hint: bool = True) -> None:
    print(f"storage_root={paths.storage_root}")
    print(f"CI_RUN_ROOT={paths.run_root}")
    print(f"TRAIN_EXPERIMENTS_ROOT={paths.experiments_root}")
    print(f"BASICSR_TB_LOGGER_ROOT={paths.tb_logger_root}")
    if paths.dataset_dir is not None:
        print(f"TRAIN_DATASET_DIR={paths.dataset_dir}")
    if host_hint:
        host_root = container_to_host_path(paths.storage_root)
        host_run = container_to_host_path(paths.run_root)
        print("--- GPU host (SSH / docker-dev with HOME_MOUNT_SOURCE=/mnt/home) ---")
        print(f"  storage_root: {host_root}")
        print(f"  this run:     {host_run}")
        print(f"  latest link:  {host_root / 'ci_outputs' / 'latest'}")
        print(f"  text logs:    {host_run}/experiments/<exp_name>/train_*.log")
        print(f"  tensorboard:  {host_run}/tb_logger/<exp_name>/")
        print("  serve TB: bash scripts/ci/serve_tensorboard.sh", host_root / "ci_outputs")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI persistent storage helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create run dirs and print paths")
    prepare.add_argument("--debug", action="store_true", help="Use {pipeline_id}_debug run folder")
    prepare.add_argument("--emit-shell", action="store_true", help="Print export statements for bash eval")
    prepare.add_argument(
        "--dataset-subdir",
        default="paired_tiff",
        help="Dataset folder under storage_root/datasets/",
    )
    prepare.add_argument("--write-markers", action="store_true", help="Write .ci_* marker files in CI_PROJECT_DIR")
    prepare.add_argument(
        "--latest-symlink",
        action="store_true",
        help="Update storage_root/ci_outputs/latest -> this run",
    )

    sub.add_parser("resolve", help="Print storage_root only")
    host = sub.add_parser("host-path", help="Print GPU host path for container storage_root")
    host.add_argument("container_path", nargs="?", default=None, help="Container path (default: resolve storage_root)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "resolve":
        print(resolve_storage_root())
        return 0

    if args.command == "host-path":
        if args.container_path:
            print(container_to_host_path(Path(args.container_path)))
        else:
            print(container_to_host_path(resolve_storage_root()))
        return 0

    storage_root = resolve_storage_root()
    paths = prepare_run_outputs(
        storage_root,
        debug=args.debug,
        dataset_subdir=args.dataset_subdir,
    )
    apply_tb_logger_env(paths)

    project_dir = Path(os.environ.get("CI_PROJECT_DIR", Path(__file__).resolve().parents[2]))
    if args.write_markers:
        write_markers(project_dir, paths)

    if args.latest_symlink:
        update_latest_symlink(paths)

    if args.emit_shell:
        emit_shell_exports(paths)
    else:
        print_paths(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
