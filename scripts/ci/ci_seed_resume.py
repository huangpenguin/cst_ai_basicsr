#!/usr/bin/env python3
"""Seed a stable experiments dir with the newest CI checkpoint for auto_resume.

Searches ``ci_outputs/*/experiments/<exp_name>/training_states/*.state`` and copies
the highest-iter ``*.state`` plus matching ``net_g_*.pth`` into ``--experiments-root``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment folder name; omit when --opt is set (reads YAML name).",
    )
    parser.add_argument(
        "--opt",
        type=Path,
        default=None,
        help="Train YAML; when set, exp name is taken from top-level name: field.",
    )
    parser.add_argument(
        "--ci-outputs-root",
        type=Path,
        default=None,
        help="Default: <experiments-root>/../ci_outputs or /home/<user>/cst_ai/ci_outputs",
    )
    return parser.parse_args()


def _exp_name_from_args(args: argparse.Namespace) -> str:
    if args.opt is not None:
        import yaml

        opt_path = args.opt.expanduser().resolve()
        opt = yaml.safe_load(opt_path.read_text(encoding="utf-8"))
        if not isinstance(opt, dict) or "name" not in opt:
            raise SystemExit(f"YAML missing top-level name: {opt_path}")
        return str(opt["name"])
    if args.exp_name:
        return str(args.exp_name)
    raise SystemExit("Provide --exp-name or --opt")


def _iter_from_state(path: Path) -> int | None:
    match = re.fullmatch(r"(\d+)\.state", path.name)
    return int(match.group(1)) if match else None


def find_best_state(ci_outputs_root: Path, exp_name: str) -> tuple[Path, int] | None:
    best: tuple[Path, int] | None = None
    pattern = f"*/experiments/{exp_name}/training_states/*.state"
    for state_path in ci_outputs_root.glob(pattern):
        iteration = _iter_from_state(state_path)
        if iteration is None:
            continue
        if best is None or iteration > best[1]:
            best = (state_path, iteration)
    return best


def main() -> None:
    args = parse_args()
    experiments_root = args.experiments_root.expanduser().resolve()
    exp_name = _exp_name_from_args(args)
    target_root = experiments_root / exp_name
    target_states = target_root / "training_states"
    target_models = target_root / "models"

    existing_iters = [
        iteration for path in target_states.glob("*.state") if (iteration := _iter_from_state(path)) is not None
    ]
    local_best = max(existing_iters) if existing_iters else -1

    if args.ci_outputs_root is not None:
        ci_outputs_root = args.ci_outputs_root.expanduser().resolve()
    else:
        candidate = experiments_root.parent / "ci_outputs"
        ci_outputs_root = candidate if candidate.is_dir() else Path("/home/huang/cst_ai/ci_outputs")

    best = find_best_state(ci_outputs_root, exp_name)
    if best is None:
        print(f"No CI checkpoint found under {ci_outputs_root} for {exp_name}")
        print(f"Local best iter in {target_states}: {local_best if local_best >= 0 else 'none'}")
        return

    source_state, iteration = best
    if iteration <= local_best:
        print(f"Stable experiments already at iter {local_best}; CI best is {iteration}. No copy.")
        return

    source_exp = source_state.parents[1]
    source_model = source_exp / "models" / f"net_g_{iteration}.pth"
    if not source_model.is_file():
        raise SystemExit(f"Missing model for state {source_state}: {source_model}")

    target_states.mkdir(parents=True, exist_ok=True)
    target_models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_state, target_states / source_state.name)
    shutil.copy2(source_model, target_models / source_model.name)
    print(f"Seeded resume from {source_exp} -> {target_root} (iter={iteration})")


if __name__ == "__main__":
    main()
