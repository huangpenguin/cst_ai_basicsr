from __future__ import annotations

from pathlib import Path

from scripts.ci_swinir_cylinderblock_debug import build_train_command


def test_build_train_command_forces_single_gpu() -> None:
    cmd = build_train_command(
        dataset_dir=Path("/tmp/dataset"),
        experiments_root=Path("/tmp/experiments"),
        total_iter=30,
        batch_size_per_gpu=1,
        nproc=1,
        master_port=4332,
    )
    assert "num_gpu=1" in cmd
