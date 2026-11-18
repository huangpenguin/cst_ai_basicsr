"""Run real four-GPU smoke checks, resume check, training, then fixed test preview."""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = Path("/training_run")
DATA = Path("/data")
PY = os.environ.get("BASICSR_TRAIN_PYTHON", "/opt/basicsr-venv/bin/python")


def state(phase, **extra):
    value = {"phase": phase, "updated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(), **extra}
    temp = RUN / "pipeline_status.json.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(RUN / "pipeline_status.json")
    print(json.dumps(value), flush=True)


def train(configuration):
    subprocess.run(
        [
            PY,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            "basicsr/train.py",
            "-opt",
            f"/training_run/control/{configuration}",
            "--launcher",
            "pytorch",
            "--auto_resume",
        ],
        check=True,
    )


def main():
    import importlib.metadata
    import platform

    import torch
    from torch.version import cuda

    assert torch.cuda.is_available() and torch.cuda.device_count() == 4
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
    }
    (RUN / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    train_names = (DATA / "train.txt").read_text().splitlines()
    val_names = (DATA / "val.txt").read_text().splitlines()
    test_names = (DATA / "test.txt").read_text().splitlines()
    assert not (
        set(train_names) & set(val_names) or set(train_names) & set(test_names) or set(val_names) & set(test_names)
    )
    if not (RUN / "smoke_verified.json").exists():
        # A smoke check can use a checked, complete subset while other ZIPs are prepared.
        while True:
            available = [n for n in train_names if (DATA / "LQ" / n).is_file() and (DATA / "GT" / n).is_file()][:64]
            available_val = [
                n
                for n in (DATA / "val_monitor.txt").read_text().splitlines()
                if (DATA / "LQ" / n).is_file() and (DATA / "GT" / n).is_file()
            ][:2]
            if len(available) == 64 and len(available_val) == 2:
                break
            state("waiting_for_smoke_pairs", train_available=len(available), val_available=len(available_val))
            time.sleep(30)
        (RUN / "control/smoke_train.txt").write_text("\n".join(available) + "\n")
        (RUN / "control/smoke_val.txt").write_text("\n".join(available_val) + "\n")
        state("smoke_training")
        train("smoke.yml")
        state("smoke_resume")
        train("smoke_resume.yml")
        states = list((RUN / "artifacts/smoke_cylinderhead_ddp/training_states").glob("*.state"))
        assert any(p.name == "12.state" for p in states)
        (RUN / "smoke_verified.json").write_text(
            json.dumps({"four_gpu_training": True, "validation": True, "checkpoint_resume_to_iter_12": True}, indent=2)
            + "\n"
        )
    while True:
        status = json.loads((DATA / "prepare_status.json").read_text())
        if status["phase"] == "failed":
            raise RuntimeError(status)
        if status["phase"] == "complete":
            break
        state("waiting_for_pairs", prepared_pairs=status.get("completed"), total_pairs=status.get("total"))
        time.sleep(30)
    assert status["no_slice_overlap"] and status["source_pixels_unchanged"]
    state(
        "training",
        total_iter=30000,
        train_pairs=len(train_names),
        val_pairs=len(val_names),
        test_pairs_reserved=len(test_names),
    )
    train("train.yml")
    state("final_test_preview")
    subprocess.run([PY, "/training_run/control/evaluate_test.py"], check=True)
    state("complete", test_preview="/training_run/test_preview/summary.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state("failed", error=str(exc))
        raise
