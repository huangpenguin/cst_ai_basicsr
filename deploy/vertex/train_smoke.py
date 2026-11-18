"""Run bounded BasicSR GPU training and export its checkpoint/log to GCS."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import torch
from gcs import upload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--data-archive", type=Path, default=Path(__file__).parent / "bundle/smoke-data.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/vertex-training-smoke"))
    parser.add_argument("--output-uri", default=os.environ.get("VERTEX_OUTPUT_GCS"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This training validation requires a GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = args.output_dir / "data"
    data.mkdir()
    with tarfile.open(args.data_archive) as archive:
        archive.extractall(data, filter="data")
    config = json.loads((Path(__file__).parent / "train-template.json").read_text())
    train = config["datasets"]["train"]
    train.update(
        dataroot_gt=str(data / "GT"), dataroot_lq=str(data / "LQ"), normalization=str(data / "normalization.json")
    )
    train.pop("meta_info_file", None)
    config["path"]["experiments_root"] = str(args.output_dir / "training")
    config["train"]["total_iter"] = args.iterations
    config_path = args.output_dir / "effective_train.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    log_path = args.output_dir / "train.log"
    started = time.monotonic()
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "basicsr.train", "-opt", str(config_path)],
            cwd="/workspace",
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    checkpoint = args.output_dir / "training" / config["name"] / "models/net_g_latest.pth"
    summary = {
        "purpose": "GPU/data/optimizer/artifact connectivity validation; not a completed model training run",
        "exit_code": result.returncode,
        "configured_iterations": args.iterations,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch_size_per_gpu": 1,
        "gpu_count": 1,
        "crop_size": 128,
        "original_training": {"gpu_count": 4, "batch_size_per_gpu": 4, "global_batch_size": 16},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_uri:
        prefix = args.output_uri.rstrip("/")
        for source, relative in [
            (config_path, "training/effective_train.json"),
            (log_path, "training/train.log"),
            (summary_path, "training/summary.json"),
        ]:
            upload(source, prefix + "/" + relative)
        if checkpoint.is_file():
            upload(checkpoint, prefix + "/model/net_g_latest.pth")
    print(json.dumps(summary), flush=True)
    print(log_path.read_text()[-6000:], flush=True)
    if result.returncode or not checkpoint.is_file():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
