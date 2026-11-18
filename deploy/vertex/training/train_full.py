"""Full training runner for managed Vertex GPU workers; never called by the UI.

Accepts an immutable GCS tar and SHA256, persists checkpoints while training,
and can restore a committed artifact manifest from a previous run.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urllib.parse import quote

import requests
from gcs import download, headers, object_parts, upload
from prepare_data import inspect


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    return root.joinpath(*p.parts)


def unpack(archive_path, destination):
    with tarfile.open(archive_path) as archive:
        # Large raw data may use hardlinks, but never device files or symlinks.
        members = archive.getmembers()
        required = sum(m.size for m in members if m.isfile())
        if shutil.disk_usage(destination).free < required + 10 * 1024**3:
            raise ValueError("Insufficient free space for data plus 10 GiB reserve")
        for member in members:
            safe_path(destination, member.name)
            if member.issym() or not (member.isfile() or member.isdir() or member.islnk()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
            if member.islnk():
                safe_path(destination, member.linkname)
        archive.extractall(destination, members=members, filter="data")


class Persist:
    def __init__(self, root, uri):
        self.root, self.uri = root, uri.rstrip("/")
        self.saved = {}

    def sync(self, final=False):
        for path in sorted(self.root.rglob("*")):
            if path.suffix == ".tmp" or not path.is_file() or path.name == "artifact-manifest.json":
                continue
            before = path.stat()
            relative = path.relative_to(self.root).as_posix()
            stamp = [before.st_size, before.st_mtime_ns]
            if self.saved.get(relative, {}).get("stamp") == stamp:
                continue
            if not final and path.suffix in (".pth", ".state") and time.time() - before.st_mtime < 30:
                continue
            with tempfile.TemporaryDirectory(prefix="vertex-persist-") as staging:
                snapshot = Path(staging) / "file"
                shutil.copyfile(path, snapshot)
                after = path.stat()
                if [after.st_size, after.st_mtime_ns] != stamp:
                    continue
                digest = sha256(snapshot)
                # Snapshot bytes never change during upload. Old manifests keep
                # referring to immutable content even while logs continue growing.
                object_name = f"objects/{digest}"
                upload(snapshot, f"{self.uri}/{object_name}")
                self.saved[relative] = dict(stamp=stamp, sha256=digest, object=object_name)
        manifest = self.root / "artifact-manifest.json"
        manifest.write_text(json.dumps({"files": self.saved}, indent=2) + "\n")
        # Commit only after every referenced object exists.
        upload(manifest, self.uri + "/artifact-manifest.json")


def claim_output(uri, data_sha256):
    bucket, prefix = object_parts(uri.rstrip("/"))
    url = f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o"
    auth = headers()
    existing = requests.get(url, params={"prefix": prefix + "/", "maxResults": 1}, headers=auth, timeout=60)
    existing.raise_for_status()
    if existing.json().get("items"):
        raise ValueError("Output prefix already contains objects; choose a new run ID")
    response = requests.post(
        f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o",
        params={"uploadType": "media", "name": prefix + "/_run_claim.json", "ifGenerationMatch": 0},
        headers={**auth, "Content-Type": "application/json"},
        data=json.dumps({"data_sha256": data_sha256, "claimed_unix": time.time()}).encode(),
        timeout=60,
    )
    if response.status_code == 412:
        raise ValueError("Output prefix was claimed by another run")
    response.raise_for_status()


def restore(uri, output):
    manifest = output / "restore-manifest.json"
    download(uri.rstrip("/") + "/artifact-manifest.json", manifest)
    files = json.loads(manifest.read_text())["files"]
    for relative, item in files.items():
        target = safe_path(output, relative)
        obj = item["object"]
        if obj != "objects/" + item["sha256"]:
            raise ValueError("Invalid restore object")
        target.parent.mkdir(parents=True, exist_ok=True)
        download(uri.rstrip("/") + "/" + obj, target)
        if sha256(target) != item["sha256"]:
            raise ValueError(f"Restore checksum mismatch: {relative}")
    states = list((output / "experiments").rglob("training_states/*.state"))
    states = [s for s in states if s.stem.isdigit() and (s.parent.parent / "models" / f"net_g_{s.stem}.pth").is_file()]
    if not states:
        raise ValueError("No committed optimizer state with matching model checkpoint")
    return max(states, key=lambda s: int(s.stem))


def configure(base, args):
    import copy

    config = copy.deepcopy(base)
    if args.gpu_count not in (1, 2, 4, 8) or args.batch_per_gpu < 1:
        raise ValueError("Use GPU count 1/2/4/8 and a positive per-GPU batch")
    if args.crop_size < 8 or args.crop_size % config["network_g"]["window_size"]:
        raise ValueError("Crop must be a positive multiple of the network window size")
    if args.iterations < 1 or (args.mode == "acceptance" and args.iterations > 32):
        raise ValueError("Acceptance is bounded to 1..32 optimizer updates")
    config["num_gpu"] = args.gpu_count
    train = config["datasets"]["train"]
    train.update(batch_size_per_gpu=args.batch_per_gpu, gt_size=args.crop_size)
    config["train"]["total_iter"] = args.iterations
    config["val"]["tile_size"] = args.validation_tile
    config["val"]["tile_overlap"] = min(32, args.validation_tile // 4)
    if args.validation_max < 1 or args.validation_interval < 1:
        raise ValueError("Validation maximum and interval must be positive")
    config["val"].update(max_num=args.validation_max, val_freq=args.validation_interval)
    if args.validation_tile < 8 or args.validation_tile % config["network_g"]["window_size"]:
        raise ValueError("Validation tile must be a positive multiple of the network window size")
    if args.mode == "acceptance":
        config["name"] = "vertex_swinir_acceptance"
        train.update(num_worker_per_gpu=0, pin_memory=False)
        config["train"]["scheduler"]["milestones"] = []
        config["logger"].update(print_freq=1, save_checkpoint_freq=max(1, args.iterations // 2))
        config["val"].update(val_freq=min(args.validation_interval, args.iterations), save_img=True)
    else:
        config["name"] = "vertex_swinir_training"
    return config


def verify_artifacts(out, config, expected_updates):
    import torch

    experiment = out / "experiments" / config["name"]
    checkpoint = experiment / "models" / "net_g_latest.pth"
    if not checkpoint.is_file():
        raise RuntimeError("Training returned without a latest checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not saved.get("params"):
        raise RuntimeError("Saved checkpoint has no trainable model parameters")
    if not all(torch.isfinite(tensor).all().item() for tensor in saved["params"].values()):
        raise RuntimeError("Saved checkpoint contains non-finite model values")
    state_paths = sorted((experiment / "training_states").glob("*.state"), key=lambda p: int(p.stem))
    if not state_paths:
        raise RuntimeError("Training returned without a restorable optimizer state")
    state = torch.load(state_paths[-1], map_location="cpu", weights_only=True)
    if (
        not state.get("optimizers")
        or not state.get("schedulers")
        or not all(o.get("state") for o in state["optimizers"])
    ):
        raise RuntimeError("Saved training state lacks optimizer or scheduler")
    if not (experiment / "models" / f"net_g_{state['iter']}.pth").is_file():
        raise RuntimeError("Optimizer state does not have a matching model checkpoint")
    reports = [json.loads(path.read_text()) for path in sorted((out / "telemetry").glob("worker-*.json"))]
    if len(reports) != config["num_gpu"]:
        raise RuntimeError("Missing GPU worker telemetry")
    for report in reports:
        if report["status"] != "SUCCEEDED" or report.get("optimizer_updates") != expected_updates:
            raise RuntimeError("Worker did not complete the expected optimizer updates")
        if not report.get("weights_changed") or report.get("training_peak_allocated_bytes", 0) <= 0:
            raise RuntimeError("Missing actual weight-change or GPU-memory evidence")
    if config["val"]["val_freq"] <= expected_updates and reports[0].get("validation_calls", 0) < 1:
        raise RuntimeError("Expected validation did not execute")
    return {
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "optimizer_state_iteration": state["iter"],
        "workers": reports,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-uri", required=True)
    ap.add_argument("--data-sha256", required=True)
    ap.add_argument("--output-uri", required=True, help="Unique private GCS prefix per run")
    ap.add_argument("--run-dir", type=Path, default=Path("/tmp/full-training"))
    ap.add_argument("--resume-prefix")
    ap.add_argument("--mode", choices=["full", "acceptance"], default="full")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--batch-per-gpu", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--validation-tile", type=int, default=128)
    ap.add_argument("--validation-max", type=int, default=20)
    ap.add_argument("--validation-interval", type=int, default=2000)
    ap.add_argument("--check", action="store_true", help="Print configuration without downloading or training")
    args = ap.parse_args()
    for uri in (args.data_uri, args.output_uri, args.resume_prefix):
        if uri and (not uri.startswith("gs://") or len(uri[5:].split("/")) < 2):
            ap.error("Expected gs://bucket/object-or-prefix")
    if len(args.data_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.data_sha256):
        ap.error("A lowercase SHA256 of the input archive is required")
    config = configure(json.loads((Path(__file__).parent / "train-full.json").read_text()), args)
    if args.check:
        print(
            json.dumps(
                {
                    "gpu_count": config["num_gpu"],
                    "batch_per_gpu": config["datasets"]["train"]["batch_size_per_gpu"],
                    "iterations": config["train"]["total_iter"],
                    "input": args.data_uri,
                    "output": args.output_uri,
                    "mode": args.mode,
                    "crop_size": args.crop_size,
                    "global_batch_size": args.gpu_count * args.batch_per_gpu,
                    "training_started": False,
                },
                indent=2,
            )
        )
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != config["num_gpu"]:
        raise RuntimeError(f"This configuration requires exactly {config['num_gpu']} visible GPUs")
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    data.mkdir()
    archive = root / "dataset.tar"
    download(args.data_uri, archive)
    if sha256(archive) != args.data_sha256:
        raise ValueError("Input archive SHA256 mismatch")
    unpack(archive, data)
    inspect(data)
    archive.unlink()
    claim_output(args.output_uri, args.data_sha256)
    out = root / "output"
    out.mkdir()
    resume_state = restore(args.resume_prefix, out) if args.resume_prefix else None
    contract = {
        "data_sha256": args.data_sha256,
        "network_g": config["network_g"],
        "training": config["train"],
        "batch_per_gpu": config["datasets"]["train"]["batch_size_per_gpu"],
        "gpu_count": config["num_gpu"],
        "crop_size": args.crop_size,
        "validation_tile": args.validation_tile,
        "validation_max": args.validation_max,
        "validation_interval": config["val"]["val_freq"],
        "mode": args.mode,
    }
    contract_path = out / "run-contract.json"
    if resume_state and (not contract_path.is_file() or json.loads(contract_path.read_text()) != contract):
        raise ValueError("Resume requires the same dataset, network and training contract")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    for split in config["datasets"].values():
        for key, filename in [("dataroot_gt", "GT"), ("dataroot_lq", "LQ"), ("normalization", "normalization.json")]:
            split[key] = str(data / filename)
        split["meta_info_file"] = str(data / Path(split["meta_info_file"]).name)
    config["path"]["pretrain_network_g"] = (
        "/workspace/experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
    )
    config["path"]["experiments_root"] = str(out / "experiments")
    config["path"]["resume_state"] = str(resume_state) if resume_state else None
    config_path = out / "effective-train.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    environment = dict(
        os.environ, BASICSR_TB_LOGGER_ROOT=str(out / "tensorboard"), VERTEX_TELEMETRY_DIR=str(out / "telemetry")
    )
    worker = str(Path(__file__).parent / "worker.py")
    if args.gpu_count == 1:
        command = [sys.executable, worker, "-opt", str(config_path)]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.gpu_count}",
            worker,
            "-opt",
            str(config_path),
            "--launcher",
            "pytorch",
        ]
    initial_iteration = int(resume_state.stem) if resume_state else 0
    if initial_iteration >= args.iterations:
        raise ValueError("Resume checkpoint already reached the configured iteration limit")
    persistence = Persist(out, args.output_uri)
    started = time.time()
    with (out / "train.log").open("w") as log:
        child = subprocess.Popen(
            command, cwd="/workspace", env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

        def stop(signum, frame):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        failures = 0
        try:
            while True:
                try:
                    result = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        persistence.sync()
                        failures = 0
                    except Exception as error:
                        failures += 1
                        print(f"Artifact persistence failed ({failures}/3): {type(error).__name__}", flush=True)
                        if failures >= 3:
                            raise RuntimeError("Stopping training after persistent GCS failures") from error
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            summary = {
                "exit_code": child.returncode,
                "elapsed_seconds": time.time() - started,
                "data_sha256": args.data_sha256,
                "configured_iterations": config["train"]["total_iter"],
                "mode": args.mode,
                "gpu_count": args.gpu_count,
                "batch_per_gpu": args.batch_per_gpu,
                "global_batch_size": args.gpu_count * args.batch_per_gpu,
                "crop_size": args.crop_size,
                "initial_iteration": initial_iteration,
                "status": "FAILED",
            }
            try:
                if child.returncode == 0:
                    summary["artifacts"] = verify_artifacts(out, config, args.iterations - initial_iteration)
                    summary["status"] = "SUCCEEDED"
            except Exception as error:
                summary.update(artifact_validation_error=str(error))
                raise
            finally:
                (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                persistence.sync(final=True)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
