"""Freeze code/configuration for this training run; never edit the shared checkout."""

import copy
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
job = json.loads((HERE.parent / "config/job.json").read_text())
run = Path(job["run"])
repo = Path(job["repo"])
code = run / "code"
if (run / "control/train.yml").exists():
    raise SystemExit(
        "Run already configured; leave the historical run unchanged and choose a new run directory in config/job.json."
    )
run.mkdir(parents=True, exist_ok=True)
if not code.exists():
    code.mkdir()
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", "HEAD"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(code, filter="data")
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
(run / "source_commit.txt").write_text(commit + "\n")
(run / "control").mkdir(exist_ok=True)
(run / "pretrained").mkdir(exist_ok=True)
weight = repo / "experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
target = run / "pretrained/swinir_noise25.pth"
if not target.exists():
    shutil.copy2(weight, target)

dataset = {
    "type": "PairedTiffImageDataset",
    "dataroot_gt": "/data/GT",
    "dataroot_lq": "/data/LQ",
    "normalization": "/data/normalization.json",
    "filename_tmpl": "{}",
    "io_backend": {"type": "disk"},
}
train = dict(
    dataset,
    name="CylinderheadMean3Train",
    meta_info_file="/data/train.txt",
    gt_size=128,
    use_hflip=True,
    use_rot=True,
    num_worker_per_gpu=2,
    batch_size_per_gpu=4,
    dataset_enlarge_ratio=1,
    prefetch_mode=None,
    pin_memory=True,
)
val = dict(dataset, name="CylinderheadMean3ValidationMonitor", meta_info_file="/data/val_monitor.txt")
name = "train_SwinIR_cylinderhead_mean3_raw_P128_bs4x4_i30k"
opt = {
    "name": name,
    "model_type": "SwinIRModel",
    "scale": 1,
    "num_gpu": 4,
    "manual_seed": 0,
    "datasets": {"train": train, "val": val},
    "network_g": {
        "type": "SwinIR",
        "upscale": 1,
        "in_chans": 1,
        "img_size": 128,
        "window_size": 8,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "",
        "resi_connection": "1conv",
    },
    "path": {
        "pretrain_network_g": "/training_run/pretrained/swinir_noise25.pth",
        "param_key_g": "params_ema",
        "strict_load_g": True,
        "resume_state": None,
        "experiments_root": "/training_run/artifacts",
    },
    "train": {
        "ema_decay": 0.999,
        "optim_g": {"type": "Adam", "lr": 1e-4, "weight_decay": 0, "betas": [0.9, 0.99]},
        "scheduler": {"type": "MultiStepLR", "milestones": [15000, 25000], "gamma": 0.5},
        "total_iter": 30000,
        "warmup_iter": -1,
        "pixel_opt": {"type": "MSELoss", "loss_weight": 1.0, "reduction": "mean"},
    },
    "val": {
        "val_freq": 2000,
        "save_img": True,
        "max_num": 20,
        "tile_size": 256,
        "tile_overlap": 32,
        "tb_image_count": 2,
        "metrics": {
            "psnr": {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": False},
            "ssim": {"type": "calculate_ssim", "crop_border": 0, "test_y_channel": False},
        },
    },
    "logger": {
        "print_freq": 100,
        "save_checkpoint_freq": 1000,
        "use_tb_logger": True,
        "wandb": {"project": None, "entity": None, "resume_id": None},
    },
    "dist_params": {"backend": "nccl", "timeout_min": 120},
}


# JSON is valid YAML and is read by the repository's YAML loader.
def write(name, value):
    (run / "control" / name).write_text(json.dumps(value, indent=2) + "\n")


write("train.yml", opt)
smoke = copy.deepcopy(opt)
smoke["name"] = "smoke_cylinderhead_ddp"
smoke["train"]["total_iter"] = 8
smoke["train"]["scheduler"]["milestones"] = []
smoke["logger"].update(print_freq=1, save_checkpoint_freq=4, use_tb_logger=False)
smoke["val"].update(val_freq=8, max_num=2, save_img=False)
smoke["datasets"]["train"]["meta_info_file"] = "/training_run/control/smoke_train.txt"
smoke["datasets"]["val"]["meta_info_file"] = "/training_run/control/smoke_val.txt"
write("smoke.yml", smoke)
resume = copy.deepcopy(smoke)
resume["train"]["total_iter"] = 12
write("smoke_resume.yml", resume)
write(
    "training_contract.json",
    {
        "code_commit": commit,
        "model": name,
        "gpus": 4,
        "batch_per_gpu": 4,
        "global_batch_size": 16,
        "patch_size": 128,
        "total_iter": 30000,
        "train_manifest": "/data/train.txt",
        "validation_monitor": "/data/val_monitor.txt",
        "test_policy": "/data/test.txt and /data/test_preview.txt are not referenced by training or validation",
        "alignment": "none; preserved original source TIFFs at user request",
        "metrics_note": (
            "Training validation uses the existing BasicSR uint8 display metrics; final test "
            "preview uses normalized float PSNR/SSIM"
        ),
    },
)
print("Prepared run", run, "source commit", commit, flush=True)

# Place runtime entrypoints beside the generated configuration.
for source in [
    HERE / "pipeline.py",
    HERE / "prepare_pairs.py",
    HERE / "start_on_gpu01.sh",
    HERE.parent / "evaluation/evaluate_test.py",
]:
    shutil.copy2(source, run / "control" / source.name)
