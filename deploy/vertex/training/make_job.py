"""Generate a reviewable CustomJobSpec. Does not submit or mutate cloud resources."""

import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--image", required=True, help="New full-training image pinned by digest; never the smoke image")
p.add_argument("--machine", required=True)
p.add_argument("--accelerator", required=True)
p.add_argument("--data-uri", required=True)
p.add_argument("--data-sha256", required=True)
p.add_argument("--output-uri", required=True)
p.add_argument("--resume-prefix")
p.add_argument("--mode", choices=["full", "acceptance"], default="full")
p.add_argument("--gpu-count", type=int, choices=[1, 2, 4, 8], default=1)
p.add_argument("--batch-per-gpu", type=int, default=1)
p.add_argument("--iterations", type=int, default=30000)
p.add_argument("--crop-size", type=int, default=128)
p.add_argument("--validation-tile", type=int, default=128)
p.add_argument("--validation-max", type=int, default=20)
p.add_argument("--validation-interval", type=int, default=2000)
timeout = p.add_mutually_exclusive_group(required=True)
timeout.add_argument("--timeout-hours", type=int)
timeout.add_argument("--timeout-seconds", type=int)
p.add_argument("--disk-gb", type=int, default=300)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not re.fullmatch(
    r"asia-northeast1-docker\.pkg\.dev/hallowed-pipe-497802-i0/denoise/[^@]+@sha256:[0-9a-f]{64}", a.image
):
    p.error("Use a pinned training image in the existing private registry")
if "fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f" in a.image:
    p.error("The original image contains smoke only; build the full-training overlay first")
if not re.fullmatch("[0-9a-f]{64}", a.data_sha256):
    p.error("Expected the input archive SHA256")
duration = a.timeout_seconds if a.timeout_seconds is not None else a.timeout_hours * 3600
if not 60 <= duration <= 168 * 3600 or a.disk_gb < 150:
    p.error("Use a finite timeout (60 seconds..168 hours), and at least 150 GiB disk")
if a.mode == "full" and a.iterations < 1000:
    p.error(
        "Full mode requires at least 1000 iterations to save optimizer state; use bounded acceptance for short checks"
    )
if a.mode == "acceptance" and (a.iterations > 32 or duration > 3600):
    p.error("Acceptance is limited to 1..32 iterations and at most 3600 seconds")
if (
    a.validation_max < 1
    or a.validation_interval < 1
    or a.iterations < 1
    or a.batch_per_gpu < 1
    or a.crop_size < 8
    or a.crop_size % 8
    or a.validation_tile < 8
    or a.validation_tile % 8
):
    p.error("Positive iterations/batch and crop/tile multiples of 8 are required")
# This project targets Tokyo. Location availability for serving does not imply
# availability for Custom Training; keep unsupported families out of job specs.
if a.accelerator in ("NVIDIA_TESLA_A100", "NVIDIA_A100_80GB", "NVIDIA_L4", "NVIDIA_TESLA_V100"):
    p.error("This accelerator is not a Tokyo Custom Training option; select a supported training family")
if a.accelerator == "NVIDIA_TESLA_T4" and not a.machine.startswith("n1-"):
    p.error("T4 Custom Training uses an N1 machine type")
if a.mode == "acceptance" and a.gpu_count != 1:
    p.error("This bounded acceptance profile uses exactly one GPU")
for uri in [a.data_uri, a.output_uri, a.resume_prefix]:
    if uri and not re.fullmatch(r"gs://[^/]+/.+", uri):
        p.error("Expected private gs://bucket/object-or-prefix")
args = [
    "python",
    "/opt/vertex/training/train_full.py",
    "--data-uri",
    a.data_uri,
    "--data-sha256",
    a.data_sha256,
    "--output-uri",
    a.output_uri,
    "--mode",
    a.mode,
    "--gpu-count",
    str(a.gpu_count),
    "--batch-per-gpu",
    str(a.batch_per_gpu),
    "--iterations",
    str(a.iterations),
    "--crop-size",
    str(a.crop_size),
    "--validation-tile",
    str(a.validation_tile),
    "--validation-max",
    str(a.validation_max),
    "--validation-interval",
    str(a.validation_interval),
]
if a.resume_prefix:
    args += ["--resume-prefix", a.resume_prefix]
spec = {
    "serviceAccount": "denoise-vertex@hallowed-pipe-497802-i0.iam.gserviceaccount.com",
    "baseOutputDirectory": {"outputUriPrefix": a.output_uri},
    "workerPoolSpecs": [
        {
            "machineSpec": {
                "machineType": a.machine,
                "acceleratorType": a.accelerator,
                "acceleratorCount": a.gpu_count,
            },
            "replicaCount": "1",
            "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": a.disk_gb},
            "containerSpec": {"imageUri": a.image, "args": args},
        }
    ],
    "scheduling": {"timeout": f"{duration}s", "disableRetries": True, "restartJobOnWorkerRestart": False},
}
with a.output.open("x") as f:
    f.write(json.dumps(spec, indent=2) + "\n")
print(a.output)
