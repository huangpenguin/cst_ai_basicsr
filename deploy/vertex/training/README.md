# Vertex training operations

The 2026-11-18 real eight-step job succeeded; all 15 artifacts passed checksum
verification and model/optimizer state passed independent CPU inspection. See
[acceptance report](../../../docs/deploy/training-acceptance-20261118.md).

Canonical project: `/mnt/home/huang/code/cst_ai/BasicSR`. The CPU UI and training
are separate processes. Uploading data never submits training. Training never
replaces the live inference model. Credentials stay on work1 local disk.

## Reproduce the bounded real-data acceptance run

The recorded specification is
`docs/deploy/training-readiness-20261118/job-spec.json`: one Tokyo T4,
`n1-standard-8`, float32, batch 1, 128×128 crop, 8 optimizer updates,
validation after updates 4 and 8, 150 GiB SSD, 1800-second timeout, retries off.
It uses four real training pairs and one monitoring pair, no test data.
The archive SHA256 and container digest are fixed. A new output prefix is
created for every run. This reproduces the procedure; CUDA results need not
be bit-identical across machines.

Run on work1 as `huang`, with the existing local authorized user ADC:

```bash
cd /mnt/home/huang/code/cst_ai/BasicSR
# Choose a NEW directory; its parent must already exist.
python3 deploy/vertex/training/acceptance.py prepare /home/huang/basicsr-acceptance-manual-01
cat /home/huang/basicsr-acceptance-manual-01/job-spec.json
# This is the explicit paid GPU submission step:
python3 deploy/vertex/training/acceptance.py submit /home/huang/basicsr-acceptance-manual-01
python3 deploy/vertex/training/acceptance.py status /home/huang/basicsr-acceptance-manual-01
# After JOB_STATE_SUCCEEDED, download and verify every artifact checksum:
python3 deploy/vertex/training/acceptance.py fetch /home/huang/basicsr-acceptance-manual-01
```

Optional stop: `python3 deploy/vertex/training/acceptance.py cancel RUN_DIRECTORY`.
A submission marker blocks duplicate submissions. If the CLI response is lost,
find the recorded display name in Vertex before deciding how to recover; do not
blindly resubmit. `prepare` changes only local files. `fetch` requires cloud
success and a new local `artifacts/` directory. If a download is interrupted, rename the incomplete `artifacts/` directory to
a separate backup name before retrying `fetch`; partial files are not accepted
as verified results. Job status, summary, eight actual
optimizer updates, finite losses, changed weights and valid checkpoints are
separate acceptance checks; HTTP success alone is insufficient.

## New or large input datasets

Data contract: a tar with `GT/`, `LQ/`, `train.txt`, `val_monitor.txt`, and
`normalization.json` at its root; flat paired filenames; original uint16
normalization 0..65535. Independent test images are excluded. A new incoming
archive with another schema must be adapted and its scientific pairing reviewed.

```bash
# Read-only inspection; does not hash, archive, upload, or train:
python3 deploy/vertex/training/prepare_data.py --data-root /ABSOLUTE/DATASET
# Explicit packaging into a new file:
python3 deploy/vertex/training/prepare_data.py --data-root /ABSOLUTE/DATASET --archive /LOCAL/dataset.tar
# Review before upload, then upload the approved dataset:
bash deploy/vertex/training/upload-data.sh --check /LOCAL/dataset.tar gs://hallowed-pipe-497802-i0-denoise-vertex/datasets/NEW_VERSION/dataset.tar
bash deploy/vertex/training/upload-data.sh /LOCAL/dataset.tar gs://hallowed-pipe-497802-i0-denoise-vertex/datasets/NEW_VERSION/dataset.tar
```

The uploader uses refreshable user ADC, resumable tracking on local private disk,
SHA256 metadata and generation-match 0. An existing target is rejected. Google
Drive is unnecessary. For a direct GCS upload from another approved workstation,
use `gcloud storage cp --if-generation-match=0`, retain SHA256, and ensure the
runtime service account can read the private object. Never copy credentials to
NFS, Git, image layers or the browser. A resumed transfer is not complete until
its resulting object and checksum are verified.

The existing full dataset contains 6090 training and 20 monitoring pairs,
56,319,719,343 logical bytes; its 2020-pair test set is excluded. Only the approved
five-pair acceptance subset has been uploaded in this session. Split checks in
the packager are filename based; acquisition/adjacent-slice leakage review is
still required for new batches.

## Formal training preparation (manual submission only)

Use the fixed tested image digest in `training-readiness-20261118/image-digests.json`.
The generator defaults to one GPU and batch 1; the former four-V100 batch-16
configuration is not a requirement. Batch 1 changes optimization behavior and
is not equivalent to batch 16. This implementation does not add gradient
accumulation. Review learning rate, training duration and sample exposure before
formal training; 30000 updates at batch 1 do not cover as many samples as the
historical batch 16. The short run proves execution, not model quality.

```bash
python3 deploy/vertex/training/make_job.py \
  --image 'PINNED_TRAINING_IMAGE_FROM_IMAGE_DIGESTS_JSON' \
  --machine n1-standard-8 --accelerator NVIDIA_TESLA_T4 \
  --gpu-count 1 --batch-per-gpu 1 --crop-size 128 \
  --mode full --iterations 30000 \
  --validation-tile 128 --validation-max 20 --validation-interval 2000 \
  --data-uri gs://hallowed-pipe-497802-i0-denoise-vertex/datasets/REVIEWED_VERSION/dataset.tar \
  --data-sha256 ACTUAL_SHA256 \
  --output-uri gs://hallowed-pipe-497802-i0-denoise-vertex/full-training/UNIQUE_RUN_ID \
  --timeout-hours REVIEWED_LIMIT --disk-gb 300 --output /tmp/full-training-job.json
```

Full mode requires at least 1000 updates so the frozen runner saves an optimizer
checkpoint; use acceptance mode for short checks. Generation does not submit. Supply real values and review the JSON. The existing
runtime identity is `denoise-vertex@hallowed-pipe-497802-i0.iam.gserviceaccount.com`;
the operator needs actAs. Input/output access was exercised by the bounded run.
No quota request, formal long job or model promotion is performed automatically.
Tokyo T4 quota is 6; H100 is a conditional option. See the dated quota request
and acceptance report in `docs/deploy/`. GPU family support, available quota and
physical capacity are separate checks. Do not substitute Tokyo serving A100/L4
quota for Custom Training quota.

## Persistence and recovery

The worker rejects nonempty outputs and atomically claims a unique run prefix.
It verifies the archive checksum, rejects unsafe archive paths, and runs the
existing BasicSR training loop (one process for one GPU, torchrun/DDP for multiple
GPUs). Telemetry records real optimizer updates, loss, weight hashes, validation
metrics and PyTorch allocated/reserved GPU memory peaks.

Stable snapshots are uploaded every minute. Content-addressed objects are
written before the manifest commits. Checkpoints, optimizer/scheduler state,
logs, validation images and TensorBoard events are persisted. Three consecutive
periodic persistence failures stop training; final persistence failure fails
the job. Timeout or VM loss may interrupt the last upload, so the latest committed
snapshot is the recovery boundary.

`--resume-prefix gs://.../PREVIOUS_RUN` restores a committed manifest into a NEW
run prefix, verifies every hash and uses the newest optimizer state with a
matching model checkpoint. Dataset, network and training contract must match.
The real acceptance verifies saved optimizer/model files, but crash-and-resume
has not been exercised as a second paid GPU job.

Local I/O tests: `make -f deploy/vertex/Makefile training-io-test`.
Image rebuilds use `Dockerfile`; its default command only prints help. New runtime
changes require a new immutable image digest. See `source-manifest.json` for the
exact files embedded in the accepted image; later operator docs/scripts are
host-side helpers, not changes to that frozen image.
