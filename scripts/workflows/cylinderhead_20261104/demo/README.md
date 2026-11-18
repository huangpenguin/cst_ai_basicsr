> **Retired local GPU deployment (2026-11-18).** The active CPU → Vertex UI is http://192.168.3.11:7861, managed from `/mnt/home/huang/code/cst_ai/BasicSR/deploy/vertex/`. The instructions below are historical. Metadata and image details now live in the active UI; keep these sources for shared model components and reproducibility.

# BasicSR Demo on jil-08-01

Code and weights are read from NFS. Python dependencies are installed in the
local Docker image, at `/opt/basicsr-venv`; no host venv or shared NFS venv is used.
The manifest and lockfile in this directory are snapshots from commit `41d7bfd`.

```sh
cd /mnt/home/huang/code/cst_ai/BasicSR/scripts/workflows/cylinderhead_20261104/demo
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 -f
```

Open http://192.168.3.11:7860 on the internal network. The service restarts
automatically after reboot unless explicitly stopped. GPU model caches expire
after 600 idle seconds, following the application default.

Stop with `docker compose down`. After updating NFS source, use
`docker compose restart demo`. If dependencies change, copy the updated
`pyproject.toml` and `uv.lock` from NFS into this directory and rebuild.
The base image `ai-coding-rules-mlops:smoke` already exists on this host.

Verification (checks all checkpoint paths, then SwinIR, TSFDAN and LoRA inference):

```sh
docker exec -i denoise-demo /opt/basicsr-venv/bin/python - < smoke.py
```

The NFS repository and experiment tree are mounted read-only. Uploaded files
and generated downloads reside in the container's temporary storage and are
discarded when the container is recreated. Container logs rotate at 3 × 10 MB.
The host NVIDIA driver and NVIDIA Container Toolkit must work before starting.
The service retains UID/GID 1003 and adds host GPU group 1001 (`vglusers`),
because this host's NVIDIA device nodes have mode 0660.

This deployment does not require gpu01's old-path symlink or `BasicSR.bak`.
Remove those on gpu01 only after checking that its development sessions and
scripts use the NFS path and that the backup has no unique files.

## Validation on 2026-10-27

- Host and GPU container: NVIDIA driver 595.91.07, RTX 2080 Ti.
- Container PyTorch 2.6.0+cu124 detects CUDA.
- All 10 registered checkpoint paths exist.
- `swinir_mix`, `tsfdan_preft`, and `tsfdan_lora_mix` pass 64×64 uint16 TIFF
  inference and output-file validation through the application's callback.
- HTTP GET to `http://192.168.3.11:7860/` returns 200; Docker reports healthy.
- Restart policy is configured as `unless-stopped`; a further host reboot
  has not been performed as part of validation.
