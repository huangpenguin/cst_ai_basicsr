# work1 X-ray intake portal

Application code is baked into the locally built Docker image; production does not
bind-mount mutable source files. Redeploy through the launcher after a reviewed code change.

CPU-only portal on `http://192.168.3.11:7862`; the existing Vertex inference UI
remains at `http://192.168.3.11:7861`. All code belongs to BasicSR. No training
submission or serving-model publication occurs in this application.

```bash
cd /mnt/home/huang/code/cst_ai/BasicSR
bash deploy/internal/start-portal.sh --check
bash deploy/internal/start-portal.sh
```

The launcher verifies the actual NFS sources, writeability, local state/config
filesystem and LAN IP **before** creating any directory. Run as work1 `huang`
(UID 1003). It generates a private admin password and session/hook secrets only
on first launch. Read the password locally from
`/home/huang/.config/basicsr-xray-portal/admin-password.txt`; do not paste it into
Git or chat. This HTTP pilot has no transport encryption.

## Storage and recovery

- Original file bytes: real NFS `/mnt/data/cst_ai/intake`, mounted `/data`.
- SQLite and tus metadata/locks: local disk
  `/home/huang/.local/state/basicsr-xray-portal`, mounted `/state`.
- Secret GCS resumable sessions: separate local directory
  `/home/huang/.local/state/basicsr-xray-portal-cloud`, mounted `/cloud-state`
  **only in the worker**; the API and upload service cannot read it.
- Secrets: `/home/huang/.config/basicsr-xray-portal`, local only.
- Existing authorized-user ADC is mounted read-only **only** in the worker.
- API and uploader receive no ADC, Docker socket or NVIDIA devices. The runtime
  is explicitly `runc`, with `NVIDIA_VISIBLE_DEVICES=void`.

The proxy streams chunks without buffering an entire upload. tus protocol
metadata stays local while the trusted pre-create hook selects the NFS binary
path. Every upload request is authorized by the API, including resumed PATCH
and HEAD requests. Hook routes are not exposed on the LAN.

Restart services without deleting state or uploads:

```bash
docker compose -f deploy/internal/compose.yaml restart
```

A paused browser upload can resume after reselecting the **same** source file.
Uploaded bytes are not yet approved data. CPU validation must finish, a reviewer
must confirm GT/LQ mapping and split, and a frozen version must exist before a
cloud sync can be queued. Network failures leave resumable state on local disk;
retry the same immutable version rather than inventing a new version for each
attempt. Never copy cloud session URLs to a browser or log.

GCS targets are allowlisted to
`gs://hallowed-pipe-497802-i0-denoise-vertex/datasets/portal/<version>/`.
Only `dataset.tar` and explicit provenance JSON files are uploaded; raw archives
and test images are not recursively copied. Every write uses generation-match 0;
retry accepts an existing object only if size, CRC32C and SHA256 match. A
`cloud-manifest.json` object is published last to commit the complete version.
This operation does not call Vertex training APIs.

## Operations

Use `docker compose -f deploy/internal/compose.yaml ps` for status and
`logs --tail=100 api worker tusd proxy` for sanitized errors. Avoid `compose
config` without `--quiet`, environment dumps, or printing local resume files:
they contain secrets. Stop only this project with `docker compose ... stop`;
never run broad Docker cleanup commands. Retain NFS files and local state for
recovery. Before opening access beyond the pilot LAN, add trusted HTTPS.

## Reproduce the bounded GCS resume check

This explicitly uploads one synthetic 24–32 MiB TIFF dataset to a unique private
prefix. It simulates losing the response after GCS acknowledges the first 8 MiB,
then verifies that the next sync resumes at that offset. Finally it streams the
archive back and checks SHA256, CRC32C and size. It never starts training.

```bash
docker exec -i basicsr-xray-portal-worker-1 python - --run-real \
  < deploy/internal/accept_gcs_resume.py > /tmp/portal-gcs-resume-result.json
```

The JSON output contains only sanitized statuses, ranges, hashes and object
identifiers. The script uses the worker's existing ADC without printing or
copying it. Successful synthetic cloud objects are retained as evidence;
local temporary data and private resumable sessions are removed on exit.
