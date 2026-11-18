# work1 migration — 2026-11-18 JST

Status: **PASS — frontend migration, real Vertex predictions, old entrance cutover and fresh idle-zero evidence**.

New entrance: **[http://192.168.3.11:7861](http://192.168.3.11:7861)**.
Only the CPU frontend moved; the existing Tokyo Vertex GPU inference resources remain in use.
Formal training has not started; its prepared image and input/management tools are deployed separately.

## Host, NFS and identity

- Saved host work1, actual OS hostname `jil-08-01`, LAN IPv4 `192.168.3.11`.
- `/mnt/home` is actual `nfs4`, source `192.168.3.14:/mnt/storage/home`, `rw`.
  Actual create/write/fsync/unlink in the existing checkout succeeded; the probe was removed.
- Reused `/mnt/home/huang/code/cst_ai/BasicSR`. No checkout copy, force reset, clean,
  code/weight/data move, host Torch/CUDA installation or training launch.
- `/mnt/data` is NFS from `192.168.3.14:/mnt/storage/data`.
- User `huang`, UID/GID `1003:1003`. Legal `authorized_user` ADC subject is
  `huang.pengbin@johnan.com`, quota project `hallowed-pipe-497802-i0`.
- ADC was authorized through a browser flow initiated on work1, stored only on
  work1 local ext4 at `/home/huang/.config/basicsr-vertex/application_default_credentials.json`,
  mode 0600 under a 0700 directory. Credential/token contents are absent from NFS,
  Git, images and these records. See [nonsecret metadata](work1-20261118/authentication-metadata.json).
- Existing checkout HEAD is `fcca7f5a210659df4cd8324af09faee611afe22f`.
  Preexisting README edits, handoff files and startup script were preserved.
  New source changes remain in the working tree; no commit or Git push was made.

## Exact deployment

- Container `denoise-vertex-ui`, restart `unless-stopped`, user `1003:1003`.
- Registry image is frozen at
  `asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex@sha256:fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f`.
- Actual image ID `sha256:c5a18e2e2fdda2d1c0b4795c0aba58a11581d268edc4bcf4360853ccbb31efc4`, Docker-reported size 28423078506 bytes.
- Pulled from Artifact Registry using work1's new authorized ADC. A prior SSH
  streaming-reuse attempt was cancelled before import. No image tar was placed on NFS.
  Temporary registry login directories were removed after pull/push.
- `--check` passed before startup: [startup evidence](work1-20261118/start-check.txt).
- Only host `192.168.3.11:7861` is published to container 7860; `share=False`.
  Source `internal_ui.py` and local ADC are read-only mounts.
- `NVIDIA_VISIBLE_DEVICES=void`, empty `CUDA_VISIBLE_DEVICES`, DeviceRequests null,
  actual `/dev/nvidia*` empty. See [runtime proof](work1-20261118/runtime.json) and
  [final configuration](work1-20261118/frontend-final-config.json).
- English interface, 11 original preset entries; all 10 unavailable presets were
  verified to return a disabled Denoise button through the raw Gradio event response.
  Only Cylinderhead final can run.

The project remains `hallowed-pipe-497802-i0`, region `asia-northeast1`,
Endpoint `8911645097217490944`, Model `6657078912276758528`,
DeployedModel `1239217774224474112`. No inference deployment, endpoint, model,
VM, Cloud Run, DNS, IAM or quota was rebuilt or changed.

## Real page → Vertex → download acceptance

| Case | Input/output size | Download | End-to-end | Cloud handler | Result |
|---|---|---|---:|---:|---|
| Cold small PNG | 64×64 | uint16 TIFF, 8314 bytes | 231.861 s | 1.835 s | PASS |
| Representative TIFF | 1518×1518 | uint16 TIFF, 4608770 bytes | 51.755 s | 47.361 s | PASS |

The returned preset is `swinir_cylinderhead_mean3_20261104_173035` and device `cuda`.
The full-image source was read-only mounted from
`/mnt/home/huang/cst_ai/experiments/cylinderhead_300kv_mean3_20261104_173035_JST/resource_review/benchmark_input/input_validation.tif`.
Records and retained downloads: [small](work1-20261118/small.json),
[full](work1-20261118/full.json), `output-small.tif`, `output-full.tif` in the
same evidence directory (binary images are ignored by Git).

An initial UI-only test stopped before any prediction because GradioClient
omits Button updates from its Python return value. The acceptance tool was
corrected to inspect the raw event, and the initial failure is retained as
[small-client-check-failed.json](work1-20261118/small-client-check-failed.json).
There was one real small-image flow and one real full-image flow, with the
existing frontend's automatic cold-start retries. No extra cold or stress test.

Before prediction, the control plane still reported the historical `System error`.
During this run it returned `availableReplicaCount: 1`, and both actual predictions
succeeded without cloud rebuild. The scoped error-log query returned no ERROR
entries. See [before](work1-20261118/cloud-before.json),
[start](work1-20261118/cloud-at-start.json), and
[warm status](work1-20261118/cloud-during-test-2.json). The final idle sample records the current control-plane status as well;
its status has a last-update timestamp and no error message.

## Cutover and network verification

After both real cases passed:

- Work1's former local GPU `denoise-demo` was rechecked for its image, Compose
  project `basicsr-demo`, command and shared checkout mount, then stopped at
  `2026-11-18T05:14:11.256411+00:00`.
- GPU01's former CPU/Vertex `denoise-vertex-ui` was rechecked over verified-key
  SSH for the exact frozen image ID, same project/Endpoint and source mount,
  then stopped at `2026-11-18T05:14:12.862246+00:00`.
- Both old containers and images are retained. Other services were untouched.
  Old 7860 listeners are closed. See [work1 cutover](work1-20261118/old-work1-cutover.json)
  and [gpu01 cutover](work1-20261118/old-gpu01-cutover.json).
- A request from the separate physical host gpu01 reached work1's new URL with
  HTTP 200 and 11 English presets: [cross-host test](work1-20261118/cross-host.json).
  The Mac browser was not operated or separately verified.

Rollback: stop work1 `denoise-vertex-ui`, recheck the retained gpu01 configuration,
then `docker start denoise-vertex-ui` on gpu01. The stopped work1 local GPU demo
also remains available as an explicitly chosen rollback; it is a different compute path.

## Fresh idle-zero evidence

Fresh Monitoring evidence: replica 1 at `2026-11-18T05:17:14.040000+00:00` → replica 0 at `2026-11-18T05:18:14.040000+00:00`. The zero sample is 370.562 seconds after the last successful full-image request. See [idle-final.json](work1-20261118/idle-final.json) and its raw sample reference. No further test prediction was sent.

The existing v1beta1 scale-to-zero periods remain 300 seconds, max replicas 1.
Read-only control-plane/Monitoring queries do not send prediction traffic.

## Project management and directory cleanup

Use `make -f deploy/vertex/Makefile ui-check`, `ui-start`, `ui-status`, `ui-logs`
from the canonical checkout. `ui-start` refuses to overwrite an existing container.
A stopped valid container can be resumed with `docker start denoise-vertex-ui`.
Host-only configuration is `/home/huang/.config/basicsr-vertex/ui.env` (0600).
`cloud-cli.sh` uses native refreshable ADC in an ephemeral SDK container;
`read-cloud.sh` reads identity, Endpoint and actual replica samples.

Four local compatibility links in `/home/huang/scripts` were removed only after
verifying their NFS targets: `cylinderhead-import-20261102`,
`cylinderhead-resource-review_20261104`, `cylinderhead-training_20261104_173035_JST`,
and `basicsr-demo`. No cron/systemd references were found for the three completed
historical entries. Their NFS archive, canonical workflow code, weights and datasets
were not removed or moved. The local README points to the canonical project.

## Full-training deployment preparation

Prepared private training image:

```text
asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex-training@sha256:a8722a628ec5bad342844da39da1060078def7be46ced20865fc351ce4b5ed9a
```

It overlays only about 35 kB of tools/configuration onto the frozen base and
has a default help command. The original inference image/resources are unchanged.
The image passed a no-local-GPU dry-run and four container I/O tests: split leakage
and missing pairs, unsafe tar members, immutable artifact snapshots and corrupt
restore rejection, and existing/concurrent output-prefix rejection. Runtime source
hashes match the recorded build source. See [container check](work1-20261118/training-container-check.json),
[source hashes](work1-20261118/training-source-manifest.json),
[container hashes](work1-20261118/training-container-source.json), and
[published digest](work1-20261118/training-published-digests.json).

The [training guide](../../deploy/vertex/training/README.md) covers unchanged
split validation/packaging, direct internal-server upload, GCS SHA256-verified
archive intake, original 4-GPU/batch-4/30000-iteration configuration, torchrun,
periodic model/optimizer/log/validation/TensorBoard persistence, immutable
committed manifests, matching-state recovery and a non-submitting job generator.
Uploads use refreshable user ADC, private local resumable trackers and a generation-0
precondition. Upload dry-run and read-only bucket access passed; no data upload occurred.

Existing dataset preflight: 6090 train pairs, 20 monitor pairs, 2020 held-out test pairs,
12223 required train/monitor/metadata files, 56319719343 logical bytes. The test split
is excluded from training packages. New incoming archives must meet the documented
pairing/split contract rather than being automatically resplit.

The existing runtime service account
`denoise-vertex@hallowed-pipe-497802-i0.iam.gserviceaccount.com` already has
`roles/storage.objectUser` on the existing private Tokyo bucket. No IAM change.
See [bucket role](work1-20261118/training-bucket-roles.json).

**Still incomplete for full training:** Tokyo has T4 training quota 6, while A100/V100
training quota buckets show no nonzero effective allowance. Serving GPU quota is
separate. The original per-GPU workload used about 25.5 GiB, so the T4 smoke
hardware cannot silently substitute for the original batch. Compatible four-GPU
quota/capacity, an approved/versioned data archive and SHA256, and a real cloud
training/recovery acceptance run remain prerequisites. No quota increase, full-data
upload, CustomJob or paid training run was initiated. This preparation is not a
claim of completed formal training.
See [Tokyo GPU quota evidence](work1-20261118/training-gpu-quotas.json).
