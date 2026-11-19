"""Single durable CPU worker; never submits a GPU job."""

import fcntl
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

from . import store
from .pipeline import export_dataset, validate_batch


def run_job(job):
    batch_id, kind = job["batch_id"], job["kind"]
    store.check_storage()
    with store.connect() as db:
        batch = store.get_batch(db, batch_id)
    root = store.DATA / batch_id
    if kind == "validate":
        active = [f for f in batch["files"] if f["status"] != "excluded"]
        if not active:
            raise ValueError("No active uploads remain; add TIFF or ZIP files")
        for f in active:
            path = root / "original" / f["id"]
            if path.stat().st_size != f["size"]:
                raise ValueError("Received size differs from the registered upload: " + f["filename"])
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != f["fingerprint"]:
                f["status"] = "invalid"
                with store.transaction() as db:
                    store.put_batch(db, batch)
                raise ValueError("Received SHA256 differs from the selected source: " + f["filename"])
            f["status"], f["sha256"] = "received", digest
        report = validate_batch(root, active)
        store.save_json(root / "validation-report.json", report)
        batch["report"], batch["decisions"], batch["status"] = report, None, "awaiting_review"
        store.event(batch, "Source checksums verified. Pairing candidates require reviewer approval.")
    elif kind == "freeze":
        version_id = batch["pending_version"]
        version_dir = root / "versions" / version_id
        if version_dir.exists():
            # A completed atomic export may precede the DB commit if the worker stopped.
            manifest = json.loads((version_dir / "manifest.json").read_text())
            with (version_dir / "dataset.tar").open("rb") as archive:
                if hashlib.file_digest(archive, "sha256").hexdigest() != manifest["sha256"]:
                    raise ValueError("Recovered frozen archive checksum failed")
        else:
            manifest = export_dataset(root, batch["report"], batch["decisions"], version_dir)
        batch["version"] = {"id": version_id, "manifest": manifest}
        batch["status"] = "frozen"
        store.event(
            batch, "Reviewed dataset version frozen; cloud synchronization requires a separate action.", "reviewer"
        )
    elif kind == "sync":
        from .cloud_sync import sync_version

        version_id = batch["version"]["id"]
        result = sync_version(
            root / "versions" / version_id, version_id, Path(os.environ.get("PORTAL_CLOUD_STATE_ROOT", "/cloud-state"))
        )
        batch["version"]["cloud"] = result
        batch["status"] = "cloud_ready"
        store.event(batch, "Approved training version copied to private GCS and verified. No training submitted.")
    else:
        raise ValueError("Unknown worker task")
    with store.transaction() as db:
        store.put_batch(db, batch)
        db.execute("UPDATE jobs SET status='succeeded' WHERE id=?", (job["id"],))


def main():
    store.initialize()
    with (store.STATE / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with store.connect() as db:
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
        while True:
            with store.transaction() as db:
                job = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if job:
                    job = dict(job)
                    db.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
            if not job:
                time.sleep(2)
                continue
            process = multiprocessing.Process(target=execute_job, args=(job,))
            process.start()
            process.join(timeout=7200)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
                mark_failure(job, "Task exceeded its two-hour processing limit; review the batch before retrying.")
            elif process.exitcode != 0:
                mark_failure(job, "Worker stopped unexpectedly; retry this task after checking server resources.")


def mark_failure(job, message):
    with store.transaction() as db:
        current = db.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if current and current["status"] == "succeeded":
            return
        batch = store.get_batch(db, job["batch_id"])
        batch["status"] = {"validate": "validation_failed", "freeze": "freeze_failed", "sync": "sync_failed"}[
            job["kind"]
        ]
        batch["error"] = message
        store.event(batch, message)
        store.put_batch(db, batch)
        db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (message, job["id"]))


def execute_job(job):
    try:
        run_job(job)
    except Exception as exc:
        # Never put a secret resumable session URL into logs or browser errors.
        message = (
            "Cloud synchronization failed; retry after checking connectivity and server credentials."
            if job["kind"] == "sync"
            else str(exc)
        )
        mark_failure(job, message)
        print(f"Task {job['id']} ({job['kind']}) failed: {type(exc).__name__}", flush=True)


if __name__ == "__main__":
    main()
