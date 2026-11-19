"""Local durable state. Image bytes live on the verified NFS mount."""

import contextlib
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import time
from pathlib import Path

DATA = Path(os.environ.get("PORTAL_DATA_ROOT", "/data"))
STATE = Path(os.environ.get("PORTAL_STATE_ROOT", "/state"))
MAX_FILE = 32 * 1024**3
MAX_BATCH = 64 * 1024**3
MAX_EXPANDED = 128 * 1024**3
BUSY = {"validating", "freezing", "syncing"}


def now():
    return time.time()


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def connect():
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = sqlite3.connect(STATE / "portal.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY, access_hash TEXT NOT NULL, doc TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, doc TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, kind TEXT NOT NULL,
          status TEXT NOT NULL, created REAL NOT NULL, error TEXT);
        CREATE TABLE IF NOT EXISTS attempts(key TEXT NOT NULL, created REAL NOT NULL);
        """)


@contextlib.contextmanager
def transaction():
    db = connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def get_batch(db, batch_id):
    row = db.execute("SELECT doc FROM batches WHERE id=?", (batch_id,)).fetchone()
    if row is None:
        raise KeyError("Batch not found")
    return json.loads(row["doc"])


def put_batch(db, batch):
    batch["updated_at"] = now()
    db.execute("UPDATE batches SET doc=? WHERE id=?", (json.dumps(batch), batch["id"]))


def event(batch, message, actor="system"):
    batch.setdefault("events", []).append({"time": now(), "actor": actor, "message": message})


def enqueue(db, batch, kind):
    active = db.execute(
        "SELECT id FROM jobs WHERE batch_id=? AND status IN ('queued','running')", (batch["id"],)
    ).fetchone()
    if active:
        raise ValueError("A task is already active for this batch")
    job_id = secrets.token_hex(16)
    db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,NULL)", (job_id, batch["id"], kind, "queued", now()))
    batch["status"] = {"validate": "validating", "freeze": "freezing", "sync": "syncing"}[kind]
    batch["error"] = None
    event(batch, f"{kind.title()} requested", "reviewer" if kind != "validate" else "uploader")
    put_batch(db, batch)
    return job_id


def check_storage(required=0):
    """Never mkdir an unmounted lookalike. Bind mounts retain their NFS source/type."""
    expected = os.environ.get("PORTAL_NFS_SOURCE", "192.168.3.14:/mnt/storage/data")
    root = str(DATA.resolve())
    found = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, fs = before.split(), after.split()
        mount = fields[4].replace("\\040", " ")
        if root == mount or root.startswith(mount.rstrip("/") + "/"):
            if found is None or len(mount) > len(found[0]):
                found = (mount, fs[0], fs[1], fields[5])
    if (
        not found
        or found[1] not in ("nfs", "nfs4")
        or found[2] not in (expected, expected + "/cst_ai/intake")
        or "rw" not in found[3].split(",")
    ):
        raise RuntimeError("Company NFS is not mounted read-write at the configured data root")
    if not DATA.is_dir() or not os.access(DATA, os.W_OK):
        raise RuntimeError("Company storage is not writable")
    if shutil.disk_usage(DATA).free < required + 5 * 1024**3:
        raise RuntimeError("Insufficient company storage; upload is paused")
    return {"source": expected, "free_bytes": shutil.disk_usage(DATA).free}


def save_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as out:
        json.dump(data, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)
