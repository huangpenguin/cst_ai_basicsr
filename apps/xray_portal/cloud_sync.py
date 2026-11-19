"""Resumable, create-only publication of reviewed portal datasets to private GCS.

Session URLs are bearer credentials: persist only on private local disk and never
include HTTP exception strings, headers, or response bodies in application errors.
"""

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import google.auth
import google_crc32c
import requests
from google.auth.transport.requests import AuthorizedSession

BUCKET = "hallowed-pipe-497802-i0-denoise-vertex"
PROJECT = "hallowed-pipe-497802-i0"
CHUNK = 8 * 1024 * 1024
FILES = ("dataset.tar", "manifest.json", "pairs_manifest.json", "split_manifest.json", "normalization.json")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".partial")
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.flush()
        os.fsync(output.fileno())
    tmp.replace(path)


def _checksums(path):
    sha = hashlib.sha256()
    crc = google_crc32c.Checksum()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            sha.update(chunk)
            crc.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": sha.hexdigest(), "crc32c": base64.b64encode(crc.digest()).decode()}


def _matching(metadata, expected):
    return (
        int(metadata.get("size", -1)) == expected["size"]
        and metadata.get("crc32c") == expected["crc32c"]
        and metadata.get("metadata", {}).get("sha256") == expected["sha256"]
    )


def _request(session, method, url, **kwargs):
    try:
        return session.request(method, url, timeout=(15, 180), **kwargs)
    except (requests.RequestException, google.auth.exceptions.GoogleAuthError):
        # requests exception strings can contain the secret resumable session URL.
        raise RuntimeError("Cloud transport or authentication failed; retry this sync to resume.") from None


def _object(session, name):
    response = _request(
        session, "GET", f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{quote(name, safe='')}"
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"Cloud object verification failed (HTTP {response.status_code}).")
    return response.json()


def _upload(session, path, name, state_dir):
    expected = _checksums(path)
    existing = _object(session, name)
    if existing:
        if not _matching(existing, expected):
            raise RuntimeError("Cloud object already exists with different content; a new dataset version is required.")
        return {"uri": f"gs://{BUCKET}/{name}", **expected, "generation": existing["generation"]}
    state_path = state_dir / (hashlib.sha256(name.encode()).hexdigest() + ".json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and (state.get("name") != name or state.get("checksums") != expected):
        raise RuntimeError("Local resumable state does not match this immutable dataset.")
    url = state.get("session_url")
    if url and (urlparse(url).scheme != "https" or urlparse(url).hostname != "storage.googleapis.com"):
        raise RuntimeError("Invalid private cloud resume state.")
    for attempt in range(4):
        if not url:
            response = _request(
                session,
                "POST",
                f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o",
                params={"uploadType": "resumable", "ifGenerationMatch": "0"},
                headers={
                    "X-Upload-Content-Type": "application/octet-stream",
                    "X-Upload-Content-Length": str(expected["size"]),
                },
                json={"name": name, "metadata": {"sha256": expected["sha256"]}, "crc32c": expected["crc32c"]},
            )
            if response.status_code not in (200, 201):
                raise RuntimeError(f"Cloud upload initialization failed (HTTP {response.status_code}).")
            url = response.headers.get("Location")
            if not url or urlparse(url).scheme != "https" or urlparse(url).hostname != "storage.googleapis.com":
                raise RuntimeError("Cloud upload returned an invalid session endpoint.")
            _save(state_path, {"name": name, "checksums": expected, "session_url": url})
        response = _request(
            session,
            "PUT",
            url,
            headers={"Content-Length": "0", "Content-Range": f"bytes */{expected['size']}"},
            data=b"",
        )
        if response.status_code in (404, 410):
            url = None
            continue
        if response.status_code in (200, 201):
            break
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2**attempt, 8))
            continue
        if response.status_code != 308:
            raise RuntimeError(f"Cloud resume query failed (HTTP {response.status_code}).")
        received = response.headers.get("Range", "")
        offset = int(received.rsplit("-", 1)[1]) + 1 if received else 0
        if not 0 <= offset <= expected["size"]:
            raise RuntimeError("Cloud resume offset is outside the file.")
        with path.open("rb") as source:
            source.seek(offset)
            while offset < expected["size"]:
                chunk = source.read(min(CHUNK, expected["size"] - offset))
                if not chunk:
                    raise RuntimeError("Frozen dataset changed while uploading.")
                end = offset + len(chunk) - 1
                response = _request(
                    session,
                    "PUT",
                    url,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end}/{expected['size']}",
                    },
                    data=chunk,
                )
                if response.status_code in (200, 201):
                    offset = expected["size"]
                    break
                if response.status_code == 308:
                    received = response.headers.get("Range", "")
                    confirmed = int(received.rsplit("-", 1)[1]) + 1 if received else 0
                    if confirmed != end + 1:
                        raise RuntimeError("Cloud confirmed an unexpected upload offset; retry to reconcile.")
                    offset = confirmed
                elif response.status_code in (404, 410):
                    url = None
                    break
                elif response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2**attempt, 8))
                    break
                else:
                    raise RuntimeError(f"Cloud chunk upload failed (HTTP {response.status_code}).")
            if offset == expected["size"]:
                break
    metadata = _object(session, name)
    if not metadata or not _matching(metadata, expected):
        raise RuntimeError("Cloud sync remains incomplete or checksum verification failed; retry the same version.")
    state_path.unlink(missing_ok=True)
    return {"uri": f"gs://{BUCKET}/{name}", **expected, "generation": metadata["generation"]}


def sync_version(version_dir: Path, version_id: str, state_dir: Path) -> dict:
    """Publish training archive and reviewed provenance, then the commit manifest.

    Caller must hold the durable job lease and enforce reviewer authorization.
    Only allowlisted files are sent. No training API is invoked.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", version_id):
        raise ValueError("Invalid immutable dataset version ID")
    version_dir = version_dir.resolve(strict=True)
    paths = []
    for filename in FILES:
        path = version_dir / filename
        if path.is_symlink():
            raise ValueError("Publication files cannot be symbolic links")
        if path.exists():
            if not path.is_file():
                raise ValueError("Publication requires regular files")
            paths.append(path)
    if not (version_dir / "dataset.tar").is_file() or not (version_dir / "manifest.json").is_file():
        raise ValueError("Frozen dataset archive and provenance manifest are required")
    manifest = json.loads((version_dir / "manifest.json").read_text())
    archive_hash = _checksums(version_dir / "dataset.tar")["sha256"]
    if manifest.get("sha256") != archive_hash:
        raise ValueError("Frozen archive no longer matches its reviewed manifest")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"], quota_project_id=PROJECT
        )
    except google.auth.exceptions.GoogleAuthError:
        raise RuntimeError(
            "Cloud credentials unavailable; ask the administrator to restore the local user ADC."
        ) from None
    session = AuthorizedSession(credentials)
    try:
        records = {}
        for path in paths:
            records[path.name] = _upload(session, path, f"datasets/portal/{version_id}/{path.name}", state_dir)
        commit = {"schema_version": 1, "version_id": version_id, "files": records, "training_started": False}
        commit_path = state_dir / f"{version_id}-cloud-manifest.json"
        _save(commit_path, commit)
        commit_record = _upload(session, commit_path, f"datasets/portal/{version_id}/cloud-manifest.json", state_dir)
        return {
            "version_id": version_id,
            **records["dataset.tar"],
            "manifest_uri": commit_record["uri"],
            "files": records,
        }
    finally:
        session.close()
