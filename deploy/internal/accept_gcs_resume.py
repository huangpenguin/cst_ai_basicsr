"""Explicit real GCS transport acceptance, using synthetic TIFF pairs only.

Run inside the portal worker with its existing user ADC. This deliberately
uploads a bounded synthetic archive, loses one acknowledged chunk response,
then resumes the same session. Output is sanitized JSON; session URLs and auth
headers are never logged. No training/serving API is called.
"""

import argparse
import base64
import hashlib
import json
import secrets
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import google.auth
import google_crc32c
import numpy as np
import requests
from google.auth.transport.requests import AuthorizedSession
from PIL import Image

from apps.xray_portal import cloud_sync


def download_check(session, name, expected_sha):
    # Streaming verification deliberately never saves company image bytes.
    endpoint = f"https://storage.googleapis.com/storage/v1/b/{cloud_sync.BUCKET}/o/{quote(name, safe='')}"
    metadata_response = session.get(endpoint, timeout=(15, 60))
    if metadata_response.status_code != 200:
        raise RuntimeError(f"Metadata verification failed (HTTP {metadata_response.status_code}).")
    metadata = metadata_response.json()
    response = session.get(endpoint, params={"alt": "media"}, stream=True, timeout=(15, 180))
    if response.status_code != 200:
        raise RuntimeError(f"Streaming download failed (HTTP {response.status_code}).")
    sha = hashlib.sha256()
    crc = google_crc32c.Checksum()
    size = 0
    try:
        for chunk in response.iter_content(1024 * 1024):
            sha.update(chunk)
            crc.update(chunk)
            size += len(chunk)
    finally:
        response.close()
    actual_sha = sha.hexdigest()
    actual_crc = base64.b64encode(crc.digest()).decode()
    if actual_sha != expected_sha or actual_crc != metadata["crc32c"] or size != int(metadata["size"]):
        raise RuntimeError("Streamed object checksum or size verification failed.")
    return {
        "object": name,
        "http_status": 200,
        "size": size,
        "sha256": actual_sha,
        "crc32c": actual_crc,
        "generation": metadata["generation"],
        "saved_to_disk": False,
        "verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real", action="store_true", help="Explicitly authorize synthetic GCS writes")
    parser.add_argument("--verify-existing-object", help="Optional object name within the fixed private bucket")
    parser.add_argument("--expected-sha256", help="Required known digest for optional streaming verification")
    args = parser.parse_args()
    if not args.run_real:
        parser.error("Use --run-real to explicitly execute the bounded cloud transfer")
    if bool(args.verify_existing_object) != bool(args.expected_sha256):
        parser.error("Provide both optional existing-object name and expected SHA256")
    if args.verify_existing_object and not args.verify_existing_object.startswith("datasets/portal/"):
        parser.error("Existing object must be within datasets/portal/ in the fixed bucket")
    version = "resume-acceptance-" + datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + secrets.token_hex(4)
    trace = []
    state = {"run": 1, "injected": False, "resume_query_offset": None, "resumed_chunk_start": None}

    class ObservedSession:
        def __init__(self, credentials):
            self.session = AuthorizedSession(credentials)

        def close(self):
            self.session.close()

        def request(self, method, url, **kwargs):
            response = self.session.request(method, url, **kwargs)
            content_range = kwargs.get("headers", {}).get("Content-Range")
            record = {"run": state["run"], "method": method, "http_status": response.status_code}
            if content_range:
                record["content_range"] = content_range
            received_range = response.headers.get("Range")
            if received_range:
                record["acknowledged_range"] = received_range
            trace.append(record)
            if state["run"] == 2 and method == "PUT" and content_range:
                if content_range.startswith("bytes */") and received_range and state["resume_query_offset"] is None:
                    state["resume_query_offset"] = int(received_range.rsplit("-", 1)[1]) + 1
                elif not content_range.startswith("bytes */") and state["resumed_chunk_start"] is None:
                    state["resumed_chunk_start"] = int(content_range.split(" ", 1)[1].split("-", 1)[0])
            if (
                not state["injected"]
                and method == "PUT"
                and content_range
                and content_range.startswith("bytes 0-8388607/")
                and response.status_code == 308
            ):
                if received_range != "bytes=0-8388607":
                    raise RuntimeError("Cloud did not acknowledge exactly the first 8 MiB chunk.")
                state["injected"] = True
                response.close()
                raise requests.ConnectionError("Injected lost response after first acknowledged synthetic chunk")
            return response

    with tempfile.TemporaryDirectory(prefix="portal-gcs-resume-") as temporary:
        root = Path(temporary)
        source = root / "source"
        version_dir = root / "version"
        resume = root / "private-resume"
        for folder in (source / "GT", source / "LQ", version_dir):
            folder.mkdir(parents=True)
        rng = np.random.default_rng(20261119)
        names = []
        for index in range(6):
            name = f"synthetic_{index:03d}.tif"
            names.append(name)
            gt = rng.integers(0, 65536, size=(1024, 1024), dtype=np.uint16)
            noisy = np.clip(gt.astype(np.int32) + rng.integers(-64, 65, gt.shape), 0, 65535).astype(np.uint16)
            Image.fromarray(gt).save(source / "GT" / name)
            Image.fromarray(noisy).save(source / "LQ" / name)
        (source / "train.txt").write_text("\n".join(names[:4]) + "\n")
        (source / "val_monitor.txt").write_text("\n".join(names[4:]) + "\n")
        (source / "normalization.json").write_text(json.dumps({"raw_min": 0, "raw_max": 65535}))
        archive = version_dir / "dataset.tar"
        with tarfile.open(archive, "w") as output:
            for file in sorted(source.rglob("*")):
                if file.is_file():
                    info = output.gettarinfo(str(file), arcname=str(file.relative_to(source)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o444
                    with file.open("rb") as data:
                        output.addfile(info, data)
        checksums = cloud_sync._checksums(archive)
        if not 24 * 1024**2 <= checksums["size"] <= 32 * 1024**2:
            raise RuntimeError("Synthetic archive escaped its 24–32 MiB acceptance bound.")
        (version_dir / "manifest.json").write_text(
            json.dumps({"sha256": checksums["sha256"], "synthetic": True, "purpose": "resumable transport acceptance"})
        )
        original_factory = cloud_sync.AuthorizedSession
        cloud_sync.AuthorizedSession = ObservedSession
        try:
            try:
                cloud_sync.sync_version(version_dir, version, resume)
            except RuntimeError as error:
                if not state["injected"] or "retry this sync to resume" not in str(error):
                    raise RuntimeError("Expected injected interruption was not observed.") from None
            else:
                raise RuntimeError("First sync unexpectedly finished without the injected interruption.")
            private_sessions = list(resume.glob("*.json"))
            if len(private_sessions) != 1 or private_sessions[0].stat().st_mode & 0o777 != 0o600:
                raise RuntimeError("Resumable session was not persisted with private mode 0600.")
            state["run"] = 2
            result = cloud_sync.sync_version(version_dir, version, resume)
        finally:
            cloud_sync.AuthorizedSession = original_factory
        if state["resume_query_offset"] != 8 * 1024**2 or state["resumed_chunk_start"] != 8 * 1024**2:
            raise RuntimeError("Retry did not resume from the acknowledged first 8 MiB.")
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"], quota_project_id=cloud_sync.PROJECT
        )
        with AuthorizedSession(credentials) as session:
            verified = download_check(session, f"datasets/portal/{version}/dataset.tar", checksums["sha256"])
            existing = (
                download_check(session, args.verify_existing_object, args.expected_sha256)
                if args.verify_existing_object
                else None
            )
        report = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "version_id": version,
            "bucket": cloud_sync.BUCKET,
            "synthetic_only_upload": True,
            "training_started": False,
            "archive": checksums,
            "lost_response_after_acknowledged_bytes": 8 * 1024**2,
            "private_session_mode": "0600",
            "resume_query_offset": state["resume_query_offset"],
            "resumed_chunk_start": state["resumed_chunk_start"],
            "uploaded_generation": result["generation"],
            "commit_manifest_uri": result["manifest_uri"],
            "synthetic_download": verified,
            "existing_reviewed_dataset_download": existing,
            "trace": trace,
            "passed": True,
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Bound failure output so requests exceptions cannot expose secret URLs.
    try:
        main()
    except Exception as error:
        print(
            json.dumps({"passed": False, "error_type": type(error).__name__, "error": "GCS resume acceptance failed"})
        )
        raise SystemExit(1) from None
