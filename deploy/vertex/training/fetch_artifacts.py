"""Download a committed training manifest and verify every referenced SHA256.

Run using the bundled Cloud SDK Python with local user ADC mounted at /adc.json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse

sys.path.insert(0, "/usr/lib/google-cloud-sdk/lib/third_party")
import google.auth
from google.auth.transport.requests import AuthorizedSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parsed = urlparse(a.prefix)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Expected gs://bucket/run-prefix")
    creds, _ = google.auth.load_credentials_from_file(
        "/adc.json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(creds)
    base = f"https://storage.googleapis.com/storage/v1/b/{quote(parsed.netloc, safe='')}/o/"
    prefix = parsed.path.strip("/")

    def fetch(name, path):
        url = base + quote(prefix + "/" + name, safe="")
        digest = hashlib.sha256()
        with session.get(url, params={"alt": "media"}, stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with path.open("xb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
                    digest.update(chunk)
        return digest.hexdigest()

    a.output.mkdir(parents=True, exist_ok=False)
    manifest = a.output / "artifact-manifest.json"
    fetch("artifact-manifest.json", manifest)
    files = json.loads(manifest.read_text())["files"]
    checks = []
    for relative, item in sorted(files.items()):
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("Unsafe artifact path")
        if item["object"] != "objects/" + item["sha256"]:
            raise ValueError("Invalid immutable object reference")
        target = a.output.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = fetch(item["object"], target)
        if digest != item["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
        checks.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest})
    summary = json.loads((a.output / "summary.json").read_text())
    if summary.get("status") != "SUCCEEDED" or summary.get("exit_code") != 0:
        raise ValueError("Downloaded training summary is not successful")
    report = {"status": "PASS", "prefix": a.prefix, "verified_files": checks, "summary_status": summary["status"]}
    (a.output / "download-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "verified_file_count": len(checks), "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
