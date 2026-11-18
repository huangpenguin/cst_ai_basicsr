"""Download only the approved ZIP manifest and verify it against Google Drive."""

import collections
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
DEST = Path((HERE.parent / "config" / "destination.txt").read_text().strip())
RCLONE = "/home/huang/scripts/tools/rclone/rclone"
CONFIG = "/home/huang/.config/rclone/rclone.conf"
REMOTE = "cylinderhead_ro:"
MANIFEST = DEST / "source_manifest.json"
STATUS = DEST / "download_status.json"


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def status(phase, **extra):
    save(STATUS, {"updated_at": now(), "phase": phase, **extra})
    print(phase, extra, flush=True)


def main():
    mount = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(DEST)], text=True).strip()
    if mount not in {"nfs", "nfs4"}:
        raise RuntimeError(f"Destination must be NFS, found {mount!r}")
    manifest = json.loads(MANIFEST.read_text())
    remote = json.loads(
        subprocess.check_output([RCLONE, "lsjson", REMOTE, "--files-only", "--hash", "--config", CONFIG], text=True)
    )
    by_name = {entry["Name"]: entry for entry in remote}
    for entry in manifest["files"]:
        actual = by_name[entry["name"]]
        if actual["ID"] != entry["id"] or actual["Size"] != entry["size_bytes"]:
            raise RuntimeError(f"Source changed: {entry['name']}")
        if not actual.get("Hashes", {}).get("md5"):
            raise RuntimeError(f"No source MD5: {entry['name']}")
    selected = [by_name[entry["name"]] for entry in manifest["files"]]
    save(DEST / "remote_checksums.json", selected)
    filelist = DEST / "files.txt"
    filelist.write_text("".join(entry["name"] + "\n" for entry in manifest["files"]))
    common = [
        "--config",
        CONFIG,
        "--files-from-raw",
        str(filelist),
        "--checkers",
        "4",
        "--stats",
        "30s",
        "--stats-one-line-date",
        "--log-level",
        "INFO",
    ]
    status("downloading", files=len(selected), total_bytes=manifest["total_bytes"])
    subprocess.run(
        [
            RCLONE,
            "copy",
            REMOTE,
            str(DEST / "archives"),
            "--checksum",
            "--transfers",
            "3",
            "--multi-thread-streams",
            "2",
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--log-file",
            str(DEST / "download.log"),
            *common,
        ],
        check=True,
    )
    status("verifying_checksums")
    subprocess.run(
        [
            RCLONE,
            "check",
            REMOTE,
            str(DEST / "archives"),
            "--one-way",
            "--log-file",
            str(DEST / "verification.log"),
            *common,
        ],
        check=True,
    )
    inventory = []
    for entry in manifest["files"]:
        path = DEST / "archives" / entry["name"]
        if path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"Size mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            files = [f for f in archive.infolist() if not f.is_dir()]
            suffixes = collections.Counter(Path(f.filename).suffix.lower() for f in files)
            inventory.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "file_count": len(files),
                    "extensions": dict(suffixes),
                    "uncompressed_bytes": sum(f.file_size for f in files),
                    "sample_paths": [f.filename for f in files[:3]],
                }
            )
    save(DEST / "zip_inventory.json", inventory)
    manifest.update(status="downloaded_and_checksum_verified", completed_at=now())
    save(MANIFEST, manifest)
    status(
        "complete",
        files=len(inventory),
        total_bytes=manifest["total_bytes"],
        archived_file_count=sum(i["file_count"] for i in inventory),
        tiff_count=sum(i["extensions"].get(".tif", 0) + i["extensions"].get(".tiff", 0) for i in inventory),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
