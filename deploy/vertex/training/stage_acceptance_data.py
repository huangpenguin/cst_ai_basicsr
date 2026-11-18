"""Package fixed real train/monitor pairs, with byte hashes, for bounded acceptance."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    splits = {
        name: [line.split()[0] for line in (root / name).read_text().splitlines() if line.strip()]
        for name in ("train.txt", "val_monitor.txt", "test.txt")
    }
    for a, b in [("train.txt", "val_monitor.txt"), ("train.txt", "test.txt"), ("val_monitor.txt", "test.txt")]:
        if set(splits[a]) & set(splits[b]):
            raise ValueError("Source split overlap")
    selected = {
        "train.txt": [splits["train.txt"][i * len(splits["train.txt"]) // 4] for i in range(4)],
        "val_monitor.txt": [splits["val_monitor.txt"][0]],
    }
    members = {}
    for split, names in selected.items():
        members[split] = ("\n".join(names) + "\n").encode()
        for name in names:
            if Path(name).name != name:
                raise ValueError("Expected flat safe filenames")
            for folder in ("GT", "LQ"):
                members[f"{folder}/{name}"] = (root / folder / name).read_bytes()
    members["normalization.json"] = (root / "normalization.json").read_bytes()
    manifest = {
        "source_root": str(root),
        "selected_splits": selected,
        "test_included": False,
        "split_sha256": {k: hashlib.sha256((root / k).read_bytes()).hexdigest() for k in splits},
        "members": {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in members.items()},
    }
    with args.output.open("xb") as output, tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    with args.output.open("rb") as f:
        manifest["archive_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    manifest["archive_bytes"] = args.output.stat().st_size
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "archive": str(args.output),
                "bytes": manifest["archive_bytes"],
                "sha256": manifest["archive_sha256"],
                "train_pairs": 4,
                "monitor_pairs": 1,
                "test_pairs": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
