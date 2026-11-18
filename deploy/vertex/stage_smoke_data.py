"""Package a few unchanged real training pairs; never include held-out test data."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "bundle/smoke-data.tar.gz")
    args = parser.parse_args()
    names = [line.split()[0] for line in args.train_list.read_text().splitlines() if line.strip()]
    if not 1 <= args.count <= len(names):
        raise ValueError("count must be between 1 and the training split size")
    selected = [names[index * len(names) // args.count] for index in range(args.count)]
    manifest = {"source_training_list": str(args.train_list), "pairs": [], "unchanged_pixels": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name in selected:
            pair = {"name": name, "files": []}
            for folder in ("LQ", "GT"):
                source = args.data_root / folder / name
                archive.add(source, arcname=f"{folder}/{name}", recursive=False)
                pair["files"].append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            manifest["pairs"].append(pair)
        archive.add(args.data_root / "normalization.json", arcname="normalization.json")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(args.output), "pairs": len(selected), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
