"""Export existing held-out lists as hardlinks and a small Demo input ZIP."""

import argparse
import json
import os
import zipfile
from pathlib import Path


def export(dataset):
    names = (dataset / "test.txt").read_text().splitlines()
    preview = (dataset / "test_preview.txt").read_text().splitlines()
    if len(names) != len(set(names)) or not set(preview) <= set(names):
        raise ValueError("Invalid test lists")
    for group, selected in [("test_holdout", names), ("test_demo_samples", preview)]:
        for kind in ["LQ", "GT"]:
            target = dataset / group / kind
            target.mkdir(parents=True, exist_ok=True)
            for name in selected:
                if Path(name).name != name:
                    raise ValueError(f"Expected a filename, got {name!r}")
                source = dataset / kind / name
                dest = target / name
                if not dest.exists():
                    os.link(source, dest)
                if not os.path.samefile(source, dest):
                    raise ValueError(f"Existing destination is not the expected hardlink: {dest}")
    selection = {"selection": "Fixed before training in test_preview.txt", "images": len(preview), "filenames": preview}
    selection_path = dataset / "test_demo_samples" / "selection.json"
    if not selection_path.exists():
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    archive = dataset / "test_demo_LQ_40.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name in preview:
                output.write(dataset / "LQ" / name, f"LQ/{name}")
            output.writestr("selection.json", json.dumps(selection, indent=2) + "\n")
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as output:
        expected = {f"LQ/{name}" for name in preview} | {"selection.json"}
        if set(output.namelist()) != expected or len(output.namelist()) != len(expected):
            raise ValueError("Existing Demo ZIP does not match test_preview.txt")
    return {"test_pairs": len(names), "demo_pairs": len(preview), "demo_zip": str(archive)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    print(json.dumps(export(parser.parse_args().dataset), indent=2))
