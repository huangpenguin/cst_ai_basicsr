"""Produce a rounded uint16 mean of three same-index reference scans."""

import hashlib
import io
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

SOURCE = Path("/input")
OUTPUT = Path("/output")
GT = OUTPUT / "GT"


def timestamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    started = time.monotonic()
    source_manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    checksum_records = json.loads((SOURCE / "remote_checksums.json").read_text())
    source_names = [f"Cylinderhead_300kV200uA_data{i}.zip" for i in (1, 2, 3)]
    zips = [zipfile.ZipFile(SOURCE / "archives" / name) for name in source_names]
    try:
        names = sorted(i.filename for i in zips[0].infolist() if i.filename.lower().endswith((".tif", ".tiff")))
        assert len(names) == 1524
        assert len({Path(name).name for name in names}) == len(names)
        assert all(
            set(names) == {i.filename for i in z.infolist() if i.filename.lower().endswith((".tif", ".tiff"))}
            for z in zips
        )
        metadata = [z.read("data.tsv").decode("utf-8-sig") for z in zips]
        (OUTPUT / "source_metadata").mkdir(exist_ok=True)
        for i, text in enumerate(metadata, 1):
            (OUTPUT / "source_metadata" / f"data{i}.tsv").write_text(text)
        description = {
            "created_at": timestamp(),
            "source_folder_url": source_manifest["source_folder_url"],
            "source_raw_directory": "/mnt/data/cst_ai/datasets/cylinderhead_300kv_raw_20261102_191914_JST",
            "sources": [r for r in checksum_records if r["Name"] in source_names],
            "method": "Same filename, arithmetic mean, no spatial registration or intensity rescaling",
            "formula": "(uint32(data1) + uint32(data2) + uint32(data3) + 1) // 3",
            "output_dtype": "uint16",
            "rounding": "nearest integer; maximum quantization error 1/3 raw unit",
            "voltage_kv": 300,
            "current_ua": 200,
            "pixel_spacing_um_from_metadata": 95.014,
            "source_count": 3,
            "slice_count": len(names),
            "validation": "ZIP CRC checked during reads; every saved TIFF reopened and compared to computed mean",
        }
        save(OUTPUT / "mean_manifest.json", description)
        records = []
        for index, name in enumerate(names, 1):
            arrays = []
            for z in zips:
                with Image.open(io.BytesIO(z.read(name))) as image:
                    array = np.asarray(image)
                    assert array.dtype == np.uint16 and array.shape == (1518, 1518), (name, array.shape, array.dtype)
                    arrays.append(array)
            result = ((arrays[0].astype(np.uint32) + arrays[1] + arrays[2] + 1) // 3).astype(np.uint16)
            output = GT / Path(name).name
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            temp = output.with_suffix(".tif.partial")
            Image.fromarray(result).save(temp, format="TIFF", compression="raw")
            with Image.open(temp) as saved:
                actual = np.asarray(saved)
                assert actual.dtype == np.uint16 and np.array_equal(actual, result), name
            digest = hashlib.sha256(temp.read_bytes()).hexdigest()
            temp.replace(output)
            records.append(
                {
                    "source_slice": name,
                    "output": "GT/" + output.name,
                    "sha256": digest,
                    "min": int(result.min()),
                    "max": int(result.max()),
                    "shape": list(result.shape),
                    "dtype": str(result.dtype),
                }
            )
            if index == 1 or index % 50 == 0 or index == len(names):
                elapsed = time.monotonic() - started
                state = {
                    "phase": "generating",
                    "updated_at": timestamp(),
                    "completed": index,
                    "total": len(names),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(elapsed / index * (len(names) - index), 1),
                }
                save(OUTPUT / "mean_status.json", state)
                print(json.dumps(state), flush=True)
        save(OUTPUT / "slice_manifest.json", records)
        state = {
            "phase": "complete",
            "completed_at": timestamp(),
            "completed": len(records),
            "total": len(names),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "all_output_tiffs_verified": True,
        }
        save(OUTPUT / "mean_status.json", state)
        print(json.dumps(state), flush=True)
    finally:
        for z in zips:
            z.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(OUTPUT / "mean_status.json", {"phase": "failed", "updated_at": timestamp(), "error": str(exc)})
        raise
