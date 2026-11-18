"""Preserve source TIFF bytes and construct grouped, spatially separated splits."""

import hashlib
import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
JOB = json.loads((HERE.parent / "config/job.json").read_text())
DATA = Path(JOB["data"])
MEAN = Path(JOB["mean"])
SOURCE = Path(JOB["source"])
CURRENTS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
LOCK = threading.Lock()
DONE = 0
START = time.monotonic()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def blocks(ids):
    answer = []
    for i in sorted(ids):
        if not answer or i > answer[-1][1] + 1:
            answer.append([i, i])
        else:
            answer[-1][1] = i
    return answer


def pair_name(current, index):
    return f"cylinderhead_300kV{current:03d}uA_data_{index:04d}.tif"


def prepare():
    global DONE
    (DATA / "LQ").mkdir(exist_ok=True)
    (DATA / "GT").mkdir(exist_ok=True)
    profile = json.loads((HERE.parent / "config/gt_content_profile.json").read_text())
    eligible = [r["slice"] for r in profile if r["material_fraction_gt10000"] >= 0.01]
    n = len(eligible)
    test = set(eligible[int(n * 0.2) : int(n * 0.3)] + eligible[int(n * 0.7) : int(n * 0.8)])
    val = set(eligible[int(n * 0.45) : int(n * 0.55)])
    guard = {i for i in eligible if i not in test | val and any(abs(i - j) <= 16 for j in test | val)}
    train = set(eligible) - test - val - guard
    groups = {"train": train, "val": val, "test": test}
    assert not (train & val or train & test or val & test)
    assert min(abs(a - b) for a in train for b in val | test) > 16
    assert min(abs(a - b) for a in val for b in test) > 16
    active = sorted(train | val | test)
    which = {i: split for split, ids in groups.items() for i in ids}
    split_manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "method": (
            "Group by original slice index across ALL currents; deterministic contiguous test/val "
            "blocks; 16-slice guard bands"
        ),
        "content_filter": "GT thumbnail stride=8, fraction above raw 10000 >= 0.01; no LQ or model result used",
        "alignment": "none; original TIFF bytes preserved at user request",
        "normalization": {"raw_min": 0.0, "raw_max": 65535.0},
        "currents_ua": CURRENTS,
        "guard_slices": sorted(guard),
        "content_excluded_slices": sorted(set(range(1524)) - set(eligible)),
        "sets": {
            k: {"slice_count": len(v), "pair_count": len(v) * 10, "slice_ranges": blocks(v), "slice_indices": sorted(v)}
            for k, v in groups.items()
        },
        "test_policy": (
            "Never used by the training or validation loader; fixed preview list for "
            "post-training final-checkpoint inspection only"
        ),
    }
    save(DATA / "split_manifest.json", split_manifest)
    save(DATA / "normalization.json", {"raw_min": 0.0, "raw_max": 65535.0})
    lists = {split: [pair_name(c, i) for i in sorted(ids) for c in CURRENTS] for split, ids in groups.items()}
    for split, names in lists.items():
        (DATA / f"{split}.txt").write_text("\n".join(names) + "\n")
    val_ids = sorted(val)
    val_preview = [val_ids[len(val_ids) // 4], val_ids[3 * len(val_ids) // 4]]
    test_preview = []
    for lo, hi in blocks(test):
        test_preview.extend([lo + (hi - lo) // 3, lo + 2 * (hi - lo) // 3])
    (DATA / "val_monitor.txt").write_text("".join(pair_name(c, i) + "\n" for i in val_preview for c in CURRENTS))
    (DATA / "test_preview.txt").write_text("".join(pair_name(c, i) + "\n" for i in test_preview for c in CURRENTS))
    save(
        DATA / "preview_selection.json",
        {
            "val_slice_indices": val_preview,
            "test_slice_indices": test_preview,
            "test_preview_pairs": len(test_preview) * 10,
            "selection": "Fixed by block positions before training",
        },
    )
    mean_records = {Path(r["output"]).name: r for r in json.loads((MEAN / "slice_manifest.json").read_text())}
    total = len(active) * len(CURRENTS)
    print(json.dumps({k: v["pair_count"] for k, v in split_manifest["sets"].items()}), flush=True)
    save(DATA / "prepare_status.json", {"phase": "extracting", "completed": 0, "total": total})

    def worker(current):
        global DONE
        records = []
        archive_path = SOURCE / "archives" / f"Cylinderhead_300kV{current}uA_data.zip"
        with zipfile.ZipFile(archive_path) as archive:
            assert {f"XY/data_{i:04d}.tif" for i in active} <= set(archive.namelist())
            for index in active:
                source_name = f"XY/data_{index:04d}.tif"
                name = pair_name(current, index)
                lq = DATA / "LQ" / name
                gt = DATA / "GT" / name
                blob = archive.read(source_name)  # ZipFile checks CRC while reading.
                digest = hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as image:
                    assert image.size == (1518, 1518) and np.asarray(image).dtype == np.uint16
                if lq.exists():
                    assert hashlib.sha256(lq.read_bytes()).hexdigest() == digest
                else:
                    temporary = lq.with_suffix(".tif.partial")
                    temporary.write_bytes(blob)
                    temporary.replace(lq)
                assert lq.stat().st_size == len(blob)
                source_gt = MEAN / "GT" / f"data_{index:04d}.tif"
                if not gt.exists():
                    os.link(source_gt, gt)
                assert os.path.samefile(source_gt, gt)
                records.append(
                    {
                        "filename": name,
                        "slice": index,
                        "current_ua": current,
                        "split": which[index],
                        "source_archive": archive_path.name,
                        "source_member": source_name,
                        "lq_sha256": digest,
                        "gt_sha256": mean_records[source_gt.name]["sha256"],
                        "gt_source": str(source_gt),
                    }
                )
                with LOCK:
                    DONE += 1
                    if DONE % 250 == 0 or DONE == total:
                        elapsed = time.monotonic() - START
                        status = {
                            "phase": "extracting",
                            "completed": DONE,
                            "total": total,
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": round(elapsed / DONE * (total - DONE), 1),
                        }
                        save(DATA / "prepare_status.json", status)
                        print(json.dumps(status), flush=True)
        return records

    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(worker, c) for c in CURRENTS]):
            records.extend(future.result())
    records.sort(key=lambda r: r["filename"])
    assert len(records) == total == len(list((DATA / "LQ").glob("*.tif"))) == len(list((DATA / "GT").glob("*.tif")))
    assert {r["filename"] for r in records} == set().union(*(set(x) for x in lists.values()))
    assert not list(DATA.rglob("*.partial"))
    save(DATA / "pairs_manifest.json", records)
    save(DATA / "test_checksums.json", [r for r in records if r["split"] == "test"])
    for name in ("test.txt", "test_preview.txt", "test_checksums.json", "split_manifest.json"):
        (DATA / name).chmod(0o444)
    save(
        DATA / "prepare_status.json",
        {
            "phase": "complete",
            "completed": DONE,
            "total": total,
            "elapsed_seconds": round(time.monotonic() - START, 1),
            "split_counts": {k: len(v) for k, v in lists.items()},
            "no_slice_overlap": True,
            "guard_band_slices": 16,
            "source_pixels_unchanged": True,
        },
    )
    print((DATA / "prepare_status.json").read_text(), flush=True)


if __name__ == "__main__":
    try:
        prepare()
    except Exception as exc:
        save(DATA / "prepare_status.json", {"phase": "failed", "completed": DONE, "error": str(exc)})
        raise
