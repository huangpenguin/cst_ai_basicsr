"""Representative condition ZIPs and leakage/security cases for intake review."""

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.xray_portal import pipeline


def tiff(seed, *, dtype=np.uint16):
    pixels = np.random.default_rng(seed).integers(0, 200, size=(20, 24)).astype(dtype)
    if dtype == np.uint16:
        pixels = pixels * 173
    blob = io.BytesIO()
    Image.fromarray(pixels).save(blob, format="TIFF")
    return blob.getvalue()


def upload(batch, name, members, *, role="LQ", workpiece="workpiece-a", scan_group="scan-a", condition="10uA"):
    path = batch / "original" / f"{name}.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in members:
            archive.writestr(filename, content)
    return {
        "id": name,
        "path": str(path),
        "filename": path.name,
        "role": role,
        "workpiece": workpiece,
        "scan_group": scan_group,
        "condition": condition,
    }


def representative_batch(batch):
    files = []
    for current, role, seed in (("10uA", "LQ", 100), ("20uA", "LQ", 200), ("100uA", "GT", 300)):
        members = [(f"XY/data_{index:04d}.tif", tiff(seed + index)) for index in (0, 20, 40)]
        files.append(upload(batch, current, members, role=role, condition=current))
    return files


def decisions(report, *, mode="spatial"):
    return {
        "reviewed_by": "reviewer",
        "rationale": "Checked alignment, GT and spatial split",
        "acknowledge_issues": True,
        "split_mode": mode,
        "guard_slices": 16,
        "pairs": [
            {"id": row["id"], "approved": True, "split": {0: "train", 20: "monitor", 40: "test"}[row["slice_id"]]}
            for row in report["candidate_pairs"]
        ],
    }


def test_per_condition_zips_shared_gt_review_and_export(tmp_path):
    files = representative_batch(tmp_path)
    before = {row["id"]: pipeline._sha(Path(row["path"])) for row in files}
    report = pipeline.validate_batch(tmp_path, files)
    assert not report["issues"]
    assert len(report["images"]) == 9
    assert len(report["candidate_pairs"]) == 6
    assert all(p["status"] == "candidate" for p in report["candidate_pairs"])
    for index in (0, 20, 40):
        paired = [p for p in report["candidate_pairs"] if p["slice_id"] == index]
        assert paired[0]["gt_id"] == paired[1]["gt_id"]
    output = tmp_path / "versions" / "v1"
    manifest = pipeline.export_dataset(tmp_path, report, decisions(report), output)
    assert manifest["counts"] == {"train": 2, "monitor": 2, "test": 2}
    assert manifest["test_in_archive"] is False
    with tarfile.open(output / "dataset.tar") as archive:
        names = archive.getnames()
        assert len(names) == 11
        assert all(member.isfile() for member in archive.getmembers())
        assert "test.txt" not in names
        for row in manifest["pairs"]:
            if row["split"] == "test":
                assert f"GT/{row['filename']}" not in names
    assert pipeline._sha(output / "dataset.tar") == manifest["sha256"]
    assert before == {row["id"]: pipeline._sha(Path(row["path"])) for row in files}
    again = pipeline.validate_batch(tmp_path, files)
    assert not again["issues"]
    with pytest.raises(ValueError, match="already exists"):
        pipeline.export_dataset(tmp_path, report, decisions(report), output)


@pytest.mark.parametrize(
    "member", ["../outside.tif", "/outside.tif", "C:/outside.tif", "XY\\outside.tif", "nested.zip"]
)
def test_unsafe_archives_do_not_yield_images(tmp_path, member):
    row = upload(tmp_path, "bad", [(member, tiff(1))])
    report = pipeline.validate_batch(tmp_path, [row])
    assert not report["images"]
    assert any(i["code"] == "invalid_upload" for i in report["issues"])
    assert not (tmp_path / "outside.tif").exists()


def test_symlink_zip_rejected(tmp_path):
    row = upload(tmp_path, "bad", [])
    with zipfile.ZipFile(row["path"], "w") as archive:
        info = zipfile.ZipInfo("link.tif")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")
    report = pipeline.validate_batch(tmp_path, [row])
    assert not report["images"]
    assert "links" in report["issues"][0]["message"]


def test_missing_ambiguous_and_wrong_size_gt(tmp_path):
    files = representative_batch(tmp_path)
    only_lq = pipeline.validate_batch(tmp_path, files[:2])
    assert all(p["status"] == "missing_gt" for p in only_lq["candidate_pairs"])
    duplicate = upload(tmp_path, "another-gt", [("XY/data_0000.tif", tiff(991))], role="GT", condition="90uA")
    report = pipeline.validate_batch(tmp_path, files + [duplicate])
    assert sum(p["status"] == "ambiguous_gt" for p in report["candidate_pairs"]) == 2
    with pytest.raises(ValueError, match="Resolve missing or ambiguous"):
        pipeline.export_dataset(tmp_path, report, decisions(report), tmp_path / "bad-version")
    chosen = next(i["id"] for i in report["images"] if i["file_id"] == "100uA" and i["slice_id"] == 0)
    review = decisions(report)
    for row in review["pairs"]:
        row["gt_id"] = (
            chosen if next(p for p in report["candidate_pairs"] if p["id"] == row["id"])["slice_id"] == 0 else None
        )
    assert pipeline.export_dataset(tmp_path, report, review, tmp_path / "resolved")["counts"]["train"] == 2


def test_uint8_and_multipage_are_not_silently_converted(tmp_path):
    multi = io.BytesIO()
    Image.fromarray(np.ones((5, 5), dtype=np.uint16)).save(
        multi, format="TIFF", save_all=True, append_images=[Image.fromarray(np.zeros((5, 5), dtype=np.uint16))]
    )
    row = upload(tmp_path, "bad-types", [("data_0.tif", tiff(1, dtype=np.uint8)), ("data_1.tif", multi.getvalue())])
    report = pipeline.validate_batch(tmp_path, [row])
    assert not report["images"]
    assert sum(i["code"] == "invalid_image" for i in report["issues"]) == 2


def test_group_split_blocks_same_workpiece(tmp_path):
    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    with pytest.raises(ValueError, match="same workpiece"):
        pipeline.export_dataset(tmp_path, report, decisions(report, mode="group"), tmp_path / "version")


def test_spatial_guard_checks_all_conditions(tmp_path):
    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    review = decisions(report)
    first = review["pairs"][0]
    first["split"] = "monitor"
    with pytest.raises(ValueError, match="crosses dataset splits|guard band"):
        pipeline.export_dataset(tmp_path, report, review, tmp_path / "version")
    review = decisions(report)
    review["guard_slices"] = 20
    with pytest.raises(ValueError, match="guard band"):
        pipeline.export_dataset(tmp_path, report, review, tmp_path / "version")


def test_approved_pair_cannot_use_other_slice_gt(tmp_path):
    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    review = decisions(report)
    review["pairs"][0]["gt_id"] = next(i["id"] for i in report["images"] if i["role"] == "GT" and i["slice_id"] == 40)
    with pytest.raises(ValueError, match="mismatched slice_id"):
        pipeline.export_dataset(tmp_path, report, review, tmp_path / "version")


def test_source_modified_after_review_rejected(tmp_path):
    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    source = tmp_path / report["images"][0]["relative_path"]
    source.write_bytes(tiff(88))
    with pytest.raises(ValueError, match="Input changed"):
        pipeline.export_dataset(tmp_path, report, decisions(report), tmp_path / "version")


def test_review_is_required_for_every_pair(tmp_path):
    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    review = decisions(report)
    review["pairs"].pop()
    with pytest.raises(ValueError, match="Every candidate"):
        pipeline.export_dataset(tmp_path, report, review, tmp_path / "version")
    review = decisions(report)
    review["pairs"][0]["approved"] = False
    with pytest.raises(ValueError, match="explicitly approved"):
        pipeline.export_dataset(tmp_path, report, review, tmp_path / "version")


def test_declared_expansion_limit_blocks_before_extract(tmp_path, monkeypatch):
    row = upload(tmp_path, "large", [("data_0.tif", tiff(1))])
    monkeypatch.setattr(pipeline, "MAX_EXPANDED_BYTES", 10)
    report = pipeline.validate_batch(tmp_path, [row])
    assert not report["images"]
    assert "size limit" in report["issues"][0]["message"]


def test_pixel_duplicate_across_groups_blocked(tmp_path):
    files = []
    for i, workpiece in enumerate(("one", "two", "three")):
        for role in ("LQ", "GT"):
            files.append(
                upload(
                    tmp_path,
                    f"{workpiece}-{role}",
                    [("data_0.tif", tiff(777 if role == "GT" else i))],
                    role=role,
                    workpiece=workpiece,
                )
            )
    report = pipeline.validate_batch(tmp_path, files)
    review = decisions(report, mode="group")
    for i, row in enumerate(review["pairs"]):
        row["split"] = ("train", "monitor", "test")[i]
    with pytest.raises(ValueError, match="Identical source or pixel"):
        pipeline.export_dataset(tmp_path, report, review, tmp_path / "version")


def test_zip_crc_corruption_rejected(tmp_path):
    row = upload(tmp_path, "corrupt", [])
    with zipfile.ZipFile(row["path"], "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("data_0.tif", tiff(2))
    data = bytearray(Path(row["path"]).read_bytes())
    marker = data.index(b"II*\x00")
    data[marker + 100] ^= 1
    Path(row["path"]).write_bytes(data)
    report = pipeline.validate_batch(tmp_path, [row])
    assert not report["images"]
    assert any("CRC" in issue["message"] for issue in report["issues"])


def test_export_limit_counts_replicated_shared_gt(tmp_path, monkeypatch):
    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    unique_bytes = sum((tmp_path / image["relative_path"]).stat().st_size for image in report["images"])
    # Three GT images serve six LQ images: the flat paired export needs 12
    # image files, even though intake contained only nine distinct files.
    monkeypatch.setattr(pipeline, "MAX_EXPORT_BYTES", unique_bytes + 1)
    output = tmp_path / "too-large-version"
    with pytest.raises(ValueError, match="logical export limit"):
        pipeline.export_dataset(tmp_path, report, decisions(report), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".too-large-version-*.partial"))


def test_export_preflight_reserves_copies_plus_tar_and_margin(tmp_path, monkeypatch):
    from types import SimpleNamespace

    report = pipeline.validate_batch(tmp_path, representative_batch(tmp_path))
    by_id = {image["id"]: image for image in report["images"]}
    pair_bytes = [
        sum((tmp_path / by_id[pair[key]]["relative_path"]).stat().st_size for key in ("lq_id", "gt_id"))
        for pair in report["candidate_pairs"]
    ]
    logical_bytes = sum(pair_bytes)
    archive_bytes = sum(
        size for size, pair in zip(pair_bytes, report["candidate_pairs"], strict=True) if pair["slice_id"] != 40
    )
    archive_bytes += (4 * 2 + 3) * 4096 + 10240
    required = logical_bytes + archive_bytes + pipeline.FREE_SPACE_MARGIN
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: SimpleNamespace(free=required - 1))
    output = tmp_path / "no-space-version"
    with pytest.raises(ValueError, match="Insufficient space"):
        pipeline.export_dataset(tmp_path, report, decisions(report), output)
    assert not output.exists()
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: SimpleNamespace(free=required))
    manifest = pipeline.export_dataset(tmp_path, report, decisions(report), output)
    assert manifest["logical_bytes"] == logical_bytes
    assert manifest["archive_estimated_bytes"] >= manifest["archive_bytes"]


def test_spatial_split_cannot_separate_cross_axis_views_of_one_volume(tmp_path):
    files = []
    for condition, role, seed in (("10uA", "LQ", 100), ("100uA", "GT", 300)):
        members = [
            (f"{'XY' if index == 0 else 'XZ'}/data_{index:04d}.tif", tiff(seed + index)) for index in (0, 20, 40)
        ]
        files.append(upload(tmp_path, condition, members, role=role, condition=condition))
    report = pipeline.validate_batch(tmp_path, files)
    assert not report["issues"]
    with pytest.raises(ValueError, match="axis|axes|direction"):
        pipeline.export_dataset(tmp_path, report, decisions(report), tmp_path / "cross-axis-version")
