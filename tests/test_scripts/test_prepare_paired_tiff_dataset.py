from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_paired_tiff_dataset import (
    assign_splits,
    list_tiffs,
    validate_pair_names,
    validate_pairs,
    write_metadata,
)


def write_tiff(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint16)).save(path)


def test_validate_and_write_metadata_without_copying_images(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "metadata"
    for index in range(10):
        name = f"slice_{index:04d}.tif"
        write_tiff(source_root / "LQ" / name, index)
        write_tiff(source_root / "GT" / name, index + 1)

    lq_files = list_tiffs(source_root / "LQ")
    gt_files = list_tiffs(source_root / "GT")
    names = validate_pair_names(lq_files, gt_files)
    splits = assign_splits(names, val_fraction=0.2, split_mode="tail", seed=0)
    pairs = validate_pairs(lq_files, gt_files, splits)
    output_root.mkdir()
    write_metadata(
        output_root,
        source_root,
        pairs,
        split_mode="tail",
        seed=0,
        raw_min=0.0,
        raw_max=65535.0,
    )

    assert len((output_root / "train.txt").read_text().splitlines()) == 8
    assert len((output_root / "val.txt").read_text().splitlines()) == 2
    assert not (output_root / "LQ").exists()
    assert not (output_root / "GT").exists()


def test_validate_pair_names_rejects_missing_pair(tmp_path: Path) -> None:
    write_tiff(tmp_path / "LQ" / "only_lq.tif")
    write_tiff(tmp_path / "GT" / "only_gt.tif")

    with pytest.raises(ValueError, match="filenames do not match"):
        validate_pair_names(list_tiffs(tmp_path / "LQ"), list_tiffs(tmp_path / "GT"))
