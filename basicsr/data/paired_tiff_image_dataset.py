import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY


def _load_normalization(opt: dict[str, Any]) -> tuple[float, float]:
    normalization = opt.get("normalization")
    if normalization is None:
        return 0.0, 65535.0

    if isinstance(normalization, str):
        with Path(normalization).expanduser().open("r", encoding="utf-8") as file:
            normalization = json.load(file)

    raw_min = float(normalization.get("raw_min", 0.0))
    raw_max = float(normalization.get("raw_max", 65535.0))
    if raw_max <= raw_min:
        raise ValueError(f"Invalid TIFF normalization range: raw_min={raw_min}, raw_max={raw_max}")
    return raw_min, raw_max


def _read_tiff(path: str | Path, raw_min: float, raw_max: float) -> np.ndarray:
    with Image.open(path) as image:
        img = np.array(image)
    if img.ndim == 3:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale TIFF image, got shape {img.shape}: {path}")

    img = img.astype(np.float32)
    img = np.clip((img - raw_min) / (raw_max - raw_min), 0.0, 1.0).astype(np.float32, copy=False)
    return img[..., None]


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return torch.from_numpy(img).float()


@DATASET_REGISTRY.register()
class PairedTiffImageDataset(data.Dataset):
    """Paired uint16 TIFF dataset for real grayscale denoising.

    This dataset keeps BasicSR's paired restoration contract but avoids the
    default image reader's uint8-oriented `/255` normalization.
    """

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.raw_min, self.raw_max = _load_normalization(opt)

        io_backend_opt = opt.get("io_backend", {"type": "disk"})
        if io_backend_opt.get("type") != "disk":
            raise ValueError("PairedTiffImageDataset only supports disk io_backend.")

        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.filename_tmpl = opt.get("filename_tmpl", "{}")

        if "meta_info_file" in opt and opt["meta_info_file"] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ["lq", "gt"], opt["meta_info_file"], self.filename_tmpl
            )
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ["lq", "gt"], self.filename_tmpl)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        scale = self.opt["scale"]

        gt_path = self.paths[index]["gt_path"]
        lq_path = self.paths[index]["lq_path"]
        img_gt = _read_tiff(gt_path, self.raw_min, self.raw_max)
        img_lq = _read_tiff(lq_path, self.raw_min, self.raw_max)

        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"])

        if self.opt["phase"] != "train":
            img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

        tensor_gt, tensor_lq = _to_tensor(img_gt), _to_tensor(img_lq)
        if self.mean is not None or self.std is not None:
            normalize(tensor_lq, self.mean, self.std, inplace=True)
            normalize(tensor_gt, self.mean, self.std, inplace=True)

        return {"lq": tensor_lq, "gt": tensor_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self) -> int:
        return len(self.paths)
