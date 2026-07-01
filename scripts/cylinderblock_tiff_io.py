"""TIFF I/O and geometry helpers for Cylinderblock uint16 paired preprocessing."""

from __future__ import annotations

import os
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate, zoom

TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
}


@dataclass(frozen=True)
class TiffImage:
    path: Path
    endian: str
    width: int
    height: int
    bits_per_sample: int
    compression: int
    sample_format: int
    photometric: int
    pixel_bytes: bytes


def tiff_value(endian: str, typ: int, count: int, raw_value: bytes) -> int | tuple[int, ...] | bytes:
    size = TYPE_SIZES.get(typ)
    if size is None:
        return raw_value
    total_size = size * count
    value_bytes = raw_value[:total_size]
    if typ == 3:
        values = struct.unpack(endian + f"{count}H", value_bytes)
    elif typ == 4:
        values = struct.unpack(endian + f"{count}I", value_bytes)
    elif typ == 8:
        values = struct.unpack(endian + f"{count}h", value_bytes)
    elif typ == 9:
        values = struct.unpack(endian + f"{count}i", value_bytes)
    else:
        return value_bytes
    return values[0] if count == 1 else values


def read_tiff(path: Path, load_pixels: bool = True) -> TiffImage:
    with path.open("rb") as file:
        header = file.read(8)
        byte_order = header[:2]
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError(f"Not a classic TIFF file: {path}")

        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic != 42:
            raise ValueError(f"Unsupported TIFF magic {magic}: {path}")

        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        file.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", file.read(2))[0]
        tags: dict[int, int | tuple[int, ...] | bytes] = {}
        for _ in range(entry_count):
            entry = file.read(12)
            tag, typ, count = struct.unpack(endian + "HHI", entry[:8])
            value_or_offset = entry[8:12]
            total_size = TYPE_SIZES.get(typ, 1) * count
            if total_size <= 4:
                value_bytes = value_or_offset[:total_size]
            else:
                current_pos = file.tell()
                value_offset = struct.unpack(endian + "I", value_or_offset)[0]
                file.seek(value_offset)
                value_bytes = file.read(total_size)
                file.seek(current_pos)
            tags[tag] = tiff_value(endian, typ, count, value_bytes)

    width = int(tags[256])
    height = int(tags[257])
    bits = tags.get(258, 16)
    bits_per_sample = int(bits[0] if isinstance(bits, tuple) else bits)
    compression = int(tags.get(259, 1))
    photometric = int(tags.get(262, 1))
    sample_format = int(tags.get(339, 1))
    offsets = tags.get(273) or tags.get(324)
    byte_counts = tags.get(279) or tags.get(325)
    if offsets is None or byte_counts is None:
        raise ValueError(f"TIFF is missing strip/tile offsets or byte counts: {path}")
    if isinstance(offsets, int):
        offsets = (offsets,)
    if isinstance(byte_counts, int):
        byte_counts = (byte_counts,)

    if compression != 1 or bits_per_sample != 16 or sample_format != 1:
        raise ValueError(
            "Only uncompressed unsigned uint16 grayscale TIFF is supported: "
            f"{path} (compression={compression}, bits={bits_per_sample}, sample_format={sample_format})"
        )

    pixel_bytes = b""
    if load_pixels:
        with path.open("rb") as file:
            chunks = []
            for offset, count in zip(offsets, byte_counts, strict=True):
                file.seek(int(offset))
                chunks.append(file.read(int(count)))
            pixel_bytes = b"".join(chunks)
    expected_bytes = width * height * 2
    if load_pixels and len(pixel_bytes) < expected_bytes:
        raise ValueError(
            f"TIFF pixel payload is shorter than expected in {path}: {len(pixel_bytes)} < {expected_bytes}"
        )
    return TiffImage(
        path=path,
        endian=endian,
        width=width,
        height=height,
        bits_per_sample=bits_per_sample,
        compression=compression,
        sample_format=sample_format,
        photometric=photometric,
        pixel_bytes=pixel_bytes[:expected_bytes] if load_pixels else b"",
    )


def u16_bytes_to_array(image: TiffImage) -> np.ndarray:
    dtype = "<u2" if image.endian == "<" else ">u2"
    arr = np.frombuffer(image.pixel_bytes, dtype=dtype).reshape(image.height, image.width)
    if dtype.startswith(">"):
        return arr.byteswap().newbyteorder()
    return arr.copy()


def array_to_u16_bytes(arr: np.ndarray, endian: str = "<") -> bytes:
    out = np.ascontiguousarray(arr.astype("<u2", copy=False))
    if endian == ">":
        out = out.byteswap()
    return out.tobytes()


def pack_ifd_entry(tag: int, typ: int, count: int, value: int) -> bytes:
    value_bytes = struct.pack("<H", value) + b"\x00\x00" if typ == 3 and count == 1 else struct.pack("<I", value)
    return struct.pack("<HHI", tag, typ, count) + value_bytes


def write_u16_tiff(path: Path, pixel_bytes: bytes, width: int, height: int, photometric: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = width * height * 2
    if len(pixel_bytes) != expected_bytes:
        raise ValueError(f"Pixel payload size mismatch for {path}: {len(pixel_bytes)} != {expected_bytes}")

    entries = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, photometric),
        (273, 4, 1, 0),
        (277, 3, 1, 1),
        (278, 4, 1, height),
        (279, 4, 1, expected_bytes),
        (296, 3, 1, 1),
        (339, 3, 1, 1),
    ]
    ifd_size = 2 + len(entries) * 12 + 4
    pixel_offset = 8 + ifd_size
    entries = [(tag, typ, count, pixel_offset if tag == 273 else value) for tag, typ, count, value in entries]

    with path.open("wb") as file:
        file.write(b"II")
        file.write(struct.pack("<H", 42))
        file.write(struct.pack("<I", 8))
        file.write(struct.pack("<H", len(entries)))
        for entry in entries:
            file.write(pack_ifd_entry(*entry))
        file.write(struct.pack("<I", 0))
        file.write(pixel_bytes)


def u16_stats(pixel_bytes: bytes, endian: str = "<") -> tuple[int, int, int, int]:
    values = array("H")
    values.frombytes(pixel_bytes)
    host_little = struct.pack("=H", 1) == b"\x01\x00"
    file_little = endian == "<"
    if host_little != file_little:
        values.byteswap()
    if not values:
        return 0, 0, 0, 0
    min_value = min(values)
    max_value = max(values)
    zero_count = values.count(0)
    saturated_count = values.count(65535)
    return min_value, max_value, zero_count, saturated_count


def rotate_u16_cw(
    arr: np.ndarray,
    degrees_cw: float,
    *,
    edge_mode: str = "nearest",
) -> np.ndarray:
    """Rotate a 2D uint16 image clockwise by ``degrees_cw`` degrees.

    Out-of-bounds samples are filled from the nearest in-bounds pixel (``edge_mode='nearest'``)
    so rotated CT ring edges stay gray instead of turning black.
    """
    if degrees_cw % 360 == 0:
        return arr.copy()
    rotated = rotate(
        arr.astype(np.float32),
        -degrees_cw,
        reshape=False,
        order=1,
        mode=edge_mode,
    )
    return np.clip(np.round(rotated), 0, 65535).astype(np.uint16)


def resize_u16(
    arr: np.ndarray,
    target_height: int,
    target_width: int,
    *,
    edge_mode: str = "nearest",
) -> np.ndarray:
    """Resize a 2D uint16 image to ``target_height`` x ``target_width`` (bilinear)."""
    if arr.shape[0] == target_height and arr.shape[1] == target_width:
        return arr.copy()
    zoom_h = target_height / arr.shape[0]
    zoom_w = target_width / arr.shape[1]
    resized = zoom(
        arr.astype(np.float32),
        (zoom_h, zoom_w),
        order=1,
        mode=edge_mode,
    )
    return np.clip(np.round(resized), 0, 65535).astype(np.uint16)


def center_pad_u16(arr: np.ndarray, target_height: int, target_width: int, *, fill_value: int = 0) -> np.ndarray:
    if arr.shape[0] > target_height or arr.shape[1] > target_width:
        raise ValueError(f"Cannot center-pad {arr.shape[1]}x{arr.shape[0]} into {target_width}x{target_height}")
    top = (target_height - arr.shape[0]) // 2
    left = (target_width - arr.shape[1]) // 2
    canvas = np.full((target_height, target_width), fill_value, dtype=np.uint16)
    canvas[top : top + arr.shape[0], left : left + arr.shape[1]] = arr
    return canvas


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(target)
