"""Inspect checkpoint files and YAML without building a network on GPU."""

from __future__ import annotations

import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from apps.denoise_demo.registry import (
    REPO_ROOT,
    Family,
    Preset,
    resolve_ckpt_path,
)

HASH_BYTES: int = 8 * 1024 * 1024
SHA256_HEX_PREFIX: int = 12

ARCH_DESCRIPTION: dict[Family, str] = {
    "swinir_m": "SwinIR-M embed_dim=180 depths 6x6",
    "tsfdan": "TSFDANSwinIR embed_dim=60 depths 3x4",
}


class TrainYamlInfo(BaseModel):
    """Subset of a BasicSR train YAML used for debug display."""

    model_config = ConfigDict(extra="forbid")

    path: str
    exists: bool
    name: str | None = None
    experiments_root: str | None = None
    param_key_g: str | None = None


class TorchCkptInfo(BaseModel):
    """CPU-only torch.load key inspection (no SwinIR/TSFDAN construction)."""

    model_config = ConfigDict(extra="forbid")

    is_dict: bool
    top_level_keys: list[str] = Field(default_factory=list)
    has_params: bool = False
    has_params_ema: bool = False
    chosen_state: str | None = None
    tensor_count: int | None = None
    has_lora_key: bool = False


class FileInfo(BaseModel):
    """Filesystem metadata for a checkpoint path."""

    model_config = ConfigDict(extra="forbid")

    resolved_path: str
    exists: bool
    size_bytes: int | None = None
    mtime_iso: str | None = None
    sha256_12: str | None = None
    partial_hash_8mib: bool = False


class CacheInfo(BaseModel):
    """GPU/CPU model-cache status from the demo engine."""

    model_config = ConfigDict(extra="forbid")

    loaded: bool
    device: str | None = None
    last_load_seconds: float | None = None


class CheckpointDebug(BaseModel):
    """Full debug payload rendered as Markdown in the Gradio UI."""

    model_config = ConfigDict(extra="forbid")

    family: Family
    expected_arch: str
    file: FileInfo
    train_yaml: TrainYamlInfo | None = None
    torch: TorchCkptInfo | None = None
    cache: CacheInfo
    error: str | None = None


class _FileStamp(NamedTuple):
    """Cache key: resolved path plus size and mtime in nanoseconds."""

    path: str
    mtime_ns: int
    size: int


_INSPECT_LOCK: Lock = Lock()
_FILE_INFO_CACHE: dict[_FileStamp, FileInfo] = {}
_TORCH_INFO_CACHE: dict[tuple[_FileStamp, Family, str | None], TorchCkptInfo] = {}
_YAML_INFO_CACHE: dict[_FileStamp, TrainYamlInfo] = {}


def _file_stamp(path: Path) -> _FileStamp | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return _FileStamp(path=str(path.resolve()), mtime_ns=stat.st_mtime_ns, size=stat.st_size)


def _mtime_iso(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.isoformat()


def _partial_sha256_12(path: Path, size_bytes: int) -> tuple[str, bool]:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(HASH_BYTES)
    hasher.update(chunk)
    return hasher.hexdigest()[:SHA256_HEX_PREFIX], size_bytes > HASH_BYTES


def inspect_file(path: Path) -> FileInfo:
    """Stat + partial SHA256. Does not torch.load. Cached by path/mtime/size."""
    stamp = _file_stamp(path)
    if stamp is None:
        return FileInfo(resolved_path=str(path), exists=False)
    with _INSPECT_LOCK:
        cached = _FILE_INFO_CACHE.get(stamp)
    if cached is not None:
        return cached
    size = stamp.size
    digest, partial = _partial_sha256_12(path, size)
    info = FileInfo(
        resolved_path=str(path.resolve()),
        exists=True,
        size_bytes=size,
        mtime_iso=_mtime_iso(path),
        sha256_12=digest,
        partial_hash_8mib=partial,
    )
    with _INSPECT_LOCK:
        _FILE_INFO_CACHE[stamp] = info
    return info


def parse_train_yaml(train_yaml: str | None, repo_root: Path | None = None) -> TrainYamlInfo | None:
    """Parse `name` and `path.{experiments_root,param_key_g}` if the YAML exists."""
    if train_yaml is None or train_yaml.strip() == "":
        return None
    root = repo_root if repo_root is not None else REPO_ROOT
    yaml_path = Path(train_yaml)
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    yaml_path = yaml_path.resolve()
    stamp = _file_stamp(yaml_path)
    if stamp is None:
        return TrainYamlInfo(path=str(yaml_path), exists=False)
    with _INSPECT_LOCK:
        cached_yaml = _YAML_INFO_CACHE.get(stamp)
    if cached_yaml is not None:
        return cached_yaml
    info = TrainYamlInfo(path=str(yaml_path), exists=True)
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"train YAML must be a mapping: {yaml_path}")
    path_block = payload.get("path")
    experiments_root: str | None = None
    param_key_g: str | None = None
    if isinstance(path_block, dict):
        raw_root = path_block.get("experiments_root")
        raw_key = path_block.get("param_key_g")
        experiments_root = None if raw_root is None else str(raw_root)
        param_key_g = None if raw_key is None else str(raw_key)
    name = payload.get("name")
    parsed = info.model_copy(
        update={
            "name": None if name is None else str(name),
            "experiments_root": experiments_root,
            "param_key_g": param_key_g,
        }
    )
    with _INSPECT_LOCK:
        _YAML_INFO_CACHE[stamp] = parsed
    return parsed


def _state_dict_from_checkpoint(
    checkpoint: dict[str, Any],
    family: Family,
    param_key_hint: str | None,
) -> tuple[str, dict[str, Any]]:
    """Choose the same state-dict slot the family loader would use (hint is optional)."""
    if param_key_hint is not None and param_key_hint in checkpoint:
        state = checkpoint[param_key_hint]
        if isinstance(state, dict):
            return param_key_hint, state
        raise TypeError(f"checkpoint[{param_key_hint!r}] is not a dict")
    if family == "swinir_m":
        if "params_ema" in checkpoint and isinstance(checkpoint["params_ema"], dict):
            return "params_ema", checkpoint["params_ema"]
        if "params" in checkpoint and isinstance(checkpoint["params"], dict):
            return "params", checkpoint["params"]
        raise KeyError("SwinIR checkpoint has neither params_ema nor params")
    if "params" in checkpoint and isinstance(checkpoint["params"], dict):
        return "params", checkpoint["params"]
    if "params_ema" in checkpoint and isinstance(checkpoint["params_ema"], dict):
        return "params_ema", checkpoint["params_ema"]
    return "top_level", checkpoint


def _tensor_count(state: dict[str, Any]) -> int:
    return sum(1 for value in state.values() if torch.is_tensor(value))


def _has_lora_key(state: dict[str, Any]) -> bool:
    return any("lora_" in str(key) for key in state)


def inspect_torch_cpu(path: Path, family: Family, param_key_hint: str | None) -> TorchCkptInfo:
    """torch.load on CPU for keys only; does not construct SwinIR/TSFDAN.

    Results are cached by resolved path, mtime_ns, size, family, and param_key_hint.
    """
    stamp = _file_stamp(path)
    if stamp is None:
        raise FileNotFoundError(f"checkpoint not found: {path}")
    torch_key = (stamp, family, param_key_hint)
    with _INSPECT_LOCK:
        cached_torch = _TORCH_INFO_CACHE.get(torch_key)
    if cached_torch is not None:
        return cached_torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        info = TorchCkptInfo(
            is_dict=False,
            top_level_keys=[type(checkpoint).__name__],
        )
        with _INSPECT_LOCK:
            _TORCH_INFO_CACHE[torch_key] = info
        return info
    keys = [str(key) for key in checkpoint.keys()]
    chosen_name, state = _state_dict_from_checkpoint(checkpoint, family, param_key_hint)
    info = TorchCkptInfo(
        is_dict=True,
        top_level_keys=keys,
        has_params="params" in checkpoint and isinstance(checkpoint["params"], dict),
        has_params_ema="params_ema" in checkpoint and isinstance(checkpoint["params_ema"], dict),
        chosen_state=chosen_name,
        tensor_count=_tensor_count(state),
        has_lora_key=_has_lora_key(state),
    )
    with _INSPECT_LOCK:
        _TORCH_INFO_CACHE[torch_key] = info
    return info


def inspect_checkpoint(
    *,
    family: Family,
    ckpt: str | Path,
    param_key_hint: str | None = None,
    train_yaml: str | None = None,
    cache: CacheInfo | None = None,
    repo_root: Path | None = None,
) -> CheckpointDebug:
    """Collect file + YAML + torch key metadata. Missing ckpt skips torch.load.

    File hash, YAML parse, and torch key inspection are cached by path + mtime_ns +
    size. `cache` (GPU/CPU model-engine status) is never cached and always taken
    from the caller so denoise can refresh loaded/device/last_load_seconds.
    """
    cache_info = cache if cache is not None else CacheInfo(loaded=False)
    resolved = resolve_ckpt_path(str(ckpt), repo_root)
    file_info = inspect_file(resolved)
    error: str | None = None
    torch_info: TorchCkptInfo | None = None
    train_info: TrainYamlInfo | None = None
    try:
        train_info = parse_train_yaml(train_yaml, repo_root)
    except Exception:
        error = traceback.format_exc()
    if file_info.exists:
        try:
            torch_info = inspect_torch_cpu(resolved, family, param_key_hint)
        except Exception:
            torch_error = traceback.format_exc()
            error = torch_error if error is None else error + "\n" + torch_error
    return CheckpointDebug(
        family=family,
        expected_arch=ARCH_DESCRIPTION[family],
        file=file_info,
        train_yaml=train_info,
        torch=torch_info,
        cache=cache_info,
        error=error,
    )


def inspect_preset(
    preset: Preset,
    cache: CacheInfo | None = None,
    repo_root: Path | None = None,
) -> CheckpointDebug:
    """Inspect a registry preset."""
    return inspect_checkpoint(
        family=preset.family,
        ckpt=preset.ckpt,
        param_key_hint=preset.param_key,
        train_yaml=preset.train_yaml,
        cache=cache,
        repo_root=repo_root,
    )


def format_debug_markdown(debug: CheckpointDebug) -> str:
    """Render CheckpointDebug as Markdown for the Gradio panel."""
    file_info = debug.file
    lines: list[str] = [
        "## Checkpoint debug",
        "",
        "### File",
        f"- resolved path: `{file_info.resolved_path}`",
        f"- exists: `{file_info.exists}`",
        f"- size bytes: `{file_info.size_bytes}`",
        f"- mtime (UTC ISO): `{file_info.mtime_iso}`",
        f"- SHA256 first 12 hex (first 8 MiB): `{file_info.sha256_12}`",
    ]
    if file_info.partial_hash_8mib:
        lines.append("- hash note: `partial hash 8MiB` (file larger than 8 MiB)")
    lines.extend(
        [
            "",
            "### Family / architecture",
            f"- family: `{debug.family}`",
            f"- expected arch: `{debug.expected_arch}`",
            "",
            "### train_yaml",
        ]
    )
    if debug.train_yaml is None:
        lines.append("- (none)")
    else:
        ty = debug.train_yaml
        lines.extend(
            [
                f"- path: `{ty.path}`",
                f"- exists: `{ty.exists}`",
                f"- name: `{ty.name}`",
                f"- path.experiments_root: `{ty.experiments_root}`",
                f"- path.param_key_g: `{ty.param_key_g}`",
            ]
        )
    lines.extend(["", "### torch checkpoint"])
    if not file_info.exists:
        lines.append("- skipped `torch.load` because the file does not exist")
    elif debug.torch is None:
        lines.append("- torch inspection failed (see exception)")
    else:
        info = debug.torch
        keys = ", ".join(f"`{key}`" for key in info.top_level_keys[:80])
        if len(info.top_level_keys) > 80:
            keys += ", …"
        lines.extend(
            [
                f"- is dict: `{info.is_dict}`",
                f"- top-level keys: {keys if keys else '(none)'}",
                f"- has `params`: `{info.has_params}`",
                f"- has `params_ema`: `{info.has_params_ema}`",
                f"- chosen state: `{info.chosen_state}`",
                f"- tensor count in chosen state: `{info.tensor_count}`",
                f"- any key contains `lora_`: `{info.has_lora_key}`",
            ]
        )
    lines.extend(
        [
            "",
            "### GPU cache",
            f"- loaded: `{debug.cache.loaded}`",
            f"- device: `{debug.cache.device}`",
            f"- last load seconds: `{debug.cache.last_load_seconds}`",
        ]
    )
    if debug.error is not None:
        lines.extend(
            [
                "",
                "### Exception",
                "```",
                debug.error.rstrip(),
                "```",
            ]
        )
    return "\n".join(lines)
