"""Load and validate the denoise-demo model registry YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Family = Literal["swinir_m", "tsfdan"]

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_YAML: Path = Path(__file__).resolve().parent / "models.yaml"
HOME_PREFIX: str = "/home/huang/"
MNT_HOME_PREFIX: str = "/mnt/home/huang/"
ENV_MODELS_YAML: str = "DENOISE_DEMO_MODELS_YAML"
ENV_DEFAULT_ID: str = "DENOISE_DEMO_DEFAULT_ID"


class Preset(BaseModel):
    """One selectable checkpoint preset."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    family: Family
    ckpt: str
    param_key: str | None = None
    train_yaml: str | None = None
    normalization: str | None = None
    notes: str | None = None

    @field_validator("id", "label", "ckpt")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @field_validator("param_key")
    @classmethod
    def _param_key_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped not in {"params", "params_ema"}:
            raise ValueError("param_key must be 'params', 'params_ema', or null")
        return stripped


class NormalizationSpec(BaseModel):
    """uint16 display/window mapping stored as JSON on the preset."""

    model_config = ConfigDict(extra="forbid")

    raw_min: float
    raw_max: float

    @model_validator(mode="after")
    def _range_ok(self) -> NormalizationSpec:
        if self.raw_max <= self.raw_min:
            raise ValueError("raw_max must be greater than raw_min")
        return self


class ModelsRegistry(BaseModel):
    """Top-level models.yaml document."""

    model_config = ConfigDict(extra="forbid")

    default_id: str
    presets: list[Preset] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_in_presets(self) -> ModelsRegistry:
        ids = [preset.id for preset in self.presets]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate preset ids: {ids}")
        if self.default_id not in set(ids):
            raise ValueError(f"default_id {self.default_id!r} is not in presets")
        return self

    def by_id(self) -> dict[str, Preset]:
        """Return presets keyed by id."""
        return {preset.id: preset for preset in self.presets}

    def get(self, preset_id: str) -> Preset:
        """Return a preset or raise KeyError."""
        mapping = self.by_id()
        if preset_id not in mapping:
            raise KeyError(f"unknown preset id: {preset_id!r}")
        return mapping[preset_id]


def alternate_home_mnt_path(path: Path) -> Path | None:
    """Map `/home/huang/` ↔ `/mnt/home/huang/` (container vs host NFS)."""
    text = str(path)
    if text.startswith(MNT_HOME_PREFIX):
        return Path(HOME_PREFIX + text[len(MNT_HOME_PREFIX) :])
    if text.startswith(HOME_PREFIX):
        return Path(MNT_HOME_PREFIX + text[len(HOME_PREFIX) :])
    return None


def resolve_ckpt_path(ckpt: str, repo_root: Path | None = None) -> Path:
    """Make `ckpt` absolute; if missing, try the home↔mnt prefix swap.

    Missing files are still returned (the listed absolute path) so the UI can
    mark them missing. This function never substitutes a different checkpoint.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    listed = Path(ckpt)
    if not listed.is_absolute():
        listed = root / listed
    listed_abs = listed.resolve()
    if listed_abs.exists():
        return listed_abs
    alternate = alternate_home_mnt_path(listed_abs)
    if alternate is not None and alternate.exists():
        return alternate.resolve()
    return listed_abs


def parse_normalization(raw: str | None, repo_root: Path | None = None) -> NormalizationSpec:
    """Load optional `normalization.json` (path) or inline JSON; default 0 / 65535."""
    if raw is None or raw.strip() == "":
        return NormalizationSpec(raw_min=0.0, raw_max=65535.0)
    stripped = raw.strip()
    if stripped.startswith("{"):
        return NormalizationSpec.model_validate_json(stripped)
    root = repo_root if repo_root is not None else REPO_ROOT
    listed = Path(stripped)
    if not listed.is_absolute():
        listed = root / listed
    resolved = resolve_ckpt_path(str(listed), root)
    if not resolved.is_file():
        raise FileNotFoundError(f"normalization JSON not found: {resolved}")
    return NormalizationSpec.model_validate_json(resolved.read_text(encoding="utf-8"))


def models_yaml_path() -> Path:
    """Registry YAML path, honoring `DENOISE_DEMO_MODELS_YAML`."""
    override = os.environ.get(ENV_MODELS_YAML)
    if override is None or override.strip() == "":
        return DEFAULT_MODELS_YAML
    return Path(override).expanduser().resolve()


def load_registry(yaml_path: Path | None = None) -> ModelsRegistry:
    """Load and validate the registry. Raises if the file or schema is invalid."""
    path = yaml_path if yaml_path is not None else models_yaml_path()
    if not path.is_file():
        raise FileNotFoundError(f"models registry YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"models registry must be a mapping, got {type(payload).__name__}")
    registry = ModelsRegistry.model_validate(payload)
    default_override = os.environ.get(ENV_DEFAULT_ID)
    if default_override is not None and default_override.strip() != "":
        override_id = default_override.strip()
        if override_id not in registry.by_id():
            raise ValueError(f"{ENV_DEFAULT_ID}={override_id!r} is not a preset id in {path}")
        registry = registry.model_copy(update={"default_id": override_id})
    return registry
