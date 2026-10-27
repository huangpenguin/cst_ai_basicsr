#!/usr/bin/env python3
"""Gradio denoise demo: upload grayscale CT → denoise → original | denoised."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from apps.denoise_demo.checkpoint_info import (  # noqa: E402
    CheckpointDebug,
    format_debug_markdown,
    inspect_checkpoint,
)
from apps.denoise_demo.engine import (  # noqa: E402
    FloatHW,
    OutputFormat,
    UInt8HW,
    get_engine,
    idle_seconds_from_env,
    load_input_image,
    to_display_u8,
    write_download_file,
)
from apps.denoise_demo.gpu_status import (  # noqa: E402
    format_gpu_status_markdown,
    query_gpu_status,
)
from apps.denoise_demo.registry import (  # noqa: E402
    REPO_ROOT,
    Family,
    ModelsRegistry,
    Preset,
    load_registry,
    parse_normalization,
    resolve_ckpt_path,
)

OutputImage = UInt8HW | None
DownloadPath = str | None

FORMAT_CHOICES: list[tuple[str, str]] = [
    ("8-bit PNG (preview)", OutputFormat.PNG8.value),
    ("16-bit TIFF (input raw range)", OutputFormat.TIFF16.value),
    ("32-bit float TIFF (normalized 0-1)", OutputFormat.TIFF32.value),
]
DebugText = str


def _upload_to_path(upload: str | Path | None) -> Path:
    if upload is None:
        raise ValueError("Please upload an image first")
    if isinstance(upload, Path):
        path = upload
    else:
        path = Path(str(upload))
    if not path.is_file():
        raise FileNotFoundError(f"uploaded file not found: {path}")
    return path


def _custom_path_text(custom_ckpt: str | None) -> str:
    if custom_ckpt is None:
        return ""
    return custom_ckpt.strip()


def resolve_runtime(
    registry: ModelsRegistry,
    preset_id: str,
    custom_ckpt: str | None,
    custom_family: Family,
) -> tuple[Family, Path, Preset | None]:
    """Preset path unless custom ckpt is non-empty; then family comes from the radio."""
    custom = _custom_path_text(custom_ckpt)
    if custom:
        return custom_family, resolve_ckpt_path(custom, REPO_ROOT), None
    preset = registry.get(preset_id)
    return preset.family, resolve_ckpt_path(preset.ckpt, REPO_ROOT), preset


def build_debug_markdown(
    registry: ModelsRegistry,
    preset_id: str,
    custom_ckpt: str | None,
    custom_family: Family,
) -> str:
    """Inspect checkpoint metadata without requiring a GPU model load."""
    family, ckpt_path, preset = resolve_runtime(registry, preset_id, custom_ckpt, custom_family)
    engine = get_engine()
    cache = engine.cache_info(family, ckpt_path)
    debug: CheckpointDebug = inspect_checkpoint(
        family=family,
        ckpt=ckpt_path,
        param_key_hint=None if preset is None else preset.param_key,
        train_yaml=None if preset is None else preset.train_yaml,
        cache=cache,
        repo_root=REPO_ROOT,
    )
    extra: list[str] = []
    if _custom_path_text(custom_ckpt):
        extra.append(
            "_Custom path overrides the preset checkpoint; family comes from the radio. "
            "SwinIR-M and TSFDAN state_dicts are not interchangeable._"
        )
    elif preset is not None and preset.notes:
        extra.append(f"_notes: {preset.notes}_")
    body = format_debug_markdown(debug)
    if extra:
        return body + "\n\n" + "\n\n".join(extra)
    return body


def denoise_and_debug(
    registry: ModelsRegistry,
    preset_id: str,
    custom_ckpt: str | None,
    custom_family: Family,
    upload: str | Path | None,
    output_format: str = OutputFormat.TIFF16.value,
) -> tuple[OutputImage, OutputImage, DebugText, DownloadPath]:
    """Load (cached) model, denoise, and always refresh the debug panel."""
    original_u8: OutputImage = None
    denoised_u8: OutputImage = None
    try:
        family, ckpt_path, preset = resolve_runtime(registry, preset_id, custom_ckpt, custom_family)
        image_path = _upload_to_path(upload)
        normalization = parse_normalization(None if preset is None else preset.normalization)
        fmt = OutputFormat(output_format)
        image_01: FloatHW = load_input_image(image_path, normalization)
        original_u8 = to_display_u8(image_01)
        output_01 = get_engine().infer(family, ckpt_path, image_01)
        denoised_u8 = to_display_u8(output_01)
        download = str(write_download_file(output_01, image_path, fmt, normalization))
        markdown = build_debug_markdown(registry, preset_id, custom_ckpt, custom_family)
        return original_u8, denoised_u8, markdown, download
    except Exception:
        markdown = build_debug_markdown(registry, preset_id, custom_ckpt, custom_family)
        err = traceback.format_exc()
        if "### Exception" in markdown:
            combined = markdown + "\n\n### Denoise exception\n```\n" + err.rstrip() + "\n```"
        else:
            combined = markdown + "\n\n### Exception\n```\n" + err.rstrip() + "\n```"
        return original_u8, denoised_u8, combined, None


def build_demo() -> gr.Blocks:
    """Construct the Gradio UI (does not launch)."""
    registry = load_registry()
    choices: list[tuple[str, str]] = [(preset.label, preset.id) for preset in registry.presets]
    default_id = registry.default_id

    with gr.Blocks(title="CT Denoise Demo") as demo:
        gr.Markdown(
            "# CT Grayscale Denoise Demo\n"
            "Upload 8-bit PNG/JPEG or uint16 TIFF. RGB uses channel 0 (same as TIFF inference). "
            "SwinIR-M and TSFDAN weights are not interchangeable."
        )
        with gr.Row():
            preset = gr.Dropdown(
                choices=choices,
                value=default_id,
                label="Preset model",
            )
            custom_family = gr.Radio(
                choices=["swinir_m", "tsfdan"],
                value="swinir_m",
                label="Model family (used with a custom checkpoint path)",
            )
        custom_ckpt = gr.Textbox(
            label="Custom checkpoint path (optional; overrides the preset when non-empty)",
            placeholder="/path/to/net_g_latest.pth",
        )
        upload = gr.File(
            label="Upload image",
            file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"],
            type="filepath",
        )
        output_format = gr.Radio(
            choices=FORMAT_CHOICES,
            value=OutputFormat.TIFF16.value,
            label="Download format (the preview below is always 8-bit)",
        )
        with gr.Row():
            denoise_btn = gr.Button("Denoise", variant="primary")
            refresh_btn = gr.Button("Refresh metadata")
        with gr.Row():
            gpu_status = gr.Markdown(value=format_gpu_status_markdown(query_gpu_status()))
            gpu_refresh_btn = gr.Button("Refresh GPU status")
        gpu_timer = gr.Timer(value=5)
        with gr.Row():
            original_out = gr.Image(label="Input", type="numpy", image_mode="L")
            denoised_out = gr.Image(label="Denoised", type="numpy", image_mode="L")
        download_out = gr.File(label="Download denoised image", interactive=False)
        debug = gr.Markdown(value=build_debug_markdown(registry, default_id, "", "swinir_m"))

        def on_refresh(preset_id: str, custom: str, family: Family) -> DebugText:
            return build_debug_markdown(registry, preset_id, custom, family)

        def refresh_gpu_status() -> str:
            return format_gpu_status_markdown(query_gpu_status())

        def on_denoise(
            preset_id: str,
            custom: str,
            family: Family,
            file_path: str | None,
            fmt: str,
        ) -> tuple[NDArray[Any] | None, NDArray[Any] | None, DebugText, DownloadPath]:
            return denoise_and_debug(registry, preset_id, custom, family, file_path, fmt)

        metadata_inputs = [preset, custom_ckpt, custom_family]
        refresh_btn.click(fn=on_refresh, inputs=metadata_inputs, outputs=debug)
        preset.change(fn=on_refresh, inputs=metadata_inputs, outputs=debug)
        custom_ckpt.change(fn=on_refresh, inputs=metadata_inputs, outputs=debug)
        custom_family.change(fn=on_refresh, inputs=metadata_inputs, outputs=debug)
        gpu_refresh_btn.click(fn=refresh_gpu_status, outputs=gpu_status)
        gpu_timer.tick(fn=refresh_gpu_status, outputs=gpu_status)
        denoise_btn.click(
            fn=on_denoise,
            inputs=[preset, custom_ckpt, custom_family, upload, output_format],
            outputs=[original_out, denoised_out, debug, download_out],
        )

    return demo


def main() -> None:
    get_engine().start_idle_reaper(idle_seconds_from_env())
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
