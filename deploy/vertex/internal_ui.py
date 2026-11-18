"""LAN-only, CPU frontend; all model inference runs on Vertex AI."""

import base64
import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from PIL import Image

from apps.denoise_demo.registry import load_registry, parse_normalization

PROJECT = os.environ.get("VERTEX_PROJECT", "hallowed-pipe-497802-i0")
PREDICT_URL = os.environ["VERTEX_PREDICT_URL"]
ADC = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
# This is the model actually deployed by the current single-preset Vertex image.
# The full image registry is a catalogue, not evidence of cloud availability.
VERTEX_PRESET_ID = "swinir_cylinderhead_mean3_20261104_173035"
REGISTRY = load_registry()
REGISTRY.get(VERTEX_PRESET_ID)
_token = ""
_expires = 0.0


def token():
    global _token, _expires
    if time.monotonic() >= _expires:
        credential = json.loads(ADC.read_text())
        if credential.get("type") != "authorized_user":
            raise RuntimeError("This internal client expects the existing user ADC login")
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "refresh_token",
                "client_id": credential["client_id"],
                "client_secret": credential["client_secret"],
                "refresh_token": credential["refresh_token"],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        _token = result["access_token"]
        _expires = time.monotonic() + int(result["expires_in"]) - 60
    return _token


def preview(path, preset_id):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.uint8:
        return array
    if array.dtype != np.uint16:
        raise gr.Error("Expected an 8-bit PNG/JPEG or a 16-bit TIFF image.")
    normalization = parse_normalization(REGISTRY.get(preset_id).normalization)
    scaled = (array.astype(np.float32) - normalization.raw_min) / (normalization.raw_max - normalization.raw_min)
    return (np.clip(scaled, 0, 1) * 255).round().astype(np.uint8)


def model_details(preset_id):
    preset = REGISTRY.get(preset_id)
    available = preset_id == VERTEX_PRESET_ID
    availability = (
        "Deployed to Vertex AI. Inference runs on the cloud GPU."
        if available
        else "Not deployed to the current Vertex endpoint. This preset cannot run here yet."
    )
    return (
        f"**Model family:** `{preset.family}`  \n"
        f"**Preset:** `{preset.id}`  \n"
        f"**Vertex status:** {availability}" + (f"\n\n{preset.notes}" if preset.notes else "")
    ), gr.update(interactive=available)


def infer(preset_id, upload, progress=gr.Progress()):
    REGISTRY.get(preset_id)
    if preset_id != VERTEX_PRESET_ID:
        raise gr.Error(
            "This preset is not deployed to Vertex AI yet. Select the Cylinderhead mean3 FINAL step 30000 preset."
        )
    if not upload:
        raise gr.Error("Please upload an image first.")
    original_preview = preview(upload, preset_id)
    started = time.monotonic()
    payload = {"instances": [{"image_b64": base64.b64encode(Path(upload).read_bytes()).decode("ascii")}]}
    if len(json.dumps(payload).encode()) > 10_000_000:
        raise gr.Error("The Vertex request limit is 10 MB including base64 encoding. Use a smaller image.")
    progress(0, desc="Calling the Vertex GPU. The first request after idle requires a cold start.")
    for attempt in range(81):
        response = requests.post(
            PREDICT_URL,
            json=payload,
            headers={"Authorization": "Bearer " + token(), "x-goog-user-project": PROJECT},
            timeout=(30, 300),
        )
        if response.status_code != 429:
            break
        if attempt == 80:
            raise gr.Error("Vertex is not ready yet. Please try again later.")
        progress(0, desc=f"Waiting for the Vertex GPU ({int(time.monotonic() - started)} seconds elapsed).")
        time.sleep(15)
    if not response.ok:
        raise gr.Error(f"Vertex request failed: HTTP {response.status_code}, {response.text[:600]}")
    prediction = response.json()["predictions"][0]
    if prediction["preset"] != preset_id:
        raise gr.Error("Vertex returned a different model from the selected preset. Check the endpoint deployment.")
    destination = Path(tempfile.mkdtemp(prefix="vertex-ui-")) / "denoised.tif"
    destination.write_bytes(base64.b64decode(prediction["image_b64"], validate=True))
    progress(1, desc="Cloud inference complete.")
    return (
        original_preview,
        preview(destination, preset_id),
        str(destination),
        (
            f"Vertex device: {prediction['device']}; model: {prediction['preset']}; "
            f"cloud processing: {prediction['seconds']} s; total: {time.monotonic() - started:.1f} s."
        ),
    )


with gr.Blocks(title="CT Denoise Demo", delete_cache=(600, 3600)) as demo:
    gr.Markdown(
        "# CT Grayscale Denoise Demo\n"
        "Upload 8-bit PNG/JPEG or uint16 TIFF. RGB uses channel 0. "
        "Inference runs on the Vertex AI GPU; the first request after idle requires a cold start."
    )
    preset = gr.Dropdown(
        choices=[
            (p.label + ("" if p.id == VERTEX_PRESET_ID else " [not deployed to Vertex]"), p.id)
            for p in REGISTRY.presets
        ],
        value=VERTEX_PRESET_ID,
        label="Preset model",
    )
    details = gr.Markdown(value=model_details(VERTEX_PRESET_ID)[0])
    upload = gr.File(
        label="Upload image",
        file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"],
        type="filepath",
    )
    submit = gr.Button("Denoise", variant="primary")
    with gr.Row():
        original = gr.Image(label="Input", image_mode="L")
        output = gr.Image(label="Denoised", image_mode="L")
    download = gr.File(label="Download denoised image (16-bit TIFF)", interactive=False)
    status = gr.Markdown()
    # Gradio attaches event methods dynamically; exercised by the UI construction check.
    preset.change(model_details, inputs=preset, outputs=[details, submit], api_name="model_details")  # pyright: ignore[reportAttributeAccessIssue]
    submit.click(infer, inputs=[preset, upload], outputs=[original, output, download, status], api_name="infer")  # pyright: ignore[reportAttributeAccessIssue]

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, share=False)
