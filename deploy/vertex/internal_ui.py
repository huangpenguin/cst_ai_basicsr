"""LAN-only, CPU frontend; all model inference runs on Vertex AI."""

import base64
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from PIL import Image

from apps.denoise_demo.checkpoint_info import inspect_preset
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
        if getattr(image, "n_frames", 1) != 1:
            raise gr.Error("Multi-page images need an explicit slice selection. Upload a single image.")
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


def image_details(path):
    if not path:
        return {}
    with Image.open(path) as image:
        array = np.asarray(image)
        return {
            "filename": Path(path).name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "pages": getattr(image, "n_frames", 1),
            "mode": image.mode,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "file_bytes": Path(path).stat().st_size,
            "pixel_min": float(array.min()),
            "pixel_max": float(array.max()),
            "pixel_mean": float(array.mean()),
            "preview": "8-bit display only; original input pixels are preserved",
        }


def inspect_upload(upload, preset_id):
    try:
        return (preview(upload, preset_id) if upload else None, image_details(upload), None, None, {}, "")
    except (OSError, ValueError) as error:
        raise gr.Error("Cannot read this image. Upload a valid PNG, JPEG or single-page TIFF.") from error


def checkpoint_details(preset_id):
    info = inspect_preset(REGISTRY.get(preset_id))
    result = info.model_dump(exclude={"cache", "error"})
    if result.get("train_yaml"):
        result["train_yaml"]["source_path"] = result["train_yaml"].pop("path")
    result["source"] = "Read-only catalogue checkpoint in the frontend image; cloud deployment is shown separately."
    result["inspection_status"] = "Some catalogue metadata could not be read" if info.error else "OK"
    return result


def cloud_details():
    """Read control-plane configuration and sampled replicas; never calls predict."""
    now = dt.datetime.now(dt.timezone.utc)
    # Derive resource identity from the configured dedicated prediction URL.
    resource = PREDICT_URL.split("/v1/", 1)[1].removesuffix(":predict")
    parts = resource.split("/")
    region, endpoint_id = parts[3], parts[5]
    result = {
        "frontend": "work1 CPU frontend; local GPU access disabled",
        "region": region,
        "endpoint_id": endpoint_id,
        "checked_utc": now.isoformat(),
        "prediction_requests_sent": 0,
        "prediction_health": "Not tested by status refresh; control-plane messages are not prediction results.",
        "gpu_memory_and_processes": "Not exposed by this endpoint; not inferred from work1 GPUs.",
    }
    try:
        auth = {"Authorization": "Bearer " + token(), "x-goog-user-project": PROJECT}
        response = requests.get(
            f"https://{region}-aiplatform.googleapis.com/v1beta1/{resource}", headers=auth, timeout=30
        )
        if not response.ok:
            return {**result, "status": f"Endpoint status unavailable (HTTP {response.status_code})"}
        endpoint = response.json()
        deployments = []
        for model in endpoint.get("deployedModels", []):
            resources = model.get("dedicatedResources", {})
            deployments.append(
                {
                    "deployed_model_id": model["id"],
                    "model": model.get("model"),
                    "display_name": model.get("displayName"),
                    "control_plane_status": model.get("status", {}),
                    "machine": resources.get("machineSpec"),
                    "min_replicas": resources.get("minReplicaCount", 0),
                    "max_replicas": resources.get("maxReplicaCount"),
                    "scale_to_zero": resources.get("scaleToZeroSpec"),
                }
            )
        result.update(status="Endpoint configuration read successfully", deployments=deployments)
        filt = (
            'metric.type="aiplatform.googleapis.com/prediction/online/replicas" '
            'AND resource.type="aiplatform.googleapis.com/Endpoint" '
            f'AND resource.labels.location="{region}" '
            f'AND resource.labels.endpoint_id="{endpoint_id}"'
        )
        metrics = requests.get(
            f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/timeSeries",
            headers=auth,
            params={
                "filter": filt,
                "interval.startTime": (now - dt.timedelta(minutes=20)).isoformat(),
                "interval.endTime": now.isoformat(),
                "pageSize": 100,
            },
            timeout=30,
        )
        if not metrics.ok:
            result["replica_status"] = f"Monitoring unavailable (HTTP {metrics.status_code}); replica count unknown"
        else:
            samples = []
            for series in metrics.json().get("timeSeries", []):
                points = series.get("points", [])
                if not points:
                    continue
                point = max(points, key=lambda item: item["interval"]["endTime"])
                stamp = point["interval"]["endTime"]
                age = (now - dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds()
                samples.append(
                    {
                        "deployed_model_id": series["metric"].get("labels", {}).get("deployed_model_id"),
                        "replicas": point["value"],
                        "sample_utc": stamp,
                        "sample_age_seconds": round(age),
                        "stale": age > 300,
                    }
                )
            result["replica_samples"] = samples
            result["replica_status"] = (
                "Sampled monitoring data, not an instantaneous measurement"
                if samples
                else "No recent samples; replica count unknown"
            )
        return result
    except (requests.RequestException, OSError, ValueError):
        return {**result, "status": "Cloud status request failed; check server connectivity or ADC login."}


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


with gr.Blocks(title="CT Denoise", delete_cache=(600, 3600)) as demo:
    gr.Markdown(
        "# CT Grayscale Denoise\n"
        "Upload 8-bit PNG/JPEG or uint16 TIFF. RGB uses channel 0. "
        "Inference runs on the Vertex AI GPU; the first request after idle requires a cold start.\n\n"
        "[Submit scans and reference images for review](http://192.168.3.11:7862)"
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
    with gr.Accordion("Model and checkpoint details", open=False):
        metadata_refresh = gr.Button("Refresh model metadata")
        metadata = gr.JSON(label="Checkpoint metadata (catalogue copy)")
    with gr.Accordion("Vertex GPU status", open=True):
        gr.Markdown(
            "The frontend runs on CPU. Refresh reads cloud status without waking the GPU or testing predictions. "
            "An idle GPU may need several minutes to start."
        )
        gpu_refresh = gr.Button("Refresh Vertex GPU status")
        gpu_info = gr.JSON(
            label="Cloud GPU and replica details",
            value={"status": "Click Refresh Vertex GPU status to read cloud information."},
        )
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
    with gr.Accordion("Image details", open=True):
        with gr.Row():
            input_info = gr.JSON(label="Input image details")
            output_info = gr.JSON(label="Output image details")
    metadata_refresh.click(checkpoint_details, inputs=preset, outputs=metadata, api_name="checkpoint_details")
    gpu_refresh.click(cloud_details, outputs=gpu_info, api_name="cloud_details")
    upload.change(
        inspect_upload,
        inputs=[upload, preset],
        outputs=[original, input_info, output, download, output_info, status],
        api_name="inspect_image",
    )
    # Gradio attaches event methods dynamically; exercised by the UI construction check.
    preset.change(model_details, inputs=preset, outputs=[details, submit], api_name="model_details").then(
        checkpoint_details, inputs=preset, outputs=metadata, api_name=False
    )  # pyright: ignore[reportAttributeAccessIssue]
    submit.click(infer, inputs=[preset, upload], outputs=[original, output, download, status], api_name="infer").then(
        image_details, inputs=download, outputs=output_info, api_name=False
    )  # pyright: ignore[reportAttributeAccessIssue]

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, share=False)
