"""IAM-protected Vertex Endpoint adapter for the existing denoise model engine."""

import base64
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from gcs import download, upload
from pydantic import BaseModel

from apps.denoise_demo.engine import OutputFormat, get_engine, load_input_image, write_download_file
from apps.denoise_demo.registry import load_registry, parse_normalization, resolve_ckpt_path

registry = load_registry()
preset = registry.get(registry.default_id)
checkpoint = resolve_ckpt_path(preset.ckpt)
normalization = parse_normalization(preset.normalization)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError("Vertex GPU inference requires CUDA")
    get_engine().get_or_load(preset.family, checkpoint)
    yield


app = FastAPI(lifespan=lifespan)


class Instance(BaseModel):
    input_uri: str | None = None
    image_b64: str | None = None
    output_uri: str | None = None


class PredictRequest(BaseModel):
    instances: list[Instance]


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health():
    return {"status": "ready", "device": "cuda", "preset": preset.id}


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
def predict(request: PredictRequest):
    predictions = []
    for item in request.instances:
        if bool(item.input_uri) == bool(item.image_b64):
            raise HTTPException(status_code=400, detail="Provide exactly one of input_uri and image_b64")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="vertex-denoise-") as directory:
            source = Path(directory) / "input.tif"
            if item.input_uri:
                download(item.input_uri, source)
            else:
                assert item.image_b64 is not None  # Enforced by the mutually exclusive input check above.
                source.write_bytes(base64.b64decode(item.image_b64, validate=True))
            image = load_input_image(source, normalization)
            output = get_engine().infer(preset.family, checkpoint, image)
            result_file = write_download_file(output, source, OutputFormat.TIFF16, normalization)
            prediction = {"shape": list(output.shape), "dtype": "uint16", "preset": preset.id, "device": "cuda"}
            try:
                if item.output_uri:
                    upload(result_file, item.output_uri)
                    prediction["output_uri"] = item.output_uri
                else:
                    prediction["image_b64"] = base64.b64encode(result_file.read_bytes()).decode("ascii")
            finally:
                result_file.unlink()
                result_file.parent.rmdir()
            prediction["seconds"] = round(time.monotonic() - started, 3)
            predictions.append(prediction)
    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("AIP_HTTP_PORT", "8080")))
