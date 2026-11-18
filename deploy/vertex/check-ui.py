"""One real Gradio->Vertex->download acceptance case. Small must pass before full."""

import argparse
import datetime as dt
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import requests
from gradio_client import Client, handle_file
from PIL import Image

PRESET = "swinir_cylinderhead_mean3_20261104_173035"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--url", required=True)
p.add_argument("--case", choices=["small", "full"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--input", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
record = {
    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "case": a.case,
    "url": a.url,
    "client_physical_host": "work1 (same host as frontend)",
    "status": "FAIL",
}
started = time.monotonic()
try:
    response = requests.get(a.url + "/config", timeout=30)
    response.raise_for_status()
    config = response.json()
    dropdown = next(c["props"] for c in config["components"] if c.get("type") == "dropdown")
    choices = dropdown["choices"]
    assert len(choices) == 11, len(choices)
    record.update(preset_count=len(choices), page_title=config.get("title"), choices=choices)
    assert dropdown["label"] == "Preset model"
    client = Client(a.url, verbose=False)
    if a.case == "small":
        for _label, identifier in choices:
            endpoint = a.url + "/gradio_api/call/model_details"
            event = requests.post(endpoint, json={"data": [identifier]}, timeout=30)
            event.raise_for_status()
            completed = requests.get(endpoint + "/" + event.json()["event_id"], timeout=30)
            completed.raise_for_status()
            block = next(part for part in completed.text.split("\n\n") if part.startswith("event: complete\n"))
            values = json.loads(next(line[6:] for line in block.splitlines() if line.startswith("data: ")))
            update = values[1]
            assert update["interactive"] == (identifier == PRESET), (identifier, update)
        record["disabled_unavailable_models"] = 10
        source = a.output / "input-small.png"
        Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)).save(source)
    else:
        assert json.loads((a.output / "small.json").read_text())["status"] == "PASS", "Small case must pass first"
        assert a.input and a.input.is_file()
        source = a.input
    with Image.open(source) as image:
        expected_size = image.size
    record.update(input=str(source), expected_size=expected_size)
    result = client.predict(PRESET, handle_file(str(source)), api_name="/infer")
    downloaded = Path(result[2])
    target = a.output / ("output-" + a.case + ".tif")
    shutil.copyfile(downloaded, target)
    with Image.open(target) as image:
        array = np.asarray(image)
        assert image.format == "TIFF"
        assert image.size == expected_size, (image.size, expected_size)
        assert array.dtype == np.uint16, str(array.dtype)
        record.update(output_size=image.size, output_dtype=str(array.dtype), output_bytes=target.stat().st_size)
    assert PRESET in result[3] and "cuda" in result[3].lower(), result[3]
    record.update(status="PASS", vertex_status=result[3], output_file=target.name)
except Exception as error:
    record.update(error_type=type(error).__name__, error=str(error)[:2500])
    traceback.print_exc()
finally:
    record.update(
        elapsed_seconds=round(time.monotonic() - started, 3), completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    )
    (a.output / (a.case + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
if record["status"] != "PASS":
    raise SystemExit(1)
