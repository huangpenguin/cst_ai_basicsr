"""Publish a timestamped checkpoint and update this run's managed Demo preset."""

import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = Path("/mnt/home/huang/cst_ai/experiments/cylinderhead_300kv_mean3_20261104_173035_JST")
MODELS = RUN / "artifacts/train_SwinIR_cylinderhead_mean3_raw_P128_bs4x4_i30k/models"
REGISTRY = Path("/mnt/home/huang/code/cst_ai/BasicSR/apps/denoise_demo/models.yaml")
PUBLISHED = Path("/mnt/home/huang/cst_ai/demo_models")
STATE = RUN / "resource_review"
STATE.mkdir(exist_ok=True)
JST = timezone(timedelta(hours=9))
MODEL_ID = "swinir_cylinderhead_mean3_20261104_173035"
DOCKER = "/usr/bin/docker"


def save(name, value):
    temp = STATE / (name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(STATE / name)


def register(final=False):
    if final:
        source = MODELS / "net_g_latest.pth"
        step = 30000
        stage = "final"
    else:
        candidates = [
            p
            for p in MODELS.glob("net_g_*.pth")
            if p.stem.rsplit("_", 1)[-1].isdigit()
            and (MODELS.parent / "training_states" / (p.stem.rsplit("_", 1)[-1] + ".state")).is_file()
        ]
        source = max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
        step = int(source.stem.rsplit("_", 1)[-1])
        stage = "training"
    assert source.is_file() and source.stat().st_size > 1000000
    stamp = datetime.fromtimestamp(source.stat().st_mtime, JST)
    PUBLISHED.mkdir(exist_ok=True)
    target = PUBLISHED / f"SwinIR_Cylinderhead_mean3_{stamp:%Y%m%d_%H%M%S}_JST_{stage}_i{step}.pth"
    if not target.exists():
        original_stat = source.stat()
        temp = target.with_suffix(".pth.partial")
        shutil.copy2(source, temp)
        assert source.stat().st_size == original_stat.st_size and source.stat().st_mtime_ns == original_stat.st_mtime_ns
        assert temp.stat().st_size == original_stat.st_size
        temp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    stage_label = "FINAL" if final else "训练中"
    entry = {
        "id": MODEL_ID,
        "label": f"[swinir_m] Cylinderhead mean3 | {stamp:%Y-%m-%d %H:%M} JST | {stage_label} step {step}",
        "family": "swinir_m",
        "ckpt": str(target),
        "param_key": "params_ema",
        "train_yaml": str(RUN / "control/train.yml"),
        "notes": (
            f"{stage_label}; run 20261104_173035 JST; three 200uA scans averaged; source pixels "
            f"unchanged; independent test reserved."
        ),
    }
    begin = "  # BEGIN managed cylinderhead 20261104_173035"
    end = "  # END managed cylinderhead 20261104_173035"
    block = (
        begin
        + "\n"
        + "\n".join(
            ("  - " if i == 0 else "    ") + k + ": " + json.dumps(v, ensure_ascii=False)
            for i, (k, v) in enumerate(entry.items())
        )
        + "\n"
        + end
    )
    with (STATE / "registry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = REGISTRY.read_text()
        if begin in original:
            updated = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda _: block, original, flags=re.S)
        else:
            updated = original.rstrip() + "\n\n" + block + "\n"
        backup = STATE / f"models_registry_before_{stage}_{step}.yaml"
        if not backup.exists():
            backup.write_text(original)
        if updated != original:
            assert REGISTRY.read_text() == original, "Registry changed concurrently"
            temporary = REGISTRY.with_name("models.yaml.cylinderhead.tmp")
            temporary.write_text(updated)
            temporary.replace(REGISTRY)
    code = (
        "import sys,json;sys.path.insert(0,'/workspace');"
        "from apps.denoise_demo.registry import load_registry,resolve_ckpt_path;"
        f"p=load_registry().get({MODEL_ID!r});assert resolve_ckpt_path(p.ckpt).is_file();"
        "print(json.dumps({'id':p.id,'label':p.label,'exists':True},ensure_ascii=False))"
    )
    subprocess.run([DOCKER, "exec", "denoise-demo", "/opt/basicsr-venv/bin/python", "-c", code], check=True)
    subprocess.run([DOCKER, "restart", "denoise-demo"], check=True)
    for _ in range(90):
        try:
            with urllib.request.urlopen("http://127.0.0.1:7860/config", timeout=5) as response:
                config = json.load(response)
            assert MODEL_ID in json.dumps(config, ensure_ascii=False)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("Demo did not return the updated model list")
    result = {
        "phase": "final_registered" if final else "intermediate_registered",
        "updated_at": datetime.now(JST).isoformat(),
        "entry": entry,
        "source_checkpoint": str(source),
        "published_sha256": digest,
        "demo_http": "http://192.168.3.11:7860/",
        "test_LQ": "/mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST/test_holdout/LQ",
        "test_GT": "/mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST/test_holdout/GT",
        "test_preview_results": str(RUN / "test_preview"),
    }
    save("demo_model_status.json", result)
    if final:
        save("completion_notice.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def watch():
    save(
        "model_watcher_status.json",
        {"phase": "waiting_for_training_and_test", "started_at": datetime.now(JST).isoformat()},
    )
    last = None
    while True:
        state = json.loads((RUN / "pipeline_status.json").read_text())
        if state["phase"] == "complete":
            register(final=True)
            save("model_watcher_status.json", {"phase": "complete", "completed_at": datetime.now(JST).isoformat()})
            return
        if state["phase"] == "failed" and last != "failed":
            save("model_watcher_status.json", {"phase": "training_or_test_failed", "pipeline_state": state})
        last = state["phase"]
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        watch()
    else:
        register()
