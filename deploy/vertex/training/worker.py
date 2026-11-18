"""Instrument the existing BasicSR training loop without replacing its optimizer.

The wrapper runs inside each torchrun worker and records actual allocator peaks,
optimizer updates, losses and complete trainable-weight fingerprints. It never
changes the model's numerical operations or claims batch-size equivalence.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path


def fingerprint(model):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.get_bare_model(model.net_g).named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
        count += value.numel()
    return digest.hexdigest(), count


def main():
    import torch

    import basicsr.train as training

    rank = int(os.environ.get("RANK", "0"))
    output = Path(os.environ["VERTEX_TELEMETRY_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"worker-{rank}.json"
    record = {
        "rank": rank,
        "status": "RUNNING",
        "started_unix": time.time(),
        "optimizer_updates": 0,
        "losses": [],
        "training_peak_allocated_bytes": 0,
        "training_peak_reserved_bytes": 0,
        "validation_peak_allocated_bytes": 0,
        "validation_peak_reserved_bytes": 0,
    }
    built = []

    def write():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    original_build = training.build_model

    def build(opt):
        model = original_build(opt)
        built.append(model)
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        record.update(
            gpu_name=properties.name,
            gpu_total_bytes=properties.total_memory,
            torch_version=torch.__version__,
            precision="float32",
            crop_size=opt["datasets"]["train"]["gt_size"],
            batch_per_gpu=opt["datasets"]["train"]["batch_size_per_gpu"],
            world_size=opt["world_size"],
        )
        before, count = fingerprint(model)
        record.update(initial_weights_sha256=before, trainable_parameters=count)
        optimize = model.optimize_parameters

        def wrapped_optimize(current_iter):
            torch.cuda.reset_peak_memory_stats(device)
            optimize(current_iter)
            torch.cuda.synchronize(device)
            record["optimizer_updates"] += 1
            record["last_optimizer_iteration"] = current_iter
            for key, getter in [
                ("allocated", torch.cuda.max_memory_allocated),
                ("reserved", torch.cuda.max_memory_reserved),
            ]:
                name = f"training_peak_{key}_bytes"
                record[name] = max(record[name], getter(device))
            losses = {key: float(value) for key, value in model.get_current_log().items()}
            if not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError("Non-finite training loss")
            if len(record["losses"]) < 32:
                record["losses"].append({"iteration": current_iter, **losses})
            record["last_loss"] = losses
            write()

        model.optimize_parameters = wrapped_optimize
        validation = model.validation

        def wrapped_validation(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats(device)
            result = validation(*args, **kwargs)
            torch.cuda.synchronize(device)
            for key, getter in [
                ("allocated", torch.cuda.max_memory_allocated),
                ("reserved", torch.cuda.max_memory_reserved),
            ]:
                name = f"validation_peak_{key}_bytes"
                record[name] = max(record[name], getter(device))
            record["validation_calls"] = record.get("validation_calls", 0) + 1
            if getattr(model, "metric_results", None):
                metrics = {key: float(value) for key, value in model.metric_results.items()}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("Non-finite validation metric")
                record["last_validation_metrics"] = metrics
            write()
            return result

        model.validation = wrapped_validation
        write()
        return model

    training.build_model = build
    try:
        training.train_pipeline("/workspace")
        record["status"] = "SUCCEEDED"
    except BaseException as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error)[:2000])
        raise
    finally:
        if built:
            try:
                after, _ = fingerprint(built[0])
                record.update(final_weights_sha256=after, weights_changed=after != record["initial_weights_sha256"])
            except Exception as error:
                record["fingerprint_error"] = type(error).__name__
        record["elapsed_seconds"] = time.time() - record["started_unix"]
        write()


if __name__ == "__main__":
    main()
