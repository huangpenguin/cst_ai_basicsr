# TensorBoard serve: TF image missing CLI + setuptools 82

## Symptom

`bash scripts/serve_tensorboard.sh` failed with `exec: "tensorboard": executable file not found in $PATH`.
`uv run tensorboard` failed with `ModuleNotFoundError: No module named 'pkg_resources'`.

## Root cause

1. `tensorflow/tensorflow:latest` no longer installs the `tensorboard` package/CLI.
2. `setuptools>=82` removed `pkg_resources`; `tb-nightly` / `tensorboard` still import it at startup.

## Fix

- Prefer project venv TensorBoard; Docker fallback uses `python:3.11-slim` + pip install.
- Pin `setuptools>=70,<82` in `pyproject.toml`.

## Prevention

Do not treat the TensorFlow Docker image as a TensorBoard runtime. Keep a setuptools upper bound while tensorboard still depends on `pkg_resources`.
