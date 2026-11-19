FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=""
WORKDIR /app
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock && mkdir -p /app/apps && touch /app/apps/__init__.py
COPY . /app/apps/xray_portal/
USER 1003:1003
CMD ["uvicorn", "apps.xray_portal.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
