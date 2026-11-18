# Customer Gradio image.
# Stack stays on the validated BasicSR combination (PyTorch 2.6 + CUDA 12.4).
# lab-docker's dl-gpu image is a different stack (Python 3.14, torch 2.9.1+cu126)
# and is not the base here. What is borrowed from lab-docker:
# tini as PID 1, a non-root runtime user, uv, and an entrypoint that maps
# NVIDIA device groups before dropping privileges.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    PATH="/workspace/.venv/bin:${PATH}" \
    GRADIO_ANALYTICS_ENABLED=False \
    TZ=Asia/Tokyo \
    LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        passwd \
        tini \
        util-linux \
    && ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && echo Asia/Tokyo > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash app

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python /opt/conda/bin/python

COPY basicsr ./basicsr
COPY apps ./apps
COPY scripts/infer ./scripts/infer
COPY options ./options
COPY experiments/pretrained_models ./experiments/pretrained_models
COPY deploy/serve/models.yaml ./apps/denoise_demo/models.yaml
COPY deploy/serve/weights/ /opt/models/
COPY deploy/serve/entrypoint.sh /usr/local/bin/serve-entrypoint

RUN chmod 755 /usr/local/bin/serve-entrypoint \
    && chown -R app:app /workspace /opt/models /home/app

EXPOSE 7860

# Start as root so the entrypoint can map GPU device groups, then drop to app.
ENTRYPOINT ["tini", "--", "/usr/local/bin/serve-entrypoint"]
CMD ["python", "apps/denoise_demo/app.py"]
