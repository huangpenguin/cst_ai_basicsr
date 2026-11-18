FROM ai-coding-rules-mlops:smoke

ENV UV_PROJECT_ENVIRONMENT=/opt/basicsr-venv \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HOME=/tmp

WORKDIR /opt/basicsr-manifest
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --dev --python /opt/conda/bin/python

WORKDIR /workspace
USER 1003:1003
EXPOSE 7860
CMD ["/opt/basicsr-venv/bin/python", "apps/denoise_demo/app.py"]
