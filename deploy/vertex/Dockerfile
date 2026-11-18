ARG BASE_IMAGE=denoise-demo:local
FROM ${BASE_IMAGE}

# Reuse the validated PyTorch 2.6/cu124 image and its baked model registry.
COPY --chown=app:app deploy/vertex /opt/vertex
ENV PYTHONPATH=/workspace:/opt/vertex
EXPOSE 8080
CMD ["python", "/opt/vertex/serve.py"]
