FROM asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex@sha256:fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f
COPY --chown=app:app . /opt/vertex/training
CMD ["python", "/opt/vertex/training/train_full.py", "--help"]
