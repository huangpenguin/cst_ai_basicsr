FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Shared by Dev Container (docker-compose) and GitLab CI (MLOPS_PROJECT_IMAGE).
# Python deps are NOT baked here — postCreate / CI run scripts/env/uv-bootstrap.sh → .venv
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    TZ=Asia/Shanghai

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime \
    && echo "${TZ}" > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
        sudo \
        tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" --shell /bin/bash \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && mkdir -p /workspace /data \
    && chown "${USERNAME}:${USERNAME}" /workspace /data

WORKDIR /workspace

ENTRYPOINT ["tini", "--"]
CMD ["bash"]
