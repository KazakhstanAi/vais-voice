# Core image. Set matched Torch version and wheel index only after host validation.
FROM python:3.11-slim-bookworm
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=""
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install ".[dev]"
RUN if [ -n "${TORCH_VERSION}" ]; then python -m pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"; fi
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
RUN useradd --create-home --uid 10001 researcher && mkdir -p data/raw data/processed data/manifests/local runs && chown -R researcher:researcher /workspace
USER researcher
CMD ["python", "-m", "vais_voice.doctor"]
