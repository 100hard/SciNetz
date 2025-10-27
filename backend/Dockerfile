# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements
COPY scripts ./scripts

RUN python scripts/check_package_index.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements/dev.txt \
    && python -m spacy download en_core_web_sm

COPY pyproject.toml ./
COPY backend ./backend
COPY config.yaml ./config.yaml
COPY docker-compose.yaml ./docker-compose.yaml

RUN --mount=type=cache,target=/root/.cache/huggingface \
    SCINETS_SKIP_ENV_FILE=1 python - <<'PY'
from backend.app.config import load_config
from backend.app.parsing.pipeline import ParsingPipeline

config = load_config()
ParsingPipeline(config=config)
PY

COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "--factory", "backend.app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
