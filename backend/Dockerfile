# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements ./requirements
COPY scripts ./scripts

RUN python scripts/check_package_index.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements/dev.txt

COPY pyproject.toml ./
COPY backend ./backend
COPY tests ./tests
COPY config.yaml ./config.yaml
COPY docker-compose.yaml ./docker-compose.yaml

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
