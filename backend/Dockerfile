# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml ./
COPY backend ./backend

RUN pip install .

COPY config.yaml ./config.yaml

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
