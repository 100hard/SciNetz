from __future__ import annotations

import yaml
from pathlib import Path


def test_docker_compose_declares_required_services() -> None:
    compose_path = Path("docker-compose.yaml")
    assert compose_path.exists(), "docker-compose.yaml is required for Phase 0"
    with compose_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    services = data.get("services", {})
    assert {"api", "frontend", "neo4j"}.issubset(set(services.keys()))
