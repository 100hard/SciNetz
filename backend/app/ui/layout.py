"""Utility functions for computing graph layouts server-side."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NodeLayoutInfo:
    """Metadata used to generate human-friendly graph layouts."""

    node_id: str
    times_seen: int
    node_type: Optional[str]
    section_distribution: Mapping[str, int]


@dataclass(frozen=True)
class LayoutResult:
    """Container for layout coordinates and semantic metadata."""

    positions: Dict[str, Tuple[float, float]]
    rings: Dict[str, int]
    importance: Dict[str, float]


CORE_TYPES = {
    "method",
    "model",
    "approach",
    "task",
    "architecture",
    "algorithm",
}
SUPPORT_TYPES = {
    "component",
    "module",
    "parameter",
    "hyperparameter",
    "protein",
    "gene",
    "sequence",
}
DATA_TYPES = {"dataset", "data", "corpus", "benchmark", "experiment"}
METRIC_TYPES = {"metric", "score", "measure", "accuracy", "loss"}
RESULT_SECTION_HINTS = {"results", "evaluation", "conclusion", "discussion"}
METHOD_SECTION_HINTS = {"methods", "approach", "implementation"}

INITIAL_POSITION_JITTER = 0.03
INITIAL_RADIUS_SCALE = 0.85


def _hash_to_unit(value: str, salt: str) -> float:
    seed = f"{salt}:{value}"
    hash_val = 0
    for char in seed:
        hash_val = ((hash_val * 31) + ord(char)) & 0xFFFFFFFF
    return hash_val / 0xFFFFFFFF if hash_val else 0.0


def compute_node_positions(
    node_infos: Mapping[str, NodeLayoutInfo],
    edges: Iterable[Tuple[str, str]],
) -> LayoutResult:
    """Compute lightweight 2D seed positions for the frontend physics layout.

    We keep every node very close to the origin and rely on the client-side
    ForceAtlas2 pass to generate the final organic arrangement. Importance
    scores remain useful for sizing even though the coordinates no longer carry
    semantic ring information.

    Args:
        node_infos: Mapping of node identifiers to their layout metadata.
        edges: Iterable of directed edges expressed as ``(source, target)`` tuples.

    Returns:
        LayoutResult containing coordinates, ring assignments, and importance values.
    """

    if not node_infos:
        return LayoutResult(positions={}, rings={}, importance={})

    node_ids: List[str] = list(node_infos.keys())
    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    for src, dst in edges:
        if src in adjacency and dst in adjacency:
            adjacency[src].append(dst)
            adjacency[dst].append(src)

    degrees: Dict[str, int] = {node_id: len(neighbours) for node_id, neighbours in adjacency.items()}

    def _importance_score(info: NodeLayoutInfo) -> float:
        degree = degrees.get(info.node_id, 0)
        return float(info.times_seen) + 2.5 * degree

    scores: Dict[str, float] = {node_id: _importance_score(info) for node_id, info in node_infos.items()}
    sorted_nodes = sorted(node_ids, key=lambda nid: scores[nid], reverse=True)
    core_count = max(1, min(3, max(round(len(node_ids) * 0.08), 1)))
    core_nodes = set(sorted_nodes[:core_count])

    rings: MutableMapping[str, int] = {}

    def _normalise_type(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.strip().lower() or None

    def _dominant_section(info: NodeLayoutInfo) -> Optional[str]:
        if not info.section_distribution:
            return None
        dominant = max(info.section_distribution.items(), key=lambda item: item[1])[0]
        return dominant.lower()

    for node_id, info in node_infos.items():
        if node_id in core_nodes:
            rings[node_id] = 0
            continue
        node_type = _normalise_type(info.node_type)
        section = _dominant_section(info)
        if node_type in CORE_TYPES:
            rings[node_id] = 1
        elif node_type in SUPPORT_TYPES or (section in METHOD_SECTION_HINTS):
            rings[node_id] = 2
        elif node_type in DATA_TYPES:
            rings[node_id] = 3
        elif node_type in METRIC_TYPES or section in RESULT_SECTION_HINTS:
            rings[node_id] = 4
        else:
            rings[node_id] = 5

    max_ring = max(rings.values(), default=0)
    if max_ring > 0:
        for node_id, ring in list(rings.items()):
            if ring > max_ring:
                rings[node_id] = max_ring

    positions: Dict[str, Tuple[float, float]] = {}
    for node_id in node_ids:
        seed_x = _hash_to_unit(node_id, "initial-x")
        seed_y = _hash_to_unit(node_id, "initial-y")
        radius_seed = _hash_to_unit(node_id, "initial-radius")
        radius = INITIAL_POSITION_JITTER * (0.2 + radius_seed * INITIAL_RADIUS_SCALE)
        angle = 2.0 * math.pi * seed_x
        jitter_x = math.cos(angle) * radius
        jitter_y = math.sin(angle) * radius
        # shift y using independent seed to avoid perfect symmetry
        jitter_y += (seed_y - 0.5) * INITIAL_POSITION_JITTER * 0.3
        positions[node_id] = (jitter_x, jitter_y)

    xs = [coord[0] for coord in positions.values()]
    ys = [coord[1] for coord in positions.values()]
    min_x = min(xs, default=0.0)
    max_x = max(xs, default=0.0)
    min_y = min(ys, default=0.0)
    max_y = max(ys, default=0.0)
    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 0.0)

    if span <= 0.5:
        normalised = dict(positions)
    else:
        scale = 2.0 / max(span, 1e-6)
        normalised = {}
        for node_id, (x, y) in positions.items():
            nx = max(min((x - centre_x) * scale, 1.0), -1.0)
            ny = max(min((y - centre_y) * scale, 1.0), -1.0)
            normalised[node_id] = (nx, ny)

    max_score = max(scores.values(), default=1.0)
    importance: Dict[str, float] = {}
    for node_id in node_ids:
        importance[node_id] = scores[node_id] / max_score if max_score else 0.0

    return LayoutResult(positions=normalised, rings=dict(rings), importance=importance)
