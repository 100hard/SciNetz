"""Utility functions for computing graph layouts server-side."""
from __future__ import annotations

import math
from collections import defaultdict, deque
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


def compute_node_positions(
    node_infos: Mapping[str, NodeLayoutInfo],
    edges: Iterable[Tuple[str, str]],
) -> LayoutResult:
    """Compute deterministic 2D positions for nodes with semantic cues.

    The layout emphasises central hypotheses or methods near the origin, with
    successive rings conveying supportive components, evidence, and contextual
    entities. Importance scores are normalised to ``[0.0, 1.0]`` for downstream
    visual emphasis.

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

    def _place_core_nodes() -> None:
        core_list = [node for node in sorted_nodes if node in core_nodes]
        if not core_list:
            return
        positions[core_list[0]] = (0.0, 0.0)
        if len(core_list) == 1:
            return
        radius = 0.22
        for index, node_id in enumerate(core_list[1:], start=1):
            angle = 2 * math.pi * (index - 1) / max(len(core_list) - 1, 1)
            positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    _place_core_nodes()

    branch_roots: Dict[str, str] = {}
    for core_id in sorted(core_nodes):
        neighbours = sorted(neighbour for neighbour in adjacency[core_id] if neighbour in node_infos and neighbour != core_id)
        if neighbours:
            for neighbour in neighbours:
                branch_roots.setdefault(neighbour, core_id)
        else:
            branch_roots[core_id] = core_id

    components: List[List[str]] = []
    visited: set[str] = set()
    for node_id in node_ids:
        if node_id in visited:
            continue
        stack = [node_id]
        component: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    stack.append(neighbour)
        components.append(component)

    for component in components:
        component_has_branch = any(node in branch_roots for node in component if node not in core_nodes)
        if component_has_branch:
            continue
        component_has_core = any(node in core_nodes for node in component)
        if component_has_core and any(node in branch_roots for node in component):
            continue
        if component_has_core:
            continue
        candidate = max(component, key=lambda nid: scores[nid])
        branch_roots[candidate] = candidate

    branch_assignment: Dict[str, str] = {core_id: core_id for core_id in core_nodes}
    queue: deque[str] = deque()
    for branch_node in sorted(branch_roots):
        branch_assignment[branch_node] = branch_node
        queue.append(branch_node)

    while queue:
        node_id = queue.popleft()
        owner = branch_assignment[node_id]
        for neighbour in adjacency[node_id]:
            if neighbour in branch_assignment:
                continue
            branch_assignment[neighbour] = owner
            queue.append(neighbour)

    for node_id in node_ids:
        if node_id not in branch_assignment:
            branch_assignment[node_id] = node_id

    unique_branches = sorted(set(branch_assignment.values()))
    if not unique_branches:
        unique_branches = [sorted_nodes[0]]
    branch_count = max(len(unique_branches), 1)
    angle_step = 2 * math.pi / branch_count
    branch_angles: Dict[str, float] = {branch: index * angle_step for index, branch in enumerate(unique_branches)}

    base_radius = 0.55
    radius_step = 0.35
    branch_ring_nodes: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for node_id, ring in rings.items():
        if ring == 0:
            continue
        branch = branch_assignment.get(node_id, node_id)
        branch_ring_nodes[branch][ring].append(node_id)

    spread_cap = 0.3

    for branch, ring_map in branch_ring_nodes.items():
        base_angle = branch_angles.get(branch, 0.0)
        for ring_index in sorted(ring_map):
            nodes_in_ring = ring_map[ring_index]
            if not nodes_in_ring:
                continue
            nodes_in_ring.sort(key=lambda nid: (_normalise_type(node_infos[nid].node_type) or "", nid))
            count = len(nodes_in_ring)
            if count == 1:
                offsets = [0.0]
            else:
                branch_spread = min(angle_step * 0.25, spread_cap)
                if branch_spread <= 0.0:
                    branch_spread = spread_cap
                if count == 2:
                    offsets = [-branch_spread / 2, branch_spread / 2]
                else:
                    step_offset = branch_spread / max(count - 1, 1)
                    start = -branch_spread / 2
                    offsets = [start + idx * step_offset for idx in range(count)]
            radius = base_radius + (ring_index - 1) * radius_step
            for offset, node_id in zip(offsets, nodes_in_ring):
                angle = base_angle + offset
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[node_id] = (x, y)

    xs = [coord[0] for coord in positions.values()]
    ys = [coord[1] for coord in positions.values()]
    min_x = min(xs, default=0.0)
    max_x = max(xs, default=0.0)
    min_y = min(ys, default=0.0)
    max_y = max(ys, default=0.0)
    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = 2.0 / span

    normalised: Dict[str, Tuple[float, float]] = {}
    for node_id, (x, y) in positions.items():
        nx = max(min((x - centre_x) * scale, 1.0), -1.0)
        ny = max(min((y - centre_y) * scale, 1.0), -1.0)
        normalised[node_id] = (nx, ny)

    max_score = max(scores.values(), default=1.0)
    importance: Dict[str, float] = {}
    for node_id in node_ids:
        importance[node_id] = scores[node_id] / max_score if max_score else 0.0

    return LayoutResult(positions=normalised, rings=dict(rings), importance=importance)
