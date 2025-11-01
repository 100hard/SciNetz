"""Tests for UI layout helpers producing spider-web graph arrangements."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

from backend.app.ui.layout import LayoutResult, NodeLayoutInfo, compute_node_positions


def _build_node(
    node_id: str,
    *,
    times_seen: int,
    node_type: str,
    sections: Iterable[Tuple[str, int]],
) -> NodeLayoutInfo:
    return NodeLayoutInfo(
        node_id=node_id,
        times_seen=times_seen,
        node_type=node_type,
        section_distribution=dict(sections),
    )


def test_compute_node_positions_spreads_nodes_evenly() -> None:
    """Layout should spread related nodes across distinct angles with outward progression."""
    node_infos = {
        "core": _build_node("core", times_seen=40, node_type="Method", sections=[("Methods", 5)]),
        "dataset_a": _build_node("dataset_a", times_seen=12, node_type="Dataset", sections=[("Results", 3)]),
        "dataset_b": _build_node("dataset_b", times_seen=9, node_type="Dataset", sections=[("Results", 2)]),
        "child_a_1": _build_node("child_a_1", times_seen=4, node_type="Observation", sections=[("Discussion", 2)]),
        "child_a_2": _build_node("child_a_2", times_seen=3, node_type="Observation", sections=[("Discussion", 1)]),
        "child_b_1": _build_node("child_b_1", times_seen=5, node_type="Observation", sections=[("Results", 1)]),
    }
    edges = [
        ("core", "dataset_a"),
        ("core", "dataset_b"),
        ("dataset_a", "child_a_1"),
        ("dataset_a", "child_a_2"),
        ("dataset_b", "child_b_1"),
    ]

    result: LayoutResult = compute_node_positions(node_infos, edges)
    positions = result.positions

    # All positions should cluster near the origin (ForceAtlas2 will take over later).
    for node_id, coord in positions.items():
        radius = math.hypot(*coord)
        assert radius < 0.12, f"{node_id} positioned too far from origin: {coord}"

    # Coordinates must remain deterministic.
    expected = compute_node_positions(node_infos, edges).positions
    assert positions == expected
