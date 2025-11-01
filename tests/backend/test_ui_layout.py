"""Tests for UI layout helpers producing spider-web graph arrangements."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

from backend.app.ui.layout import LayoutResult, NodeLayoutInfo, compute_node_positions


def _angles_from_positions(positions: Dict[str, Tuple[float, float]], *node_ids: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for node_id in node_ids:
        x, y = positions[node_id]
        result[node_id] = math.atan2(y, x)
    return result


def _radius(position: Tuple[float, float]) -> float:
    return math.hypot(position[0], position[1])


def _angular_separation(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b)
    while diff > math.pi:
        diff -= 2 * math.pi
    return abs(diff)


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


def test_compute_node_positions_groups_branch_descendants() -> None:
    """Descendants on the same branch should align along a single spoke."""
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

    angles = _angles_from_positions(
        positions,
        "dataset_a",
        "dataset_b",
        "child_a_1",
        "child_a_2",
        "child_b_1",
    )

    # Nodes on the same branch share roughly the same spoke orientation.
    branch_alignment = _angular_separation(angles["child_a_1"], angles["child_a_2"])
    assert branch_alignment < 0.35

    # Distinct branches should still have visible separation.
    cross_branch = _angular_separation(angles["child_a_1"], angles["child_b_1"])
    assert cross_branch > 0.4

    # Descendants should progress outward radially.
    assert _radius(positions["child_a_1"]) > _radius(positions["dataset_a"])
    assert _radius(positions["child_a_2"]) > _radius(positions["dataset_a"])
