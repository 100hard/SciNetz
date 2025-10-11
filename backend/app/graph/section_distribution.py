"""Utilities for encoding and decoding section distribution payloads."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableSequence, Sequence, Tuple


def encode_section_distribution(
    distribution: Mapping[str, int],
) -> Tuple[MutableSequence[str], MutableSequence[int]]:
    """Encode a section distribution mapping into parallel string and integer lists.

    Args:
        distribution: Mapping of section name to occurrence count.

    Returns:
        Tuple containing parallel lists of section keys and integer counts.
    """

    keys: MutableSequence[str] = []
    values: MutableSequence[int] = []
    for section, count in sorted(distribution.items()):
        if not section:
            continue
        try:
            numeric = int(count)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        keys.append(str(section))
        values.append(numeric)
    return keys, values


def decode_section_distribution(
    distribution: Any,
    keys: Any,
    values: Any,
) -> Dict[str, int]:
    """Decode distribution data returned from Neo4j into a dictionary.

    Args:
        distribution: Direct mapping payload if available.
        keys: Sequence of section names when stored in parallel arrays.
        values: Sequence of section counts aligned with ``keys``.

    Returns:
        Dictionary mapping section names to counts.
    """

    if isinstance(distribution, Mapping):
        return {
            str(section): int(count)
            for section, count in distribution.items()
            if _is_positive_int(count) and str(section)
        }
    sections = _coerce_sequence(keys)
    counts = _coerce_sequence(values)
    result: Dict[str, int] = {}
    for section, count in zip(sections, counts):
        if not section or not _is_positive_int(count):
            continue
        result[str(section)] = int(count)
    return result


def decode_distribution_from_mapping(data: Mapping[str, Any]) -> Dict[str, int]:
    """Decode section distribution data from a property mapping.

    Args:
        data: Mapping containing potential section distribution properties.

    Returns:
        Dictionary mapping section names to counts.
    """

    distribution = data.get("section_distribution")
    keys = data.get("section_distribution_keys")
    values = data.get("section_distribution_values")
    return decode_section_distribution(distribution, keys, values)


def _coerce_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if value is None:
        return []
    return [value]


def _is_positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False
