"""UI support utilities including paper registry and graph services."""

from .papers import PaperRecord, PaperRegistry, PaperStatus
from .repository import GraphViewRepositoryProtocol, GraphViewFilters, Neo4jGraphViewRepository
from .service import GraphEdge, GraphNode, GraphView, GraphViewService

__all__ = [
    "GraphEdge",
    "GraphNode",
    "GraphView",
    "GraphViewFilters",
    "GraphViewRepositoryProtocol",
    "Neo4jGraphViewRepository",
    "GraphViewService",
    "PaperRecord",
    "PaperRegistry",
    "PaperStatus",
]
