"""Pydantic models for the dynamic (temporal) graph data layer.

These models mirror the SQLite schema defined in
``docs/DYNAMIC_GRAPH_ROADMAP.md`` (branch: ``docs/dynamic-graph-design``),
which in turn follows Graphiti's bi-temporal data model:
  Episode  →  source_chunk_id, content, reference_time
  Entity   →  canonical_name, first_seen_at, last_seen_at
  Fact     →  subject, relation, object, valid_at, invalid_at, status
  FactEvidence → fact_id, chunk_id (many-to-many join)

Graphiti concept → SQLite roadmap equivalent:
  Episode        → episodes table
  EntityNode     → entities table
  EntityEdge     → facts table  (valid_at / invalid_at on every row)
  Episode→edge   → fact_evidence table
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core temporal models
# ---------------------------------------------------------------------------


class Episode(BaseModel):
    """A single ingested chunk, the atomic unit of temporal knowledge."""

    episode_id: str
    source_chunk_id: str
    content: str
    reference_time: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A canonical named entity that persists across episodes."""

    entity_id: str
    canonical_name: str
    entity_type: str = "UNKNOWN"
    description: str = ""
    first_seen_at: datetime
    last_seen_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    """A bi-temporal fact (subject, relation, object) with validity window.
    When a new fact contradicts an existing one, ``invalid_at`` is set on
    the old row and ``status`` is set to ``"superseded"``.  The new fact
    row is inserted with ``status = "active"``.  Facts are never deleted.
    """

    fact_id: str
    subject: str
    relation: str
    object: str
    valid_at: datetime
    invalid_at: datetime | None = None
    status: str = "active"  # "active" | "superseded"
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == "active" and self.invalid_at is None


class FactEvidence(BaseModel):
    """Join table linking a Fact to the chunk (and episode) that asserts it."""

    fact_id: str
    chunk_id: str
    episode_id: str = ""
    excerpt: str = ""


# ---------------------------------------------------------------------------
# Relation registry
# ---------------------------------------------------------------------------

_DEFAULT_RELATIONS: list[str] = [
    "WORKS_FOR",
    "KNOWS",
    "LOCATED_IN",
    "PART_OF",
    "OWNS",
    "CREATED_BY",
    "RELATED_TO",
    "CAUSES",
    "IS_A",
    "HAS",
    "MENTIONS",
    "CONTRADICTS",
]


class RelationRegistry:
    """Registry of known relation types with built-in defaults.
    Usage::
        reg = RelationRegistry(extra=["FUNDED_BY", "ACQUIRED_BY"])
        assert "WORKS_FOR" in reg
        reg.register("ACQUIRED_BY")
        print(reg.all())
    """

    def __init__(self, extra: list[str] | None = None) -> None:
        self._relations: set[str] = set(_DEFAULT_RELATIONS)
        if extra:
            self._relations.update(r.upper() for r in extra)

    def register(self, relation: str) -> None:
        """Add a new relation type (normalised to upper-case)."""
        self._relations.add(relation.upper())

    def __contains__(self, relation: object) -> bool:
        if not isinstance(relation, str):
            return False
        return relation.upper() in self._relations

    def all(self) -> list[str]:
        """Return all registered relation types sorted alphabetically."""
        return sorted(self._relations)
