"""Internal tuning constants for GraphRAG local/global context assembly.

These were dataclass fields on :class:`~axon.config.AxonConfig` until 0.5.0.
They are not knobs anyone was expected to turn: none of them shipped in the
default ``config.yaml``, none is reachable from ``/config/get`` or
``/config/update``, and ``AxonConfig.validate()`` rejected most of them as
unknown keys even while ``load()`` quietly accepted them — so a user who set
one got a working override *and* a warning telling them to change it to
something else.

Demoting them to module constants removes that inconsistency and shrinks the
config surface. The values are unchanged; this is a move, not a retune.

Two rules for anything added here:

1. **Import the constant, do not reach for it through ``getattr``.** The old
   ``getattr(cfg, "graph_rag_local_...", <fallback>)`` pattern let a fallback
   drift from the dataclass default it shadowed, harmlessly, because the field
   always existed and the fallback never fired. Six of the fields moved here had
   drifted that way — ``GLOBAL_MAP_MAX_LENGTH`` (fallback 500 vs default 1000),
   ``GLOBAL_REDUCE_MAX_LENGTH`` (500 vs 2000), ``GLOBAL_MAX_MAP_CHUNKS`` (0 vs
   200), ``GLOBAL_REDUCE_SKIP_IF_TOP_POINTS_LE`` (1 vs 0),
   ``GLOBAL_REDUCE_SKIP_IF_TOP_SCORE_GTE`` (95.0 vs 0) and
   ``LOCAL_EARLY_CUTOFF_FACTOR`` (0.2 vs 1.5). Each constant here takes the
   *dataclass* value, so behaviour is unchanged. Across the wider ``graph_rag_*``
   surface the same audit found 24 such drifts among 143 call sites, some of them
   inverting a boolean; those fields are not in this module yet, and deleting one
   without replacing its read would make its stale fallback live.
2. **If a value ever genuinely needs to be user-settable, promote it back to a
   real ``AxonConfig`` field** — with an entry in ``_KNOWN_YAML_KEYS`` and in
   ``config_routes.py``'s map, so every surface sees it. A constant here is a
   deliberate statement that tuning it is Axon's job, not the operator's.
"""

from __future__ import annotations

# ── Global search: map-reduce over community summaries ──────────────────────

#: Minimum map-phase score for a point to survive into the reduce phase.
GLOBAL_MIN_SCORE = 20
#: Maximum points assembled in the reduce phase.
GLOBAL_TOP_POINTS = 50
#: Token budget for the reduce-phase prompt.
GLOBAL_REDUCE_MAX_TOKENS = 8000
#: Map-phase window, in tokens. The call site multiplies by 4 to get the
#: character width of each community-report chunk.
GLOBAL_MAP_MAX_LENGTH = 1000
#: Reduce-phase response budget, in tokens. Interpolated into the prompt as
#: "Respond in at most N tokens" — a request to the model, not a hard cap.
GLOBAL_REDUCE_MAX_LENGTH = 2000
#: Whether the reduce prompt may draw on the model's general knowledge.
GLOBAL_ALLOW_GENERAL_KNOWLEDGE = False
#: Cap on chunks fed to the map phase.
GLOBAL_MAX_MAP_CHUNKS = 200

#: Skip the reduce phase when at most this many points survived.
#: 0 is falsy at the call site and means "no points-based skip".
GLOBAL_REDUCE_SKIP_IF_TOP_POINTS_LE = 0
#: Skip the reduce phase when the top point scored at least this.
#: 0 is falsy at the call site, where ``or 95.0`` then applies — so the
#: effective threshold is 95.0. Preserved verbatim from the field this
#: replaced; the indirection is odd but changing it is a retune, not a move.
GLOBAL_REDUCE_SKIP_IF_TOP_SCORE_GTE = 0

#: Cache fully-reduced global answers.
GLOBAL_ANSWER_CACHE = True
#: Entries retained in the global answer cache.
GLOBAL_ANSWER_CACHE_SIZE = 500
#: Cache individual map-phase results.
GLOBAL_MAP_CACHE = True
#: Entries retained in the map-phase cache.
GLOBAL_MAP_CACHE_SIZE = 2000

# ── Local search: entity / relation context assembly ────────────────────────

#: Token budget for the assembled local-search context.
LOCAL_MAX_CONTEXT_TOKENS = 8000
#: Entities pulled into the local context.
LOCAL_TOP_K_ENTITIES = 10
#: Relations pulled into the local context.
LOCAL_TOP_K_RELATIONSHIPS = 10
#: Whether relationship weight participates in local ranking.
LOCAL_INCLUDE_RELATIONSHIP_WEIGHT = False

#: Unified candidate-ranking weights, by candidate kind.
LOCAL_ENTITY_WEIGHT = 3.0
LOCAL_RELATION_WEIGHT = 2.0
LOCAL_COMMUNITY_WEIGHT = 1.5
LOCAL_TEXT_UNIT_WEIGHT = 1.0

# Performance paths. Each replaced a slower implementation that is still
# present as the `else` branch; they are constants rather than deleted
# branches so a regression stays bisectable by editing one line here.
#: Fetch candidate entities in one batched call.
LOCAL_BATCH_FETCH = True
#: Serve incoming-relation lookups from the cache.
LOCAL_CACHED_INCOMING = True
#: Serve incoming-relation *counts* from the cache.
LOCAL_CACHED_INCOMING_COUNTS = True
#: Stop scanning candidates once the score cannot improve.
LOCAL_EARLY_CUTOFF = True
#: Slack multiplier applied before the early cutoff triggers.
LOCAL_EARLY_CUTOFF_FACTOR = 1.5
#: Use the precomputed degree index instead of counting edges per entity.
LOCAL_ENTITY_DEGREE_FAST = True
#: Use the precomputed support index instead of recounting relation support.
LOCAL_RELATION_SUPPORT_FAST = True
