"""Shared test doubles for binding real GraphRagMixin/GraphRagEngine methods
onto hand-built brain fixtures (MagicMock(spec=AxonBrain) or a bare
AxonBrain.__new__(AxonBrain)) — extracted from test_main.py so other test
files (e.g. test_graphrag_backend_bugs.py) needing the same pattern don't
duplicate it.

Post-M2 (AxonBrain no longer inherits GraphRagMixin), a GraphRagMixin
method's body may call other GraphRagMixin methods or lazy-init properties
internally (e.g. self._gr_log_profile, self._gr_cache_lock). Binding such a
method directly onto a bare/spec'd brain (self=brain) breaks the moment its
body touches one of those — brain genuinely doesn't have them anymore. The
fix used throughout this module: bind against a real (but disk-I/O-free)
GraphRagEngine-like instance whose graph-state dicts live-proxy through to
the brain fixture, so MRO and lazy-init properties resolve normally.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from axon.graph_backends.graphrag_engine import GraphRagEngine


def _graphrag_class_method(method_name):
    """getattr(GraphRagEngine, method_name), handling _expand_with_entity_graph —
    renamed to the public expand_with_entity_graph() when it moved from
    QueryRouterMixin onto GraphRagEngine in the M2 ownership-inversion. Some
    call sites iterate a method-name list that mixes GraphRagMixin methods
    (now only on GraphRagEngine) with genuine AxonBrain methods (e.g. RAPTOR
    helpers) — fall back to AxonBrain for names GraphRagEngine doesn't have.
    """
    from axon.main import AxonBrain

    if method_name == "_expand_with_entity_graph":
        return GraphRagEngine.expand_with_entity_graph
    if hasattr(GraphRagEngine, method_name):
        return getattr(GraphRagEngine, method_name)
    return getattr(AxonBrain, method_name)


def _bind_graphrag_method(brain, method_name):
    """Bind method_name onto a MagicMock(spec=AxonBrain) brain for the mixed
    method-name lists used throughout these fixtures.

    GraphRagEngine methods (the vast majority) bind through the
    _bare_graphrag_engine() wrapper so their internal calls to other,
    un-enumerated GraphRagMixin methods (e.g. self._gr_log_profile,
    self.config) resolve against a real instance instead of raising
    AttributeError on the spec-restricted brain mock — spec=AxonBrain no
    longer exposes GraphRagMixin's surface now that AxonBrain doesn't
    inherit it. Genuine AxonBrain-only methods (e.g. RAPTOR helpers) still
    bind directly with self=brain, since that's correct for them.
    """
    if method_name == "_expand_with_entity_graph" or hasattr(GraphRagEngine, method_name):
        engine_attr = (
            "expand_with_entity_graph"
            if method_name == "_expand_with_entity_graph"
            else method_name
        )
        setattr(brain, method_name, getattr(_bare_graphrag_engine(brain), engine_attr))
        return
    method = _graphrag_class_method(method_name)
    setattr(brain, method_name, lambda *a, m=method, **kw: m(brain, *a, **kw))


def _make_live_proxy_property(attr, default_factory):
    """A property that reads/writes straight through to self._brain.<attr>,
    live — not a snapshot taken once at bind time. Falls back to
    default_factory() (and writes that default back onto the brain) the
    first time it's read before the test (or brain construction) has set
    anything real there.
    """

    def getter(self):
        value = getattr(self._brain, attr, None)
        if value is None or isinstance(value, MagicMock):
            value = default_factory()
            try:
                setattr(self._brain, attr, value)
            except Exception:
                pass
        return value

    def setter(self, value):
        try:
            setattr(self._brain, attr, value)
        except Exception:
            pass

    return property(getter, setter)


class _TestGraphRagEngine(GraphRagEngine):
    """GraphRagEngine variant used only by these test rebind helpers.

    Binding GraphRagMixin methods with self=brain directly
    (MagicMock(spec=AxonBrain), or a bare AxonBrain.__new__(AxonBrain)) only
    works when the method body doesn't call any OTHER GraphRagMixin method
    or lazy-init property internally — AxonBrain no longer recognises any of
    those names post-M2, so a spec'd mock raises AttributeError (and a bare
    real AxonBrain instance raises it too, spec or not) the moment a bound
    method's body touches e.g. self._gr_cache_lock or
    self._graph_rag_entity_cache_key. Binding with self=a real engine
    instance sidesteps that: GraphRagMixin's own MRO and lazy-init
    properties (locks, caches, token index) resolve normally since the
    engine is a real instance of a real class.

    Unlike the real production GraphRagEngine (which OWNS its graph state
    directly), this test-only subclass turns the core graph-state dicts into
    LIVE properties reading/writing straight through to self._brain.<attr> —
    not a one-time reference snapshot — so it doesn't matter whether the test
    sets brain._entity_graph before or after binding a method, or reassigns
    it (not just mutates in place) later; every rebound method still sees
    whatever's currently on brain.
    """


for _attr, _default in (
    ("_entity_graph", dict),
    ("_relation_graph", dict),
    ("_community_levels", dict),
    ("_community_summaries", dict),
    ("_community_hierarchy", dict),
    ("_community_children", dict),
    ("_claims_graph", dict),
    ("_entity_embeddings", dict),
    ("_entity_description_buffer", dict),
    ("_relation_description_buffer", dict),
    ("_text_unit_entity_map", dict),
    ("_text_unit_relation_map", dict),
    ("_graph_rag_cache", dict),
    ("_community_graph_dirty", lambda: False),
    ("_community_build_in_progress", lambda: False),
    ("_graph_rag_cache_dirty", lambda: False),
    ("_last_matched_entities", list),
):
    setattr(_TestGraphRagEngine, _attr, _make_live_proxy_property(_attr, _default))


# Plain (no MagicMock-replacement heuristic) passthrough for _rebel_pipeline —
# unlike the graph-state dicts above, a MagicMock IS a legitimate value here
# (tests set brain._rebel_pipeline to a fake pipeline object), so
# _make_live_proxy_property's "replace any MagicMock with the default" logic
# would wrongly discard it.
_TestGraphRagEngine._rebel_pipeline = property(
    lambda self: getattr(self._brain, "_rebel_pipeline", None),
    lambda self, value: setattr(self._brain, "_rebel_pipeline", value),
)


def _make_delegating_method(method_name):
    """A method that calls brain.<method_name>(...) instead of the real
    GraphRagMixin implementation when the test has explicitly replaced it
    with a MagicMock on the brain AFTER binding (e.g.
    brain._extract_entities = MagicMock(...), a very common pattern for
    controlling/spying on one step of a larger pipeline without invoking the
    real LLM/disk I/O) — falls back to the real implementation otherwise.
    Applied to every regular (non-property, non-staticmethod) GraphRagMixin
    method, since any of them may be the one a given test chooses to stub.
    """

    def method(self, *args, **kwargs):
        override = getattr(self._brain, method_name, None)
        if isinstance(override, MagicMock):
            return override(*args, **kwargs)
        return getattr(GraphRagEngine, method_name)(self, *args, **kwargs)

    return method


_GRAPHRAGMIXIN_PROPERTY_OR_STATIC_NAMES = frozenset(
    {
        # properties — self-contained lazy-init, must NOT be wrapped as methods
        "_graph_lock",
        "_gr_cache_lock",
        "_entity_token_index",
        "_traversal_cache",
        "_traversal_cache_lock",
        "_traversal_cache_maxsize",
        "_traversal_cache_ttl",
        "_persist_executor",
        # staticmethods — take no self, called as GraphRagEngine.<name>(...) directly
        "_graph_connected_components",
        "_build_networkx_graph_from_edges",
        "_normalize_entity_graph",
        "_normalize_relation_graph",
        "_build_synthetic_community_hierarchy",
        "_parse_rebel_output",
    }
)

for _name in dir(GraphRagEngine):
    if _name in _GRAPHRAGMIXIN_PROPERTY_OR_STATIC_NAMES:
        continue
    if _name.startswith("__"):
        continue
    _attr_on_class = getattr(GraphRagEngine, _name, None)
    if not callable(_attr_on_class):
        continue
    if isinstance(inspect.getattr_static(GraphRagEngine, _name, None), staticmethod | property):
        continue
    setattr(_TestGraphRagEngine, _name, _make_delegating_method(_name))


def _bare_graphrag_engine(brain):
    """Return a bare _TestGraphRagEngine proxying config/llm/vector_store and
    the core graph-state dicts back onto *brain*, without running
    GraphRagEngine.__init__ (no real disk I/O). See _TestGraphRagEngine's
    docstring for why this exists instead of using GraphRagEngine directly.
    """
    engine = _TestGraphRagEngine.__new__(_TestGraphRagEngine)
    engine._brain = brain
    return engine


def _bind_gr_cache_methods(brain):
    """Bind real GraphRagMixin cache helpers (via a bare GraphRagEngine) onto
    a MagicMock or real brain.

    MagicMock(spec=AxonBrain) mocks every method including _gr_cache_get,
    so real bound methods like _extract_relations get MagicMock back from
    cache lookups (not None) and think there's a cache hit. Binding the real
    implementations fixes this for the whole test.

    Also binds incoming-relation index helpers so entity-degree sorting
    produces real ints (not MagicMock) and _gr_write_json_if_changed so
    persistence round-trip tests actually write files.

    Returns the bare engine so callers needing to inspect its internal state
    (e.g. the extraction cache) can do so directly.
    """
    engine = _bare_graphrag_engine(brain)
    engine._graph_rag_cache = {}
    brain._gr_cache_get = engine._gr_cache_get
    brain._gr_cache_put = engine._gr_cache_put
    brain._gr_cache_store = engine._gr_cache_store
    brain._gr_text_hash = engine._gr_text_hash
    brain._gr_llm_complete_cached = engine._gr_llm_complete_cached
    brain._parse_extracted_entities = engine._parse_extracted_entities
    brain._parse_extracted_relations = engine._parse_extracted_relations
    brain._load_graph_rag_extraction_cache = engine._load_graph_rag_extraction_cache
    brain._save_graph_rag_extraction_cache = engine._save_graph_rag_extraction_cache
    brain._extract_graph_llm_batches = engine._extract_graph_llm_batches
    brain._build_graph_edge_payload = engine._build_graph_edge_payload
    brain._build_networkx_graph_from_edges = engine._build_networkx_graph_from_edges
    brain._graph_connected_components = engine._graph_connected_components
    brain._build_synthetic_community_hierarchy = engine._build_synthetic_community_hierarchy
    brain._get_incoming_relation_index = engine._get_incoming_relation_index
    brain._get_incoming_relation_count_map = engine._get_incoming_relation_count_map
    brain._gr_write_json_if_changed = engine._gr_write_json_if_changed
    return engine
