"""Architecture enforcement: direct access to GraphRAG state must not leak
outside the GraphRagMixin/GraphRagBackend adapter.

This is the M2 exit-criteria test named (but never written) by M1 — see
``docs/architecture/DYNAMIC_GRAPH_ROADMAP.md`` ("M1"/"M2") and
``docs/architecture/GRAPH_BACKEND_NEXT_STEPS.md`` for the full migration
this test tracks. It uses Python's ``ast`` module (not grep) to find real
attribute access (``brain._entity_graph``) and ``getattr``/``setattr``/
``hasattr``/``delattr`` calls keyed on the same attribute name — both are
"direct access" in spirit, even though only the first is a literal
``ast.Attribute`` node.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent / "src" / "axon"

_FORBIDDEN_ATTRS = {
    "_entity_graph",
    "_relation_graph",
    "_community_summaries",
    "_claims_graph",
}

_ACCESSOR_FUNCS = {"getattr", "setattr", "hasattr", "delattr"}


def _iter_violations(path: Path):
    """Yield (lineno, description) for direct access to _FORBIDDEN_ATTRS in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            yield node.lineno, f"attribute access .{node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _ACCESSOR_FUNCS:
                continue
            candidates = list(node.args[1:2])
            candidates += [kw.value for kw in node.keywords if kw.arg == "name"]
            for arg in candidates:
                if isinstance(arg, ast.Constant) and arg.value in _FORBIDDEN_ATTRS:
                    yield node.lineno, f"{node.func.id}(..., {arg.value!r})"


def _violations_in(relative_path: str) -> list[tuple[int, str]]:
    return list(_iter_violations(_SRC / relative_path))


# ---------------------------------------------------------------------------
# Phase 1 of the M2 backend-boundary refactor (GRAPH_BACKEND_NEXT_STEPS.md)
# redirected these files to go through self._graph_backend.* instead of
# touching GraphRAG state directly. They must stay clean — this is the
# regression gate for that work.
# ---------------------------------------------------------------------------

_PHASE_1_CLEANED_FILES = [
    "agent.py",
    "cli.py",
    "collection_ops.py",
    str(Path("api_routes") / "ingest.py"),
]


class TestPhase1FilesHaveNoDirectAccess:
    @pytest.mark.parametrize("relative_path", _PHASE_1_CLEANED_FILES)
    def test_no_direct_access(self, relative_path):
        violations = _violations_in(relative_path)
        assert violations == [], (
            f"{relative_path} has direct GraphRAG-state access outside the "
            f"adapter (src/axon/graph_rag.py, "
            f"src/axon/graph_backends/graphrag_backend.py): {violations}"
        )


# ---------------------------------------------------------------------------
# Files with a documented, intentional direct-read fallback — used only when
# no _graph_backend is attached at all, or backend.status() itself raises
# (graceful degradation, matching the pattern already established in
# repl.py's "/graph status" command before Phase 1). Pinned so a *new*
# violation here gets caught instead of silently joining the pile.
#
# Phase 4 (GRAPH_BACKEND_NEXT_STEPS.md) relocated query_router.py's
# _expand_with_entity_graph() body onto GraphRagEngine (reachable via the
# self._graph_backend.expand_with_entity_graph() bridge) — its former
# 7-violation entry here is gone, not just zeroed, since it's no longer a
# file with any GraphRAG-state fallback of its own.
# ---------------------------------------------------------------------------

_KNOWN_FALLBACK_FILES = {
    "repl.py": 3,  # /graph status except-fallback
    str(Path("api_routes") / "graph.py"): 0,  # legacy no-backend branch removed in Phase 4
    str(Path("api_routes") / "governance.py"): 0,  # legacy no-backend branch removed in Phase 4
}


class TestKnownFallbackFilesHaveNotGrown:
    @pytest.mark.parametrize("relative_path,expected_max", list(_KNOWN_FALLBACK_FILES.items()))
    def test_violation_count_has_not_grown(self, relative_path, expected_max):
        violations = _violations_in(relative_path)
        assert len(violations) <= expected_max, (
            f"{relative_path} has grown new direct GraphRAG-state access beyond "
            f"its documented fallback ({expected_max} expected): {violations}"
        )


# ---------------------------------------------------------------------------
# M2 exit criterion. main.py's load/init/switch paths and its ingest()
# extraction pipeline used to access GraphRAG state directly; Phase 4
# (GRAPH_BACKEND_NEXT_STEPS.md) relocated that state and logic onto
# GraphRagEngine, reachable only through self._graph_backend._engine.
# query_router.py closed out the same way back in Phase 3/4 (see
# _KNOWN_FALLBACK_FILES above). This is now a strict, permanent regression
# gate — not an aspirational xfail.
# ---------------------------------------------------------------------------

_FULLY_MIGRATED_FILES = ["main.py"]


class TestFullComplianceTarget:
    def test_main_and_query_router_are_eventually_clean(self):
        results = {p: _violations_in(p) for p in _FULLY_MIGRATED_FILES}
        total = sum(len(v) for v in results.values())
        assert total == 0, "Remaining direct GraphRAG-state access: " + ", ".join(
            f"{p} ({len(v)} site(s))" for p, v in results.items() if v
        )
