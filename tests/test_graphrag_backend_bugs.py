"""
tests/test_graphrag_backend_bugs.py

Regression tests for bugs documented in
Qualification/CodexQual/LOUVAIN_AND_REBEL_BUG_FINDINGS_2026_03_22.md

Bugs covered
------------
Finding 1 — Community backend safety
  BUG-COM-1: Default ``graph_rag_community_backend`` was "auto", now "louvain"
  BUG-COM-2: Intentional backend override used ``raise ImportError(...)`` as control
             flow, making override skips indistinguishable from real import failures.
             Fixed: ``if/else`` branching so ``except ImportError`` only fires on
             genuine unavailability.

Finding 2 — REBEL zero-edge silent failure
  BUG-REBEL-1: ``local_files_only`` leaked into pipeline generate call.  (fixed in
               _ensure_rebel; covered by test_rebel_local_path_does_not_forward_*
               in test_new_features.py)
  BUG-REBEL-2: ``generated_text`` output strips special tokens → zero triplets.
               (fixed in _extract_relations_rebel; covered by
               test_extract_relations_rebel_decodes_* in test_new_features.py)
  BUG-REBEL-3: Tensor truthiness error on ``generated_token_ids``.  (fixed; covered
               by test_extract_relations_rebel_decodes_*_tensor in test_new_features.py)
  BUG-REBEL-4 (P1 observability): Zero-triplet warning must fire when tokens decoded
              but parser returns nothing.
  BUG-REBEL-5 (P1 observability): Empty token_ids must emit debug, not warning.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain(cfg):
    """Return a bare AxonBrain instance with the given config (no __init__ call)."""
    from axon.main import AxonBrain

    brain = AxonBrain.__new__(AxonBrain)
    brain.config = cfg
    brain._rebel_pipeline = None
    return brain


# ---------------------------------------------------------------------------
# BUG-COM-1: default community backend changed to "louvain"
# ---------------------------------------------------------------------------


class TestDefaultCommunityBackend:
    """BUG-COM-1 — config.py default must be 'louvain', not 'auto'."""

    def test_axonconfig_default_is_louvain(self):
        from axon.main import AxonConfig

        cfg = AxonConfig.__new__(AxonConfig)
        cfg.__init__()
        assert cfg.graph_rag_community_backend == "louvain", (
            f"Expected default 'louvain', got '{cfg.graph_rag_community_backend}'. "
            "Changing the default to 'auto' re-enables the graspologic hang on Python 3.13."
        )

    def test_auto_is_not_the_default(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.graph_rag_community_backend != "auto"

    def test_louvain_can_be_overridden_to_leidenalg(self):
        from axon.main import AxonConfig

        cfg = AxonConfig(graph_rag_community_backend="leidenalg")
        assert cfg.graph_rag_community_backend == "leidenalg"

    def test_louvain_can_be_overridden_to_auto(self):
        from axon.main import AxonConfig

        cfg = AxonConfig(graph_rag_community_backend="auto")
        assert cfg.graph_rag_community_backend == "auto"


# ---------------------------------------------------------------------------
# BUG-COM-2: backend override must not emit WARNING log
# ---------------------------------------------------------------------------


class TestCommunityBackendLogging:
    """BUG-COM-2 — intentional override skips must use DEBUG, not WARNING.

    Before the fix, ``raise ImportError("backend override — ...")`` was the
    control-flow mechanism.  That caused the ``except ImportError`` handler to
    fire for *both* intentional skips and genuine package unavailability, always
    printing "graspologic not installed" even when the package was present but
    the user simply chose a different backend.
    """

    @staticmethod
    def _make_entity_relation_graphs(n_nodes: int = 6) -> tuple[dict, dict]:
        """Return (entity_graph_dict, relation_graph_dict) for a small path graph."""
        nodes = [f"e{i}" for i in range(n_nodes)]
        entity_graph = {node: {"description": f"entity {node}", "frequency": 2} for node in nodes}
        relation_graph: dict = {}
        for i in range(n_nodes - 1):
            src = nodes[i]
            tgt = nodes[i + 1]
            relation_graph.setdefault(src, []).append(
                {"target": tgt, "relation": "related_to", "weight": 1, "description": ""}
            )
        return entity_graph, relation_graph

    def _run_community_detection(self, backend: str):
        """Call _run_hierarchical_community_detection using entity/relation graph dicts."""
        from axon.main import AxonBrain, AxonConfig

        entity_graph, relation_graph = self._make_entity_relation_graphs()

        cfg = AxonConfig(
            graph_rag_community_backend=backend,
            graph_rag=True,
            graph_rag_community_levels=1,
        )
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg
        brain._entity_graph = entity_graph
        brain._relation_graph = relation_graph
        return brain._run_hierarchical_community_detection()

    # The logger in graph_rag.py is ``logging.getLogger("Axon")`` — use that name.
    _LOGGER = "Axon"

    def test_louvain_backend_emits_no_warning_for_graspologic(self, caplog):
        """louvain backend must not warn about graspologic being unavailable."""
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            self._run_community_detection("louvain")

        graspologic_warnings = [
            r for r in caplog.records if "graspologic" in r.message and r.levelno >= logging.WARNING
        ]
        assert not graspologic_warnings, (
            f"Unexpected graspologic WARNING when backend='louvain': "
            f"{[r.message for r in graspologic_warnings]}"
        )

    def test_louvain_backend_emits_no_warning_for_leidenalg(self, caplog):
        """louvain backend must not warn about leidenalg being unavailable."""
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            self._run_community_detection("louvain")

        leidenalg_warnings = [
            r for r in caplog.records if "leidenalg" in r.message and r.levelno >= logging.WARNING
        ]
        assert not leidenalg_warnings, (
            f"Unexpected leidenalg WARNING when backend='louvain': "
            f"{[r.message for r in leidenalg_warnings]}"
        )

    def test_louvain_backend_emits_debug_skip_for_graspologic(self, caplog):
        """louvain backend must emit a DEBUG message indicating graspologic was skipped."""
        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            self._run_community_detection("louvain")

        skip_records = [
            r
            for r in caplog.records
            if "graspologic" in r.message.lower() and r.levelno == logging.DEBUG
        ]
        assert skip_records, (
            "Expected a DEBUG log mentioning graspologic was skipped for backend='louvain', "
            "but found none. This helps operators confirm the backend selection is intentional."
        )

    def test_louvain_backend_emits_debug_skip_for_leidenalg(self, caplog):
        """louvain backend must emit a DEBUG message indicating leidenalg was skipped."""
        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            self._run_community_detection("louvain")

        skip_records = [
            r
            for r in caplog.records
            if "leidenalg" in r.message.lower() and r.levelno == logging.DEBUG
        ]
        assert skip_records, (
            "Expected a DEBUG log mentioning leidenalg was skipped for backend='louvain', "
            "but found none."
        )

    def test_leidenalg_backend_skips_graspologic_without_warning(self, caplog):
        """leidenalg backend must not produce graspologic WARNING."""
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            # leidenalg may not be installed in CI — both outcomes are fine as
            # long as no graspologic warning is emitted.
            try:
                self._run_community_detection("leidenalg")
            except Exception:
                pass

        graspologic_warnings = [
            r for r in caplog.records if "graspologic" in r.message and r.levelno >= logging.WARNING
        ]
        assert (
            not graspologic_warnings
        ), "leidenalg backend must skip graspologic silently (DEBUG), not produce a WARNING."

    def test_auto_backend_with_missing_graspologic_emits_warning(self, caplog):
        """auto backend must emit WARNING when graspologic import fails (genuine unavailability)."""
        with (
            patch.dict("sys.modules", {"graspologic": None, "graspologic.partition": None}),
            caplog.at_level(logging.WARNING, logger=self._LOGGER),
        ):
            self._run_community_detection("auto")

        graspologic_warnings = [
            r for r in caplog.records if "graspologic" in r.message and r.levelno >= logging.WARNING
        ]
        assert graspologic_warnings, (
            "auto backend must emit a WARNING when graspologic is genuinely unavailable. "
            f"Captured records: {[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_louvain_backend_returns_valid_partition(self):
        """louvain backend produces a non-empty community mapping covering all nodes."""
        entity_graph, relation_graph = self._make_entity_relation_graphs(n_nodes=6)
        result = self._run_community_detection("louvain")
        community_levels, hierarchy, children = result
        # At least level 0 must exist and cover all entity nodes
        assert 0 in community_levels
        assert set(community_levels[0].keys()) == set(entity_graph.keys())


# ---------------------------------------------------------------------------
# BUG-REBEL-4: zero-triplet warning (P1 observability)
# ---------------------------------------------------------------------------


class TestRebelObservability:
    """BUG-REBEL-4/5 — REBEL must surface failures loudly via structured logging."""

    # graph_rag.py uses ``logging.getLogger("Axon")``
    _LOGGER = "Axon"

    def _make_rebel_brain(self, pipeline_output, decoded_output="no triplets here"):
        """Wire a brain whose REBEL pipeline returns a controlled output."""
        from axon.main import AxonBrain, AxonConfig

        cfg = AxonConfig(
            graph_rag=True,
            graph_rag_relation_backend="rebel",
        )
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg

        mock_pipe = MagicMock(return_value=pipeline_output)
        mock_pipe.model = MagicMock()
        mock_pipe.model.config = SimpleNamespace(task_specific_params={"relation_extraction": {}})
        mock_pipe.tokenizer = MagicMock()
        mock_pipe.tokenizer.batch_decode.return_value = [decoded_output]
        brain._rebel_pipeline = mock_pipe
        return brain

    def test_zero_triplets_emits_warning(self, caplog):
        """BUG-REBEL-4 — When tokens are decoded but parser returns no triplets, a WARNING is logged."""
        from axon.main import AxonBrain

        # Pipeline returns non-empty token_ids, but decoded text has no REBEL triplet markers
        brain = self._make_rebel_brain(
            pipeline_output=[{"generated_token_ids": [1, 2, 3, 4, 5]}],
            decoded_output="no triplet markers here at all",
        )

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            result = AxonBrain._extract_relations_rebel(brain, "Apple and Microsoft compete.")

        assert result == []

        zero_warnings = [
            r
            for r in caplog.records
            if "parsed 0 relation triplets" in r.message and r.levelno >= logging.WARNING
        ]
        assert zero_warnings, (
            "Expected a WARNING about 0 parsed triplets when REBEL decoded tokens but "
            "the parser found none. This surfaces integration errors that were previously "
            "silently degrading to empty relation lists."
        )

    def test_zero_triplets_warning_contains_token_count(self, caplog):
        """BUG-REBEL-4 — The zero-triplet warning must mention the decoded token count."""
        from axon.main import AxonBrain

        brain = self._make_rebel_brain(
            pipeline_output=[{"generated_token_ids": [10, 20, 30]}],
            decoded_output="plain text no markers",
        )

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            AxonBrain._extract_relations_rebel(brain, "Some text.")

        warning_msgs = [
            r.message
            for r in caplog.records
            if "triplets" in r.message and r.levelno >= logging.WARNING
        ]
        # The warning should include the token count (3 tokens)
        assert any(
            "3" in msg for msg in warning_msgs
        ), f"Expected token count '3' in the zero-triplet warning. Got: {warning_msgs}"

    def test_zero_triplets_no_warning_for_whitespace_only_input(self, caplog):
        """BUG-REBEL-4 — Whitespace-only input must NOT trigger the zero-triplet warning."""
        from axon.main import AxonBrain

        brain = self._make_rebel_brain(
            pipeline_output=[{"generated_token_ids": [1]}],
            decoded_output="",
        )

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            result = AxonBrain._extract_relations_rebel(brain, "   ")  # whitespace only

        # Result should be empty but no noisy warning for trivial empty input
        assert result == []
        spurious = [
            r
            for r in caplog.records
            if "parsed 0 relation triplets" in r.message and r.levelno >= logging.WARNING
        ]
        assert (
            not spurious
        ), "No zero-triplet WARNING should fire for whitespace-only input — that would be noise."

    def test_empty_token_ids_emits_debug_not_warning(self, caplog):
        """BUG-REBEL-5 — Empty token_ids must log at DEBUG, not WARNING."""
        from axon.main import AxonBrain

        brain = self._make_rebel_brain(
            pipeline_output=[{"generated_token_ids": []}],
            decoded_output="",
        )

        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            result = AxonBrain._extract_relations_rebel(brain, "Some text.")

        assert result == []

        # Must emit DEBUG
        debug_records = [
            r
            for r in caplog.records
            if "empty token_ids" in r.message.lower() and r.levelno == logging.DEBUG
        ]
        assert debug_records, (
            "Expected a DEBUG log when REBEL returns empty token_ids. "
            "This is informative but not alarming."
        )

        # Must NOT emit WARNING for this case
        warning_records = [
            r
            for r in caplog.records
            if "empty token_ids" in r.message.lower() and r.levelno >= logging.WARNING
        ]
        assert not warning_records, (
            "Empty token_ids should be DEBUG, not WARNING — it's a normal edge case "
            "(very short input, tokenizer quirk)."
        )

    def test_rebel_debug_log_for_decoded_output(self, caplog):
        """P1 observability — decoded output (first 200 chars) must appear in DEBUG log."""
        from axon.main import AxonBrain

        decoded = "<triplet> Apple <subj> Microsoft <obj> partners with"
        brain = self._make_rebel_brain(
            pipeline_output=[{"generated_token_ids": [1, 2, 3]}],
            decoded_output=decoded,
        )

        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            AxonBrain._extract_relations_rebel(brain, "Apple and Microsoft are partners.")

        debug_msgs = [
            r.message
            for r in caplog.records
            if r.levelno == logging.DEBUG and "REBEL decoded" in r.message
        ]
        assert debug_msgs, "Expected a DEBUG log containing the REBEL decoded output prefix."

    def test_successful_rebel_extraction_emits_no_warning(self, caplog):
        """When REBEL successfully extracts relations, no zero-triplet WARNING should fire."""
        from axon.main import AxonBrain

        decoded = "<triplet> Apple <subj> Microsoft <obj> partners with"
        brain = self._make_rebel_brain(
            pipeline_output=[{"generated_token_ids": [1, 2, 3]}],
            decoded_output=decoded,
        )

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            result = AxonBrain._extract_relations_rebel(brain, "Apple and Microsoft are partners.")

        assert len(result) >= 1
        spurious = [
            r
            for r in caplog.records
            if "parsed 0 relation triplets" in r.message and r.levelno >= logging.WARNING
        ]
        assert not spurious, "No zero-triplet warning should fire when relations are found."
