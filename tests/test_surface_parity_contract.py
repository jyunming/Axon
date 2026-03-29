"""Cross-surface parity contract tests (SP-052).























Validates that the surface capability registry is consistent with:























- VS Code extension manifest (package.json tool declarations)























- REPL command set (via repl.py source inspection)























- CLI argument set (via cli.py source inspection)























- API route set (via api_routes imports)























These tests enforce the declared parity tier matrix.  They do NOT test runtime























behaviour; they test that the declared contract matches the actual code surface.























A failure here means either:























  (a) a surface drifted from the registry without updating the registry, or























  (b) the registry claims a surface is unsupported but the surface actually has it.























"""


from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------


# Helpers


# ---------------------------------------------------------------------------


def _extension_root() -> Path:
    return REPO_ROOT / "integrations" / "vscode-axon"


def _extension_manifest() -> dict:
    return json.loads((_extension_root() / "package.json").read_text(encoding="utf-8"))


def _repl_source() -> str:
    return (REPO_ROOT / "src" / "axon" / "repl.py").read_text(encoding="utf-8")


def _cli_source() -> str:
    return (REPO_ROOT / "src" / "axon" / "cli.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------


# Registry shape tests


# ---------------------------------------------------------------------------


class TestRegistryShape:

    """The registry itself is well-formed and complete."""

    def test_no_duplicate_ids(self):
        from axon.surface_contract import REGISTRY

        ids = [c.id for c in REGISTRY]

        dups = {x for x in ids if ids.count(x) > 1}

        assert len(ids) == len(set(ids)), f"Duplicate capability IDs: {dups}"

    def test_all_capabilities_have_descriptions(self):
        from axon.surface_contract import REGISTRY

        for cap in REGISTRY:
            assert cap.description, f"Capability {cap.id} has no description"

    def test_all_tier1_have_api_route(self):
        from axon.surface_contract import REGISTRY, Tier

        for cap in REGISTRY:
            if cap.tier == Tier.ONE:
                assert cap.api_route, f"Tier 1 capability {cap.id} has no api_route"

    def test_intentional_exceptions_only_for_non_supported(self):
        from axon.surface_contract import REGISTRY

        for cap in REGISTRY:
            for surface, reason in cap.intentional_exceptions.items():
                assert surface not in cap.supported_surfaces, (
                    f"Capability {cap.id} has exception for {surface} "
                    f"but that surface is also in supported_surfaces"
                )

                assert reason, f"Capability {cap.id} exception for {surface} has empty reason"


# ---------------------------------------------------------------------------


# VS Code manifest contract


# ---------------------------------------------------------------------------


class TestVsCodeManifestContract:

    """VS Code manifest must declare all Tier 1 capabilities mapped to VS Code."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not _extension_root().exists():
            pytest.skip("VS Code extension directory not found")

    def test_manifest_tool_count(self):
        """Extension manifest should declare 34 tools (32 original + config_validate + config_set)."""

        manifest = _extension_manifest()

        tools = manifest["contributes"]["languageModelTools"]

        assert (
            len(tools) == 34
        ), f"Expected 34 tools, got {len(tools)}: {[t['name'] for t in tools]}"

    def test_tier1_vscode_capabilities_in_manifest(self):
        """Every Tier 1 capability with VS Code support has a corresponding manifest tool."""

        from axon.surface_contract import REGISTRY, Surface, Tier

        manifest = _extension_manifest()

        tool_names = {t["name"] for t in manifest["contributes"]["languageModelTools"]}

        # Tool names are now identical to MCP (snake_case, no axon_ prefix)

        _CAP_TO_TOOL = {
            "query": "query_knowledge",
            "search": "search_knowledge",
            "ingest_text": "ingest_text",
            "ingest_url": "ingest_url",
            "ingest_path": "ingest_path",
            "ingest_refresh": "refresh_ingest",
            "ingest_stale": "get_stale_docs",
            "collection_inspect": "list_knowledge",
            "collection_delete": "delete_documents",
            "collection_clear": "clear_knowledge",
            "project_list": "list_projects",
            "project_switch": "switch_project",
            "project_create": "create_project",
            "project_delete": "delete_project",
            "config_update": "update_settings",
            "config_read": "get_current_settings",
            "store_status": "get_store_status",
            "store_init": "init_store",
            "share_generate": "share_project",
            "share_redeem": "redeem_share",
            "share_revoke": "revoke_share",
            "share_list": "list_shares",
            "graph_status": "graph_status",
            "graph_finalize": "graph_finalize",
            "active_leases": "get_active_leases",
        }

        for cap in REGISTRY:
            if cap.tier not in (Tier.ONE, Tier.TWO) or Surface.VSCODE not in cap.supported_surfaces:
                continue

            expected_tool = _CAP_TO_TOOL.get(cap.id)

            if expected_tool:
                assert expected_tool in tool_names, (
                    f"Tier 1 capability '{cap.id}' requires VS Code tool '{expected_tool}' "
                    f"but it is missing from manifest"
                )

    def test_get_settings_and_finalize_in_manifest(self):
        """get_current_settings and graph_finalize should be in manifest."""

        manifest = _extension_manifest()

        tool_names = {t["name"] for t in manifest["contributes"]["languageModelTools"]}

        assert "get_current_settings" in tool_names

        assert "graph_finalize" in tool_names


# ---------------------------------------------------------------------------


# CLI surface contract


# ---------------------------------------------------------------------------


class TestCliSurfaceContract:

    """CLI source must expose all Tier 1 capabilities mapped to CLI."""

    def test_tier1_cli_capabilities_in_source(self):
        """Every Tier 1 CLI capability has a corresponding argparse flag."""

        from axon.surface_contract import REGISTRY, Surface, Tier

        cli_src = _cli_source()

        # Map capability id to expected CLI flag/identifier in cli.py

        _CAP_TO_CLI_FLAG = {
            "query": '"query"',
            "search": "--dry-run",  # search_raw is dry-run mode
            "ingest_text": "--ingest",
            "ingest_url": "--ingest",  # same path handles URLs
            "ingest_path": "--ingest",
            "ingest_refresh": "--refresh",
            "ingest_stale": "--list-stale",
            "collection_inspect": "--list",
            "collection_clear": None,  # CLI clear not yet wired — intentional exception
            "project_list": "--project-list",
            "project_switch": "--project",
            "project_create": "--project-new",
            "project_delete": "--project-delete",
            "config_update": "sentence_window",  # new flags are config update surface
            "collection_delete": "--delete-doc",
            "graph_status": "--graph-status",
            "store_init": "--store-init",
            "share_generate": "--share-generate",
            "share_redeem": "--share-redeem",
            "share_revoke": "--share-revoke",
            "share_list": "--share-list",
            "session_list": "--session-list",
        }

        for cap in REGISTRY:
            if cap.tier != Tier.ONE or Surface.CLI not in cap.supported_surfaces:
                continue

            flag = _CAP_TO_CLI_FLAG.get(cap.id)

            if flag is None:
                continue  # explicitly skipped

            assert (
                flag in cli_src
            ), f"Tier 1 CLI capability '{cap.id}' expects flag/pattern '{flag}' in cli.py but not found"

    def test_modern_rag_flags_present(self):
        """The four new retrieval flags from SP-030 are in cli.py."""

        cli_src = _cli_source()

        for flag in (
            "--sentence-window",
            "--sentence-window-size",
            "--crag-lite",
            "--graph-rag-mode",
        ):
            assert flag in cli_src, f"Missing CLI flag: {flag}"

    def test_operational_flags_present(self):
        """SP-031/SP-032 operational flags are in cli.py."""

        cli_src = _cli_source()

        for flag in (
            "--refresh",
            "--list-stale",
            "--graph-status",
            "--graph-finalize",
            "--graph-export",
        ):
            assert flag in cli_src, f"Missing CLI operational flag: {flag}"

    def test_tier2_cli_capabilities_in_source(self):
        """Tier 2 capabilities on CLI have matching patterns in cli.py."""

        from axon.surface_contract import REGISTRY, Surface, Tier

        cli_src = _cli_source()

        _TIER2_CLI_FLAGS = {
            "session_list": "--session-list",
        }

        for cap in REGISTRY:
            if cap.tier != Tier.TWO or Surface.CLI not in cap.supported_surfaces:
                continue

            flag = _TIER2_CLI_FLAGS.get(cap.id)

            if flag is None:
                continue

            assert (
                flag in cli_src
            ), f"Tier 2 CLI capability '{cap.id}' expects flag '{flag}' in cli.py but not found"

    def test_cli_has_full_tier1_parity(self):
        """Every Tier 1 capability is now supported on CLI — zero undeclared gaps."""

        from axon.surface_contract import REGISTRY, Surface, Tier

        missing = [
            cap.id
            for cap in REGISTRY
            if cap.tier == Tier.ONE and Surface.CLI not in cap.supported_surfaces
        ]

        assert not missing, f"Tier 1 capabilities not declared on CLI: {missing}"


# ---------------------------------------------------------------------------


# REPL surface contract


# ---------------------------------------------------------------------------


class TestReplSurfaceContract:

    """REPL source must expose all Tier 1 capabilities mapped to REPL."""

    def test_tier1_repl_capabilities_in_source(self):
        """Every Tier 1 REPL capability has a recognisable command pattern."""

        from axon.surface_contract import REGISTRY, Surface, Tier

        repl_src = _repl_source()

        _CAP_TO_REPL_PATTERN = {
            "query": "brain.query",
            "search": "/search",
            "ingest_text": "/ingest",
            "ingest_url": "/ingest",
            "ingest_path": "/ingest",
            "ingest_refresh": "/refresh",
            "ingest_stale": "/stale",
            "collection_inspect": "/list",
            "collection_clear": "/clear",
            "project_list": 'sub == "list"',
            "project_switch": 'sub == "switch"',
            "project_create": 'sub == "new"',
            "project_delete": 'sub == "delete"',
            "config_update": "/rag",
            "share_generate": "/share generate",
            "share_redeem": "/share redeem",
            "share_revoke": "/share revoke",
            "share_list": "share list",
            "graph_status": 'sub == "status"',
        }

        for cap in REGISTRY:
            if cap.tier != Tier.ONE or Surface.REPL not in cap.supported_surfaces:
                continue

            pattern = _CAP_TO_REPL_PATTERN.get(cap.id)

            if pattern is None:
                continue

            assert (
                pattern in repl_src
            ), f"Tier 1 REPL capability '{cap.id}' expects pattern '{pattern}' in repl.py but not found"

    def test_modern_rag_controls_in_repl(self):
        """New RAG controls from SP-022 are present in repl.py."""

        repl_src = _repl_source()

        for control in (
            "sentence-window",
            "sentence-window-size",
            "crag-lite",
            "code-graph",
            "graph-rag-mode",
        ):
            assert control in repl_src, f"Missing REPL RAG control: {control}"

    def test_special_scope_switch_in_repl(self):
        """REPL handles @projects, @mounts, @store virtual scopes (SP-020)."""

        repl_src = _repl_source()

        for scope in ("@projects", "@mounts", "@store"):
            assert scope in repl_src, f"Missing virtual scope handling in REPL: {scope}"


# ---------------------------------------------------------------------------


# Registry vs surface gap coherence


# ---------------------------------------------------------------------------


class TestRegistrySurfaceGapCoherence:

    """Intentional exceptions in the registry align with actual gaps in code."""

    def test_vscode_session_gap_is_documented(self):
        """session_list VS Code exception is a deliberate product decision."""

        from axon.surface_contract import Surface, unsupported_on

        vscode_gaps = {cap.id: reason for cap, reason in unsupported_on(Surface.VSCODE)}

        assert (
            "session_list" in vscode_gaps
        ), "session_list must have a documented VS Code exception (product decision)"

    def test_all_tier2_gaps_have_reasons(self):
        """Every Tier 2 capability missing from a surface has a documented reason."""

        from axon.surface_contract import REGISTRY, Surface, Tier

        for cap in REGISTRY:
            if cap.tier != Tier.TWO:
                continue

            for surface in Surface:
                if surface not in cap.supported_surfaces:
                    reason = cap.intentional_exceptions.get(surface, "")

                    assert reason, (
                        f"Tier 2 capability '{cap.id}' is not on {surface} "
                        f"but no documented exception reason exists"
                    )
