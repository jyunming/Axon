"""Cross-surface parity contract tests (SP-052).


Validates that the surface capability registry is consistent with:


- VS Code extension manifest (package.json tool declarations) and extension.ts
  registrations


- MCP tool set (mcp_server.py ``@mcp.tool()`` registrations)


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


def _mcp_tool_names() -> set[str]:
    """MCP tools as registered in source — the same regex as
    tests/test_config_surface_parity.py, so an undecorated coroutine doesn't count."""
    import re

    src = (REPO_ROOT / "src" / "axon" / "mcp_server.py").read_text(encoding="utf-8")
    return set(re.findall(r"@mcp\.tool\(\)\s*\nasync def (\w+)", src))


def _manifest_tool_names() -> set[str]:
    return {t["name"] for t in _extension_manifest()["contributes"]["languageModelTools"]}


def _extension_registered_tool_names() -> list[str]:
    import re

    src = (_extension_root() / "src" / "extension.ts").read_text(encoding="utf-8")
    return re.findall(r"registerTool\(\s*'([^']+)'", src)


# Capability id -> agent tool name. Shared by MCP and the VS Code LM tools
# (their names are identical). Several capabilities fold into one tool.
_CAP_TO_AGENT_TOOL = {
    "query": "query_knowledge",
    "search": "search_knowledge",
    "ingest_text": "ingest_knowledge",
    "ingest_url": "ingest_knowledge",
    "ingest_path": "ingest_knowledge",
    "ingest_refresh": "ingest_knowledge",
    "collection_inspect": "list_knowledge",
    "collection_delete": "delete_documents",
    "project_list": "list_projects",
    "project_switch": "switch_project",
    "project_create": "create_project",
    "config_update": "set_config",
    "config_read": "get_config",
    "graph_retrieve": "graph_retrieve",
    "graph_fact_update": "update_fact",
    "share_generate": "share_project",
    "share_redeem": "redeem_share",
    "share_revoke": "revoke_share",
    "share_list": "list_shares",
    "share_extend": "extend_share",
}

# Agent tools that are plumbing or client-side features, not registry capabilities.
_MCP_TOOLS_WITHOUT_CAPABILITY = {"get_job_status"}
_VSCODE_TOOLS_WITHOUT_CAPABILITY = {"get_job_status", "show_graph", "ingest_image"}

# Destructive / credential operations that must never be agent-callable.
_HUMAN_ONLY_TOOL_NAMES = {
    "delete_project",
    "clear_knowledge",
    "security_change_passphrase",
    "security_bootstrap",
    "security_unlock",
    "security_lock",
    "init_store",
    "seal_project",
    "pack_project",
    "unpack_project",
}


def _expected_agent_tools(surface) -> set[str]:
    from axon.surface_contract import REGISTRY

    return {
        _CAP_TO_AGENT_TOOL[c.id]
        for c in REGISTRY
        if surface in c.supported_surfaces and c.id in _CAP_TO_AGENT_TOOL
    }


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

    def test_tier1_is_on_every_human_surface(self):
        """Tier 1 means "required on every human surface" (API, REPL, CLI)."""
        from axon.surface_contract import HUMAN_SURFACES, REGISTRY, Tier

        for cap in REGISTRY:
            if cap.tier == Tier.ONE:
                assert HUMAN_SURFACES <= cap.supported_surfaces, cap.id

    def test_every_capability_has_a_human_surface(self):
        """Nothing may be agent-only: a human can always reach every capability."""
        from axon.surface_contract import HUMAN_SURFACES, REGISTRY

        for cap in REGISTRY:
            assert cap.supported_surfaces & HUMAN_SURFACES, cap.id

    def test_agent_gaps_have_reasons(self):
        """A capability kept off MCP/VS Code must say why (and, for human-only
        operations, where a human does it)."""
        from axon.surface_contract import AGENT_SURFACES, REGISTRY

        for cap in REGISTRY:
            for surface in AGENT_SURFACES - cap.supported_surfaces:
                reason = cap.intentional_exceptions.get(surface, "")
                assert reason, f"{cap.id} missing {surface} exception"

    def test_human_only_reasons_name_a_route(self):
        from axon.surface_contract import REGISTRY, Surface

        for cap in REGISTRY:
            reason = cap.intentional_exceptions.get(Surface.MCP, "")
            if "human-only" in reason:
                assert "REST" in reason, f"{cap.id}: human-only reason names no route"

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
        """0.5.0: the Copilot LM tool set is the 18 MCP tools + show_graph + ingest_image."""
        tools = _extension_manifest()["contributes"]["languageModelTools"]
        assert (
            len(tools) == 20
        ), f"Expected 20 tools, got {len(tools)}: {[t['name'] for t in tools]}"

    def test_manifest_matches_vscode_capabilities(self):
        """The manifest declares exactly the tools of the VS Code-supported
        capabilities, plus the client-side extras — nothing more, nothing less."""
        from axon.surface_contract import Surface

        expected = _expected_agent_tools(Surface.VSCODE) | _VSCODE_TOOLS_WITHOUT_CAPABILITY
        actual = _manifest_tool_names()
        assert (
            actual == expected
        ), f"missing: {sorted(expected - actual)}  extra: {sorted(actual - expected)}"

    def test_every_vscode_capability_has_a_tool_mapping(self):
        from axon.surface_contract import REGISTRY, Surface, Tier

        unmapped = [
            c.id
            for c in REGISTRY
            if c.tier in (Tier.ONE, Tier.TWO)
            and Surface.VSCODE in c.supported_surfaces
            and c.id not in _CAP_TO_AGENT_TOOL
        ]
        assert not unmapped, f"VS Code capabilities with no LM tool mapping: {unmapped}"

    def test_manifest_shares_mcp_tool_names(self):
        """Where a tool exists on both agent surfaces it has the same name."""
        shared = _manifest_tool_names() - _VSCODE_TOOLS_WITHOUT_CAPABILITY
        assert shared <= _mcp_tool_names(), sorted(shared - _mcp_tool_names())

    def test_old_settings_and_admin_tools_removed_from_manifest(self):
        names = _manifest_tool_names()
        for gone in (
            "get_current_settings",
            "update_settings",
            "graph_finalize",
            "graph_status",
            "axonConfigSet",
            "axonConfigValidate",
            "get_active_leases",
        ):
            assert gone not in names, gone

    def test_extension_registrations_match_manifest(self):
        """registerTool() names == manifest languageModelTools ==
        onLanguageModelTool activation events. A registered-but-undeclared tool
        is dead code; a declared-but-unregistered tool errors when Copilot calls it."""
        registered = _extension_registered_tool_names()
        assert len(registered) == len(set(registered)), f"duplicate registerTool: {registered}"
        manifest = _extension_manifest()
        declared = _manifest_tool_names()
        events = {
            e.split(":", 1)[1]
            for e in manifest.get("activationEvents", [])
            if e.startswith("onLanguageModelTool:")
        }
        assert set(registered) == declared, (
            f"registered-only: {sorted(set(registered) - declared)}  "
            f"declared-only: {sorted(declared - set(registered))}"
        )
        assert (
            events == declared
        ), f"event-only: {sorted(events - declared)}  no-event: {sorted(declared - events)}"

    def test_governance_panel_removed(self):
        manifest = _extension_manifest()
        commands = {c["command"] for c in manifest["contributes"].get("commands", [])}
        assert "axon.showGovernancePanel" not in commands
        src = (_extension_root() / "src" / "extension.ts").read_text(encoding="utf-8")
        assert "governance" not in src.lower()
        assert not (_extension_root() / "src" / "governance" / "panel.ts").exists()


# ---------------------------------------------------------------------------
# MCP surface contract
# ---------------------------------------------------------------------------
class TestMcpSurfaceContract:

    """The MCP tool set is exactly the tools of the MCP-supported capabilities."""

    def test_mcp_tools_match_mcp_capabilities(self):
        from axon.surface_contract import Surface

        expected = _expected_agent_tools(Surface.MCP) | _MCP_TOOLS_WITHOUT_CAPABILITY
        actual = _mcp_tool_names()
        assert (
            actual == expected
        ), f"missing: {sorted(expected - actual)}  extra: {sorted(actual - expected)}"

    def test_every_mcp_capability_has_a_tool_mapping(self):
        from axon.surface_contract import REGISTRY, Surface, Tier

        unmapped = [
            c.id
            for c in REGISTRY
            if c.tier in (Tier.ONE, Tier.TWO)
            and Surface.MCP in c.supported_surfaces
            and c.id not in _CAP_TO_AGENT_TOOL
        ]
        assert not unmapped, f"MCP capabilities with no tool mapping: {unmapped}"

    def test_mcp_tool_count(self):
        assert len(_mcp_tool_names()) == 18

    def test_agent_surfaces_carry_the_same_capabilities(self):
        from axon.surface_contract import REGISTRY, Surface

        mcp = {c.id for c in REGISTRY if Surface.MCP in c.supported_surfaces}
        vscode = {c.id for c in REGISTRY if Surface.VSCODE in c.supported_surfaces}
        assert mcp == vscode


class TestDestructiveOpsAbsentFromAgentSurfaces:
    def test_destructive_ops_absent_from_agent_surfaces(self):
        leaked_mcp = sorted(_HUMAN_ONLY_TOOL_NAMES & _mcp_tool_names())
        assert not leaked_mcp, f"human-only ops exposed over MCP: {leaked_mcp}"
        if _extension_root().exists():
            leaked_vs = sorted(_HUMAN_ONLY_TOOL_NAMES & _manifest_tool_names())
            assert not leaked_vs, f"human-only ops in the VS Code manifest: {leaked_vs}"
            registered = set(_extension_registered_tool_names())
            leaked_reg = sorted(_HUMAN_ONLY_TOOL_NAMES & registered)
            assert not leaked_reg, f"human-only ops registered as LM tools: {leaked_reg}"

    def test_no_agent_tool_takes_rotate(self):
        """Share-key rotation (hard revoke) is human-only on every agent tool."""
        import inspect

        import axon.mcp_server as mod

        for name in _mcp_tool_names():
            params = inspect.signature(getattr(mod, name)).parameters
            assert "rotate" not in params, f"MCP tool {name} exposes rotate"
        if _extension_root().exists():
            for tool in _extension_manifest()["contributes"]["languageModelTools"]:
                props = tool.get("inputSchema", {}).get("properties", {})
                assert "rotate" not in props, f"VS Code tool {tool['name']} exposes rotate"

    def test_vscode_revoke_share_never_sends_rotate(self):
        if not _extension_root().exists():
            pytest.skip("VS Code extension directory not found")
        src = (_extension_root() / "src" / "tools" / "shares.ts").read_text(encoding="utf-8")
        assert "rotate" not in src


# ---------------------------------------------------------------------------
# CLI surface contract
# ---------------------------------------------------------------------------
class TestCliSurfaceContract:

    """CLI source must expose all Tier 1 capabilities mapped to CLI."""

    def test_clear_requires_yes(self):
        cli_src = _cli_source()
        assert '"--clear"' in cli_src
        assert '"--yes"' in cli_src

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
            "collection_clear": "--clear",
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
            "graph_fact_update": "--graph-fact",
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


# ---------------------------------------------------------------------------
# SP-B1 parity sweep: new capabilities added in the batch
# ---------------------------------------------------------------------------
class TestB1ParitySweep:

    """New capabilities from the B1 parity sweep are present in all required surfaces."""

    def test_b1_capabilities_in_registry(self):
        """The surviving SP-B1 capability IDs are present in the registry."""
        from axon.surface_contract import REGISTRY

        expected = {
            "share_extend",
            "store_whoami",
            "seal_project",
            "mount_refresh",
        }
        registered = {c.id for c in REGISTRY}
        missing = expected - registered
        assert not missing, f"SP-B1 capabilities missing from registry: {missing}"

    def test_mount_refresh_command_in_repl(self):
        """/mount-refresh is present in repl.py."""
        from axon.repl import _SLASH_COMMANDS

        # Entries may have trailing spaces (e.g. "/share " for subcommand completion)
        slash_cmd_prefixes = {c.strip() for c in _SLASH_COMMANDS}
        assert "/mount-refresh" in slash_cmd_prefixes, "Missing /mount-refresh in _SLASH_COMMANDS"
        # Verify handlers exist via source (broad search, not quote-style-sensitive)
        repl_src = _repl_source()
        assert "mount-refresh" in repl_src, "Missing /mount-refresh handler body in repl.py"

    def test_b1_flags_in_cli(self):
        """--share-extend, --store-whoami, --mount-refresh are in cli.py."""
        cli_src = _cli_source()
        for flag in ("--share-extend", "--store-whoami", "--mount-refresh"):
            assert flag in cli_src, f"Missing CLI flag: {flag}"

    def test_b1_vscode_tools_in_manifest(self):
        """extend_share stays an LM tool; seal_project and store_whoami became
        human-only in 0.5.0 (CLI/REPL/REST)."""
        if not _extension_root().exists():
            pytest.skip("VS Code extension directory not found")
        tool_names = _manifest_tool_names()
        assert "extend_share" in tool_names
        for tool in ("seal_project", "store_whoami"):
            assert tool not in tool_names, f"{tool} is human-only since 0.5.0"

    def test_share_extend_in_repl(self):
        """/share extend handler is present in repl.py."""
        repl_src = _repl_source()
        # Check for the handler by presence of the sub-command keyword (not quote-style-sensitive)
        assert (
            '"extend"' in repl_src or "'extend'" in repl_src
        ), "Missing /share extend handler in repl.py"


# ---------------------------------------------------------------------------
# PR5b: agent-writable fact update (graph_fact_update)
# ---------------------------------------------------------------------------
class TestGraphFactUpdateContract:
    """graph_fact_update is registered and actually wired on every surface it
    claims (API route, CLI flag, REPL sub-command, MCP tool, VS Code LM tool)."""

    def _cap(self):
        from axon.surface_contract import REGISTRY

        (cap,) = (c for c in REGISTRY if c.id == "graph_fact_update")
        return cap

    def test_registered_as_tier2_graph_capability(self):
        from axon.surface_contract import Surface, Tier

        cap = self._cap()
        assert cap.tier == Tier.TWO
        assert cap.category == "graph"
        assert cap.api_route == "/graph/facts"
        assert cap.supported_surfaces == frozenset(Surface)
        assert not cap.intentional_exceptions

    def test_agent_tools_exist(self):
        assert "update_fact" in _mcp_tool_names()
        if _extension_root().exists():
            assert "update_fact" in _manifest_tool_names()
            assert "update_fact" in _extension_registered_tool_names()

    def test_api_route_exists(self):
        from axon.api_routes import graph

        routes = {(r.path, m) for r in graph.router.routes for m in getattr(r, "methods", ())}
        assert ("/graph/facts", "POST") in routes

    def test_cli_flags_exist(self):
        cli_src = _cli_source()
        for flag in ("--graph-fact", "--graph-fact-mode", "--graph-fact-desc"):
            assert flag in cli_src, f"Missing CLI flag: {flag}"

    def test_repl_subcommand_exists(self):
        repl_src = _repl_source()
        assert 'sub == "fact"' in repl_src
        assert "brain.update_fact(" in repl_src
