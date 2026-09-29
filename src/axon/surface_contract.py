"""Surface capability registry for Axon.

Defines which capabilities exist and which surfaces support them.

Surfaces split into two groups (0.5.0):

* **Human surfaces** — REST API, REPL, CLI. Every capability that has a human
  use lives on at least one of these.
* **Agent surfaces** — the MCP server (``axon-mcp``) and the VS Code Copilot
  Language Model tools. ``Surface.VSCODE`` means the LM tools *only*; the
  extension's own commands and webviews are human UI, not an agent surface.
  Agent surfaces carry a deliberately small tool set: destructive, credential
  and administrative operations are human-only, and every such gap names the
  human route in its intentional-exception reason (see :func:`_human_only`).

Tier 1 = required on every human surface (``HUMAN_SURFACES``); agent support
is decided per capability.

Tier 2 = required where practical; every unsupported surface carries an
intentional-exception reason.

API-only = intentionally REST-only administration.

Usage::

    from axon.surface_contract import REGISTRY, Tier, Surface
    tier1 = [c for c in REGISTRY if c.tier == Tier.ONE]
    mcp_caps = [c for c in REGISTRY if Surface.MCP in c.supported_surfaces]
"""


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    ONE = "tier1"  # required on every human surface
    TWO = "tier2"  # required where practical
    API_ONLY = "api_only"  # intentionally API-only


class Surface(str, Enum):
    API = "api"
    REPL = "repl"
    CLI = "cli"
    VSCODE = "vscode"  # Copilot Language Model tools only
    MCP = "mcp"


ALL_SURFACES = frozenset(Surface)


HUMAN_SURFACES = frozenset({Surface.API, Surface.REPL, Surface.CLI})


AGENT_SURFACES = frozenset({Surface.MCP, Surface.VSCODE})


def _human_only(reason: str, routes: str) -> dict[Surface, str]:
    """Intentional-exception reasons for a capability kept off the agent surfaces.

    *reason* says why (e.g. "Destructive"), *routes* names where a human does
    it instead, so an agent (or its user) reading the registry knows the
    operation still exists and how to reach it.
    """
    msg = f"{reason} — human-only (0.5.0): {routes}"
    return {Surface.MCP: msg, Surface.VSCODE: msg}


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    category: str
    tier: Tier
    description: str
    supported_surfaces: frozenset[Surface]
    intentional_exceptions: dict[Surface, str] = field(default_factory=dict)
    api_route: str = ""
    docs_targets: tuple[str, ...] = ()
    test_targets: tuple[str, ...] = ()


# ---------------------------------------------------------------------------


# Registry


# ---------------------------------------------------------------------------


REGISTRY: list[Capability] = [
    # ── Query / Search ───────────────────────────────────────────────────────
    Capability(
        id="query",
        name="Grounded query",
        category="query",
        tier=Tier.ONE,
        description="Answer a question grounded in the current project knowledge base.",
        supported_surfaces=ALL_SURFACES,
        api_route="/query",
        docs_targets=("REFERENCE.md",),
        test_targets=(
            "tests/test_api.py",
            "tests/test_repl_commands.py",
        ),
    ),
    Capability(
        id="query_stream",
        name="Streaming query",
        category="query",
        tier=Tier.TWO,
        description="Stream a grounded answer token-by-token.",
        supported_surfaces=frozenset({Surface.API, Surface.CLI}),
        intentional_exceptions={
            Surface.REPL: "REPL renders tokens incrementally via print; no separate mode needed",
            Surface.VSCODE: (
                "LM tool results are single-shot; query_knowledge returns the whole answer "
                "(the @axon chat participant streams for humans)"
            ),
            Surface.MCP: (
                "MCP tool results are single-shot, so a streaming tool only re-implemented "
                "query_knowledge; removed in 0.5.0"
            ),
        },
        api_route="/query/stream",
    ),
    Capability(
        id="search",
        name="Vector search",
        category="query",
        tier=Tier.ONE,
        description="Return ranked document chunks without generating an answer.",
        supported_surfaces=ALL_SURFACES,
        api_route="/search",
    ),
    Capability(
        id="search_raw",
        name="Raw retrieval diagnostics",
        category="query",
        tier=Tier.TWO,
        description="Return full retrieval diagnostics including scores and metadata.",
        supported_surfaces=frozenset({Surface.API, Surface.CLI}),
        intentional_exceptions={
            Surface.REPL: "Available via /query --dry-run equivalent",
            Surface.VSCODE: "Retrieval-tuning diagnostics for humans; agents use search_knowledge",
            Surface.MCP: "Retrieval-tuning diagnostics for humans; agents use search_knowledge",
        },
        api_route="/search/raw",
    ),
    # ── Ingest ───────────────────────────────────────────────────────────────
    # On the agent surfaces all four ingest capabilities are one tool,
    # ``ingest_knowledge`` (text | docs | url | path | refresh — exactly one).
    Capability(
        id="ingest_text",
        name="Ingest text",
        category="ingest",
        tier=Tier.ONE,
        description="Add raw text to the knowledge base.",
        supported_surfaces=ALL_SURFACES,
        api_route="/add_text",
    ),
    Capability(
        id="ingest_url",
        name="Ingest URL",
        category="ingest",
        tier=Tier.ONE,
        description="Fetch and ingest a web page by URL.",
        supported_surfaces=ALL_SURFACES,
        api_route="/ingest_url",
    ),
    Capability(
        id="ingest_path",
        name="Ingest path",
        category="ingest",
        tier=Tier.ONE,
        description="Ingest a file or directory from the local filesystem.",
        supported_surfaces=ALL_SURFACES,
        api_route="/ingest",
    ),
    Capability(
        id="ingest_refresh",
        name="Refresh changed docs",
        category="ingest",
        tier=Tier.ONE,
        description="Re-ingest documents whose content has changed since last ingest.",
        supported_surfaces=ALL_SURFACES,
        api_route="/ingest/refresh",
    ),
    Capability(
        id="ingest_stale",
        name="List stale docs",
        category="ingest",
        tier=Tier.ONE,
        description="List documents that have not been re-ingested within a configurable window.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Collection maintenance",
            "REPL /stale, `axon --list-stale`, REST GET /collection/stale, "
            "VS Code command axon.listStaleDocs",
        ),
        api_route="/collection/stale",
    ),
    # ── Collection ───────────────────────────────────────────────────────────
    Capability(
        id="collection_inspect",
        name="Inspect collection",
        category="collection",
        tier=Tier.ONE,
        description="List ingested documents and chunk counts.",
        supported_surfaces=ALL_SURFACES,
        api_route="/collection",
    ),
    Capability(
        id="collection_delete",
        name="Delete documents",
        category="collection",
        tier=Tier.ONE,
        description="Remove chunks or whole documents by ID, clearing their dedup records.",
        supported_surfaces=ALL_SURFACES,
        api_route="/delete",
    ),
    Capability(
        id="collection_clear",
        name="Clear knowledge base",
        category="collection",
        tier=Tier.ONE,
        description="Delete all documents in the current project.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Destructive",
            "REPL /clear, `axon --clear --yes`, REST POST /clear, "
            "VS Code command axon.clearKnowledgeBase",
        ),
        api_route="/clear",
    ),
    # ── Project ──────────────────────────────────────────────────────────────
    Capability(
        id="project_list",
        name="List projects",
        category="project",
        tier=Tier.ONE,
        description="List all projects available to the current user.",
        supported_surfaces=ALL_SURFACES,
        api_route="/projects",
    ),
    Capability(
        id="project_switch",
        name="Switch project",
        category="project",
        tier=Tier.ONE,
        description="Change the active project for subsequent operations.",
        supported_surfaces=ALL_SURFACES,
        api_route="/project/switch",
    ),
    Capability(
        id="project_create",
        name="Create project",
        category="project",
        tier=Tier.ONE,
        description="Create a new project.",
        supported_surfaces=ALL_SURFACES,
        api_route="/project/new",
    ),
    Capability(
        id="project_delete",
        name="Delete project",
        category="project",
        tier=Tier.ONE,
        description="Delete a project and all its stored knowledge.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Destructive",
            "REPL /project delete <name>, `axon --project-delete <name>`, "
            "REST POST /project/delete/{name}",
        ),
        api_route="/project/delete/{name}",
    ),
    # ── Config ───────────────────────────────────────────────────────────────
    # Agent surfaces: set_config(settings: dict, persist=False) -> POST /config/set
    # (batch form); get_config(validate=False) -> GET /config (+ /config/validate).
    Capability(
        id="config_update",
        name="Update settings",
        category="config",
        tier=Tier.ONE,
        description="Change active retrieval and generation settings.",
        supported_surfaces=ALL_SURFACES,
        api_route="/config/update",
    ),
    Capability(
        id="config_read",
        name="Read settings",
        category="config",
        tier=Tier.TWO,
        description="Inspect current retrieval and generation settings.",
        supported_surfaces=ALL_SURFACES,
        api_route="/config",
    ),
    # ── Share / Store ────────────────────────────────────────────────────────
    Capability(
        id="store_status",
        name="Store status",
        category="store",
        tier=Tier.TWO,
        description="Check whether the AxonStore is initialised and return its metadata.",
        supported_surfaces=frozenset({Surface.API}),
        intentional_exceptions={
            Surface.REPL: "REPL always launches with an initialised store — startup guarantees it",
            Surface.CLI: "CLI always launches with an initialised store — startup guarantees it",
            **_human_only("Store administration", "REST GET /store/status"),
        },
        api_route="/store/status",
    ),
    Capability(
        id="store_init",
        name="Init store",
        category="store",
        tier=Tier.ONE,
        description="Move the AxonStore to a different base path (e.g. a shared drive).",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Store administration (moves every project)",
            "REPL /store init <path>, `axon --store-init <path>`, REST POST /store/init, "
            "VS Code command axon.initStore",
        ),
        api_route="/store/init",
    ),
    Capability(
        id="share_generate",
        name="Generate share",
        category="share",
        tier=Tier.ONE,
        description="Create an HMAC share token for a grantee.",
        supported_surfaces=ALL_SURFACES,
        api_route="/share/generate",
    ),
    Capability(
        id="share_redeem",
        name="Redeem share",
        category="share",
        tier=Tier.ONE,
        description="Mount a shared project using a share token.",
        supported_surfaces=ALL_SURFACES,
        api_route="/share/redeem",
    ),
    Capability(
        id="share_revoke",
        name="Revoke share",
        category="share",
        tier=Tier.ONE,
        description=(
            "Revoke an active share grant. The agent tools (MCP, VS Code) do a soft "
            "revoke only; hard revoke with DEK rotation is human-only: REPL "
            "/share revoke <ssk_id> --project <name> --rotate, `axon --share-rotate`, "
            "REST POST /share/revoke {rotate: true}."
        ),
        supported_surfaces=ALL_SURFACES,
        api_route="/share/revoke",
    ),
    Capability(
        id="share_list",
        name="List shares",
        category="share",
        tier=Tier.ONE,
        description="List active shares granted and received.",
        supported_surfaces=ALL_SURFACES,
        api_route="/share/list",
    ),
    # ── Graph ────────────────────────────────────────────────────────────────
    Capability(
        id="graph_status",
        name="Graph status",
        category="graph",
        tier=Tier.ONE,
        description="Show entity count, code node count, and community build state.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Graph administration",
            "REPL /graph status, `axon --graph-status`, REST GET /graph/status, "
            "VS Code command axon.showGraphStatus",
        ),
        api_route="/graph/status",
    ),
    Capability(
        id="graph_finalize",
        name="Graph finalize",
        category="graph",
        tier=Tier.TWO,
        description="Rebuild community summaries and finalize the knowledge graph.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Graph administration (long-running LLM rebuild)",
            "REPL /graph finalize, `axon --graph-finalize`, REST POST /graph/finalize",
        ),
        api_route="/graph/finalize",
    ),
    Capability(
        id="graph_data",
        name="Graph data export",
        category="graph",
        tier=Tier.TWO,
        description="Return the full entity/relation knowledge-graph as a JSON nodes+links payload.",
        supported_surfaces=frozenset({Surface.API}),
        intentional_exceptions={
            Surface.REPL: "Raw JSON payload not useful in interactive REPL; use graph_viz instead",
            Surface.CLI: "Raw JSON payload not useful in CLI; use graph_viz instead",
            **_human_only(
                "Whole-graph dump (too large for agent context; agents use graph_retrieve)",
                "REST GET /graph/data, VS Code graph panel (axon.showGraphForQuery)",
            ),
        },
        api_route="/graph/data",
    ),
    Capability(
        id="graph_viz",
        name="Graph visualization",
        category="graph",
        tier=Tier.TWO,
        description="Export the entity graph as an interactive HTML visualization.",
        supported_surfaces=frozenset({Surface.API, Surface.REPL, Surface.CLI}),
        intentional_exceptions={
            Surface.VSCODE: "Graph panel exists but uses /graph/data; separate from viz export",
            Surface.MCP: "HTML visualisation is for humans; agents use graph_retrieve",
        },
        api_route="/graph/visualize",
    ),
    Capability(
        id="graph_conflicts",
        name="Graph conflict inspection",
        category="graph",
        tier=Tier.TWO,
        description=(
            "List facts whose status='conflicted' (incompatible exclusive-relation "
            "facts in the same scope). Returns supported=false on backends that do "
            "not track conflicts (e.g. graphrag)."
        ),
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Graph administration (conflicts are resolved by a human)",
            "REPL /graph conflicts, `axon --graph-conflicts`, REST GET /graph/conflicts",
        ),
        api_route="/graph/conflicts",
    ),
    Capability(
        id="graph_retrieve",
        name="Graph backend retrieve (point-in-time)",
        category="graph",
        tier=Tier.TWO,
        description=(
            "Run the active graph backend's retrieve() directly with a "
            "RetrievalConfig — surfaces point_in_time historical queries and "
            "per-query federation_weights overrides without an LLM call."
        ),
        supported_surfaces=ALL_SURFACES,
        api_route="/graph/retrieve",
    ),
    Capability(
        id="graph_fact_update",
        name="Graph fact update (agent-writable)",
        category="graph",
        tier=Tier.TWO,
        description=(
            "Assert or correct one fact (subject, relation, object) in the active "
            "project's graph with bi-temporal supersession — replace or add mode. "
            "dynamic_graph and federated projects store it; graphrag and none "
            "answer status='not_applicable'."
        ),
        supported_surfaces=ALL_SURFACES,
        api_route="/graph/facts",
    ),
    # ── Session ──────────────────────────────────────────────────────────────
    Capability(
        id="session_list",
        name="List sessions",
        category="session",
        tier=Tier.TWO,
        description="List saved conversation sessions for the current project.",
        supported_surfaces=frozenset({Surface.API, Surface.REPL, Surface.CLI}),
        intentional_exceptions={
            Surface.VSCODE: "Session management is a REPL/CLI workflow; extension focuses on single-turn tool calls",
            Surface.MCP: (
                "Saved sessions are the human REPL/CLI's chat history; agents keep their "
                "own context. Human routes: REPL /sessions, `axon --session-list`, REST GET /sessions"
            ),
        },
        api_route="/sessions",
    ),
    # ── Maintenance ───────────────────────────────────────────────────────────
    Capability(
        id="active_leases",
        name="Active leases",
        category="maintenance",
        tier=Tier.API_ONLY,
        description="List active write-lease counts per project; used to confirm it is safe to enter maintenance state.",
        supported_surfaces=frozenset({Surface.API}),
        intentional_exceptions={
            Surface.REPL: "Operator diagnostic paired with API-only maintenance state control",
            Surface.CLI: "Operator diagnostic paired with API-only maintenance state control",
            **_human_only("Operator diagnostic", "REST GET /registry/leases"),
        },
        api_route="/registry/leases",
    ),
    Capability(
        id="maintenance_state",
        name="Maintenance state control",
        category="maintenance",
        tier=Tier.API_ONLY,
        description="Set project maintenance state (normal, readonly, rebuilding).",
        supported_surfaces=frozenset({Surface.API}),
        intentional_exceptions={
            Surface.REPL: "API-only administrative operation",
            Surface.CLI: "API-only administrative operation",
            Surface.VSCODE: "API-only administrative operation",
            Surface.MCP: "API-only administrative operation",
        },
        api_route="/project/maintenance",
    ),
    # ── Share extend (SP-B1 parity sweep) ────────────────────────────────────
    Capability(
        id="share_extend",
        name="Extend share expiry",
        category="share",
        tier=Tier.TWO,
        description="Renew or clear a share key's TTL so it stays valid without revoking and re-issuing.",
        supported_surfaces=ALL_SURFACES,
        api_route="/share/extend",
    ),
    # ── Store whoami (SP-B1 parity sweep) ────────────────────────────────────
    Capability(
        id="store_whoami",
        name="Store identity",
        category="store",
        tier=Tier.TWO,
        description="Return the current user's OS identity, store path, user directory, and active project.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Store administration",
            "REPL /store whoami, `axon --store-whoami`, REST GET /store/whoami",
        ),
        api_route="/store/whoami",
    ),
    # ── Seal project (SP-B1 parity sweep) ────────────────────────────────────
    Capability(
        id="seal_project",
        name="Seal project",
        category="security",
        tier=Tier.TWO,
        description="Encrypt every content file in a project at rest using AES-256-GCM.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Credential / encryption-at-rest operation",
            "REPL /project seal <name>, `axon --project-seal <name>`, REST POST /project/seal",
        ),
        api_route="/project/seal",
    ),
    # ── Pack / unpack project ─────────────────────────────────────────────────
    Capability(
        id="pack_project",
        name="Pack project",
        category="project",
        tier=Tier.TWO,
        description=(
            "Zip a project's entire on-disk footprint (index files, sessions, "
            "sub-projects, and .security/ if sealed) for backup, restore, or relocation."
        ),
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Backup / filesystem operation",
            "REPL /project pack <name>, `axon --project-pack <name>`, REST POST /project/pack",
        ),
        api_route="/project/pack",
    ),
    Capability(
        id="unpack_project",
        name="Unpack project",
        category="project",
        tier=Tier.TWO,
        description="Restore a project from a zip archive produced by pack_project into AxonStore.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Restore / filesystem operation (can overwrite a project)",
            "REPL /project unpack <path>, `axon --project-unpack <path>`, REST POST /project/unpack",
        ),
        api_route="/project/unpack",
    ),
    # ── Mount refresh (SP-B1 parity sweep) ───────────────────────────────────
    Capability(
        id="mount_refresh",
        name="Refresh mount",
        category="project",
        tier=Tier.TWO,
        description="Re-read the owner's version marker for an active mounted share and reopen project handles.",
        supported_surfaces=HUMAN_SURFACES,
        intentional_exceptions=_human_only(
            "Mount administration",
            "REPL /mount-refresh, `axon --mount-refresh`, REST POST /mount/refresh",
        ),
        api_route="/mount/refresh",
    ),
]


def capabilities_by_category() -> dict[str, list[Capability]]:
    """Return registry grouped by category."""
    groups: dict[str, list[Capability]] = {}
    for cap in REGISTRY:
        groups.setdefault(cap.category, []).append(cap)
    return groups


def tier1_capabilities() -> list[Capability]:
    """Return all Tier 1 capabilities."""
    return [c for c in REGISTRY if c.tier == Tier.ONE]


def surface_capabilities(surface: Surface) -> list[Capability]:
    """Return all capabilities supported on *surface*."""
    return [c for c in REGISTRY if surface in c.supported_surfaces]


def unsupported_on(surface: Surface) -> list[tuple[Capability, str]]:
    """Return (capability, reason) pairs for capabilities NOT on *surface*.
    Only includes Tier 1 and Tier 2 capabilities — API-only items are excluded.
    """
    result = []
    for cap in REGISTRY:
        if cap.tier == Tier.API_ONLY:
            continue
        if surface not in cap.supported_surfaces:
            reason = cap.intentional_exceptions.get(surface, "no explicit exception documented")
            result.append((cap, reason))
    return result
