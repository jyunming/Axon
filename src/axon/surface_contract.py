"""Surface capability registry for Axon.















Defines which capabilities exist and which surfaces support them.







Tier 1 = required on every supported surface.







Tier 2 = required where practical; intentional exceptions are noted.















Usage::















    from axon.surface_contract import REGISTRY, Tier, Surface















    tier1 = [c for c in REGISTRY if c.tier == Tier.ONE]







    repl_caps = [c for c in REGISTRY if Surface.REPL in c.supported_surfaces]







"""


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    ONE = "tier1"  # required everywhere

    TWO = "tier2"  # required where practical

    API_ONLY = "api_only"  # intentionally API-only


class Surface(str, Enum):
    API = "api"

    REPL = "repl"

    CLI = "cli"

    VSCODE = "vscode"

    # Streamlit web UI.  Intentionally a supplemental surface — destructive and

    # admin-oriented capabilities (graph, sharing, store) are not exposed there.

    WEBAPP = "webapp"


ALL_SURFACES = frozenset(Surface)


NO_VSCODE = frozenset({Surface.API, Surface.REPL, Surface.CLI})


# Streamlit is intentionally excluded from capabilities that are:


#   (a) administrative / destructive (delete, graph finalize, store ops)


#   (b) multi-user sharing (share keys, store migration)


#   (c) MCP/agent-oriented (raw data payloads, lease inspection)


PRIMARY_SURFACES = frozenset({Surface.API, Surface.REPL, Surface.CLI, Surface.VSCODE})


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
        docs_targets=("API_REFERENCE.md", "QUICKREF.md"),
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
        supported_surfaces=frozenset({Surface.API, Surface.CLI, Surface.VSCODE}),
        intentional_exceptions={
            Surface.REPL: "REPL renders tokens incrementally via print; no separate mode needed",
            Surface.WEBAPP: "Streamlit uses a blocking query model; streaming not exposed",
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
            Surface.VSCODE: "Not exposed as a tool; requires explicit decision to add",
            Surface.WEBAPP: "Developer diagnostic tool not exposed in Streamlit UI",
        },
        api_route="/search/raw",
    ),
    # ── Ingest ───────────────────────────────────────────────────────────────
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
        supported_surfaces=ALL_SURFACES,
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
        description="Remove specific document chunks by ID.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Destructive operation not exposed in Streamlit UI",
        },
        api_route="/delete",
    ),
    Capability(
        id="collection_clear",
        name="Clear knowledge base",
        category="collection",
        tier=Tier.ONE,
        description="Delete all documents in the current project.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Destructive operation not exposed in Streamlit UI",
        },
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
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Destructive operation not exposed in Streamlit UI",
        },
        api_route="/project/delete/{name}",
    ),
    # ── Config ───────────────────────────────────────────────────────────────
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
        supported_surfaces=frozenset({Surface.API, Surface.VSCODE}),
        intentional_exceptions={
            Surface.REPL: "REPL always launches with an initialised store — startup guarantees it",
            Surface.CLI: "CLI always launches with an initialised store — startup guarantees it",
            Surface.WEBAPP: "Streamlit always runs against an initialised store",
        },
        api_route="/store/status",
    ),
    Capability(
        id="store_init",
        name="Init store",
        category="store",
        tier=Tier.ONE,
        description="Move the AxonStore to a different base path (e.g. a shared drive).",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Administrative store migration not exposed in Streamlit UI",
        },
        api_route="/store/init",
    ),
    Capability(
        id="share_generate",
        name="Generate share",
        category="share",
        tier=Tier.ONE,
        description="Create an HMAC share token for a grantee.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Multi-user share management not exposed in Streamlit UI",
        },
        api_route="/share/generate",
    ),
    Capability(
        id="share_redeem",
        name="Redeem share",
        category="share",
        tier=Tier.ONE,
        description="Mount a shared project using a share token.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Multi-user share management not exposed in Streamlit UI",
        },
        api_route="/share/redeem",
    ),
    Capability(
        id="share_revoke",
        name="Revoke share",
        category="share",
        tier=Tier.ONE,
        description="Revoke an active share grant.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Multi-user share management not exposed in Streamlit UI",
        },
        api_route="/share/revoke",
    ),
    Capability(
        id="share_list",
        name="List shares",
        category="share",
        tier=Tier.ONE,
        description="List active shares granted and received.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Multi-user share management not exposed in Streamlit UI",
        },
        api_route="/share/list",
    ),
    # ── Graph ────────────────────────────────────────────────────────────────
    Capability(
        id="graph_status",
        name="Graph status",
        category="graph",
        tier=Tier.ONE,
        description="Show entity count, code node count, and community build state.",
        supported_surfaces=ALL_SURFACES,
        api_route="/graph/status",
    ),
    Capability(
        id="graph_finalize",
        name="Graph finalize",
        category="graph",
        tier=Tier.TWO,
        description="Rebuild community summaries and finalize the knowledge graph.",
        supported_surfaces=PRIMARY_SURFACES,
        intentional_exceptions={
            Surface.WEBAPP: "Long-running rebuild operation not exposed in Streamlit UI",
        },
        api_route="/graph/finalize",
    ),
    Capability(
        id="graph_data",
        name="Graph data export",
        category="graph",
        tier=Tier.TWO,
        description="Return the full entity/relation knowledge-graph as a JSON nodes+links payload.",
        supported_surfaces=frozenset({Surface.API, Surface.VSCODE}),
        intentional_exceptions={
            Surface.REPL: "Raw JSON payload not useful in interactive REPL; use graph_viz instead",
            Surface.CLI: "Raw JSON payload not useful in CLI; use graph_viz instead",
            Surface.WEBAPP: "Streamlit uses graph_viz (HTML) not the raw data payload",
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
            Surface.WEBAPP: "Streamlit embeds the graph panel inline rather than exporting HTML",
        },
        api_route="/graph/viz",
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
            Surface.WEBAPP: "Session history is a REPL/CLI concept; Streamlit manages its own state",
        },
        api_route="/sessions",
    ),
    # ── Governance ────────────────────────────────────────────────────────────
    Capability(
        id="active_leases",
        name="Active leases",
        category="governance",
        tier=Tier.TWO,
        description="List active write-lease counts per project; used to confirm it is safe to enter maintenance state.",
        supported_surfaces=frozenset({Surface.API, Surface.VSCODE}),
        intentional_exceptions={
            Surface.REPL: "Governance operator tool — not meaningful in single-session interactive use",
            Surface.CLI: "Governance operator tool — not meaningful in single-session interactive use",
            Surface.WEBAPP: "Operator tool not exposed in Streamlit UI",
        },
        api_route="/leases",
    ),
    # ── Maintenance ───────────────────────────────────────────────────────────
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
        },
        api_route="/project/maintenance",
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
