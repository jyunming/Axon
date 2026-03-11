"""
Multi-step Agent Orchestration Example.

Demonstrates a planner-critic loop over the Axon API:
  1. Check health
  2. Search for relevant chunks (/search)
  3. Evaluate relevance locally
  4. Synthesize answer (/query) or inject new knowledge (/add_text) and retry
  5. Print final answer + tool schema reference

No external LLM required — routing is heuristic-based for portability.

Usage:
    python examples/agent_orchestration.py
"""

import logging
import sys
from typing import Any

import httpx
from axon.tools import get_rag_tool_definition

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"

# ──────────────────────────────────────────────
# Relevance threshold used in Phase 2 (Reason).
# An LLM-based orchestrator would use the model to judge chunk quality;
# here we use a simple numeric score cutoff as a portable substitute.
# ──────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.30
# Minimum number of chunks that must exceed the threshold to consider the
# knowledge base "sufficient" for answering without injecting new facts.
MIN_RELEVANT_CHUNKS = 1


# ──────────────────────────────────────────────────────────────────────────────
# Phase 0 — Health check
# ──────────────────────────────────────────────────────────────────────────────


def check_health(client: httpx.Client) -> bool:
    """
    Call GET /health to verify the API is reachable before starting the loop.

    Returns True if the service is healthy, False otherwise.
    """
    try:
        resp = client.get(f"{API_URL}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        print(f"✅  Health check passed — status: {status}")
        return True
    except httpx.ConnectError:
        print(
            "❌  Cannot connect to the Axon API at "
            f"{API_URL}.\n"
            "    Start the server with:  uvicorn axon.api:app --reload"
        )
        return False
    except httpx.HTTPStatusError as exc:
        print(f"❌  Health check returned HTTP {exc.response.status_code}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Retrieve
# ──────────────────────────────────────────────────────────────────────────────


def retrieve_chunks(
    client: httpx.Client,
    question: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Phase 1 — POST /search to get raw document chunks.

    An LLM orchestrator would embed the question, retrieve chunks, and then
    feed them verbatim into its context window for reasoning.  Here we return
    the raw list so our local heuristic can inspect it in Phase 2.
    """
    print(f"\n🔍  Phase 1 — Retrieve: searching for '{question}' (top_k={top_k})")
    resp = client.post(
        f"{API_URL}/search",
        json={"query": question, "top_k": top_k},
        timeout=30.0,
    )
    resp.raise_for_status()
    chunks: list[dict[str, Any]] = resp.json()
    print(f"    → Retrieved {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        score = chunk.get("score", 0.0)
        preview = chunk.get("text", "")[:80].replace("\n", " ")
        print(f'      [{i + 1}] score={score:.3f}  "{preview}..."')
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Reason (local heuristic)
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_chunks(chunks: list[dict[str, Any]]) -> bool:
    """
    Phase 2 — Inspect retrieved chunks and decide whether they are sufficient.

    Decision point:
      • An LLM orchestrator would ask its model: "Given these chunks, can you
        answer the question confidently?"  The model would reply YES/NO with
        an explanation (chain-of-thought or structured output).
      • Here we use a heuristic: count how many chunks have a relevance score
        above RELEVANCE_THRESHOLD.  If enough chunks qualify we proceed to
        synthesis; otherwise we inject new knowledge first.

    Returns True  → knowledge is sufficient, go to Phase 3a (synthesize).
    Returns False → knowledge is insufficient, go to Phase 3b (learn + retry).
    """
    print("\n🤔  Phase 2 — Reason: evaluating chunk relevance locally")
    relevant = [c for c in chunks if c.get("score", 0.0) >= RELEVANCE_THRESHOLD]
    print(
        f"    → {len(relevant)}/{len(chunks)} chunk(s) exceed "
        f"relevance threshold ({RELEVANCE_THRESHOLD})"
    )

    sufficient = len(relevant) >= MIN_RELEVANT_CHUNKS
    verdict = (
        "SUFFICIENT — will synthesize" if sufficient else "INSUFFICIENT — will inject knowledge"
    )
    print(f"    → Verdict: {verdict}")
    return sufficient


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3a — Synthesize
# ──────────────────────────────────────────────────────────────────────────────


def synthesize_answer(
    client: httpx.Client,
    question: str,
) -> str | None:
    """
    Phase 3a — POST /query to get a synthesized, LLM-generated answer.

    An LLM orchestrator would call this tool when it has enough context and
    wants a fluent, grounded response rather than raw chunks.
    """
    print(f"\n💡  Phase 3a — Synthesize: calling /query for '{question}'")
    resp = client.post(
        f"{API_URL}/query",
        json={"query": question},
        timeout=60.0,
    )
    resp.raise_for_status()
    answer: str = resp.json().get("response", "")
    return answer


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3b — Learn (inject new fact) then retry
# ──────────────────────────────────────────────────────────────────────────────


def inject_knowledge(
    client: httpx.Client,
    fact: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Phase 3b — POST /add_text to teach the knowledge base a new fact.

    An LLM orchestrator would generate this fact by reasoning about what
    information is missing, then inject it so that a subsequent /query call
    can draw on the new content.  This closes the "learn" arc of the loop.
    """
    print("\n📚  Phase 3b — Learn: injecting new fact into knowledge base")
    print(f'    Fact: "{fact[:100]}..."' if len(fact) > 100 else f'    Fact: "{fact}"')
    payload: dict[str, Any] = {"text": fact}
    if metadata:
        payload["metadata"] = metadata
    resp = client.post(f"{API_URL}/add_text", json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    doc_id = data.get("id", "unknown")
    print(f"    → Stored with doc_id={doc_id}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Tool-schema helper
# ──────────────────────────────────────────────────────────────────────────────


def show_tool_schemas() -> None:
    """
    Print the tool definitions that an LLM orchestrator would receive at
    system-prompt time.  These follow the OpenAI/Anthropic tool-call schema.

    An LLM uses these schemas to decide *which* function to call and what
    arguments to supply — this is the backbone of function-calling agents.
    """
    print("\n📋  Tool schemas (from get_rag_tool_definition):")
    tools = get_rag_tool_definition(API_URL)
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = list(fn.get("parameters", {}).get("properties", {}).keys())
        print(f"   • {name}({', '.join(params)}) — {desc}")


# ──────────────────────────────────────────────────────────────────────────────
# Routing helper — mimics LLM tool selection
# ──────────────────────────────────────────────────────────────────────────────


def select_tool(question: str) -> str:
    """
    Simple keyword-based routing that simulates what an LLM tool-selector
    would do when given the available tool schemas.

    An LLM orchestrator reads the tool descriptions and picks the function
    whose description best matches the current task.  Our heuristic mirrors
    this by checking for imperative keywords that imply a write vs. read intent.

    Returns one of: 'search', 'query', 'add_text'.
    """
    lowered = question.lower()
    if any(kw in lowered for kw in ("remember", "store", "save", "learn", "add")):
        return "add_text"
    if any(kw in lowered for kw in ("detail", "raw", "chunk", "find", "list")):
        return "search"
    # Default: synthesized answer via /query
    return "query"


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration loop
# ──────────────────────────────────────────────────────────────────────────────


def orchestrate(
    question: str,
    fallback_fact: str | None = None,
) -> None:
    """
    Run the 3-phase planner-critic loop for a single question.

    Args:
        question:     The natural-language question to answer.
        fallback_fact: A fact to inject when the knowledge base lacks
                       sufficient context.  In a real agent this would be
                       generated by the LLM itself (e.g. from a web search).
    """
    print("\n" + "=" * 60)
    print(f'Orchestration loop -- question: "{question}"')
    print("=" * 60)

    with httpx.Client() as client:
        # ── Phase 0: Health ───────────────────────────────────────────────
        if not check_health(client):
            sys.exit(1)

        # ── Show tool schemas (what an LLM agent would see) ───────────────
        show_tool_schemas()

        # ── Routing decision ──────────────────────────────────────────────
        chosen_tool = select_tool(question)
        print(f"\n🗺️   Routing heuristic selected tool: '{chosen_tool}'")
        print(
            "    (An LLM orchestrator would read the tool schemas above and "
            "use its own reasoning to make this selection.)"
        )

        # ── Phase 1: Retrieve ─────────────────────────────────────────────
        chunks = retrieve_chunks(client, question, top_k=5)

        # ── Phase 2: Reason ───────────────────────────────────────────────
        knowledge_ok = evaluate_chunks(chunks)

        if not knowledge_ok:
            # ── Phase 3b: Learn ───────────────────────────────────────────
            fact = fallback_fact or (
                f"The Axon is a fully local, open-source RAG system "
                f"built with FastAPI, Streamlit, Ollama, and ChromaDB/Qdrant.  "
                f"It answers the question: {question}"
            )
            inject_knowledge(
                client,
                fact=fact,
                metadata={"source": "agent_orchestration_example", "topic": "axon"},
            )

            # Retry retrieval after injection
            print("\n🔄  Retrying retrieval after knowledge injection…")
            chunks = retrieve_chunks(client, question, top_k=5)
            # Re-evaluate (for demonstration; in production you might loop N times)
            knowledge_ok = evaluate_chunks(chunks)

        # ── Phase 3a: Synthesize ──────────────────────────────────────────
        answer = synthesize_answer(client, question)

        print("\n" + "─" * 60)
        print("🏁  Final answer:")
        print(f"    {answer}")
        print("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    demo_question = "What makes the Axon different from cloud-based RAG systems?"

    # Fallback fact that would normally be generated by the LLM or fetched
    # from an external source when the knowledge base lacks context.
    demo_fallback_fact = (
        "Axon runs entirely on-premises: all embeddings, vector "
        "storage, and LLM inference happen locally via Ollama.  No data ever "
        "leaves the user's machine, making it privacy-safe and usable offline. "
        "Cloud RAG systems (e.g. OpenAI Assistants, Vertex AI Search) require "
        "an internet connection and send documents to third-party servers."
    )

    orchestrate(question=demo_question, fallback_fact=demo_fallback_fact)
