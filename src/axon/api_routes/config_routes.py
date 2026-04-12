"""Config validation + wizard REST routes."""


from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("AxonAPI")


router = APIRouter()


# ---------------------------------------------------------------------------


# Request models


# ---------------------------------------------------------------------------


_DOT_TO_FLAT: dict[str, str] = {
    # LLM
    "llm.provider": "llm_provider",
    "llm.model": "llm_model",
    "llm.base_url": "ollama_base_url",
    "llm.temperature": "llm_temperature",
    "llm.max_tokens": "llm_max_tokens",
    "llm.timeout": "llm_timeout",
    "llm.vllm_base_url": "vllm_base_url",
    "llm.openai_api_key": "openai_api_key",
    "llm.grok_api_key": "grok_api_key",
    "llm.gemini_api_key": "gemini_api_key",
    "llm.copilot_pat": "copilot_pat",
    # Embedding
    "embedding.provider": "embedding_provider",
    "embedding.model": "embedding_model",
    "embedding.model_path": "embedding_model_path",
    # Vector store
    "vector_store.provider": "vector_store",
    "vector_store.path": "vector_store_path",
    "qdrant.url": "qdrant_url",
    "qdrant.api_key": "qdrant_api_key",
    # Chunking
    "chunk.strategy": "chunk_strategy",
    "chunk.size": "chunk_size",
    "chunk.overlap": "chunk_overlap",
    "chunk.cosine_semantic_threshold": "cosine_semantic_threshold",
    "chunk.cosine_semantic_max_size": "cosine_semantic_max_size",
    # Retrieval
    "rag.top_k": "top_k",
    "rag.similarity_threshold": "similarity_threshold",
    "rag.hybrid_search": "hybrid_search",
    "rag.hybrid_weight": "hybrid_weight",
    "rag.hybrid_mode": "hybrid_mode",
    "rag.mmr": "mmr",
    "rag.mmr_lambda": "mmr_lambda",
    "rag.parent_chunk_size": "parent_chunk_size",
    # Sentence window & CRAG-Lite
    "rag.sentence_window": "sentence_window",
    "rag.sentence_window_size": "sentence_window_size",
    "rag.crag_lite": "crag_lite",
    "rag.crag_lite_confidence_threshold": "crag_lite_confidence_threshold",
    # Reranking
    "rerank.enabled": "rerank",
    "rerank.provider": "reranker_provider",
    "rerank.model": "reranker_model",
    # Query transformations
    "query_transformations.multi_query": "multi_query",
    "query_transformations.hyde": "hyde",
    "query_transformations.step_back": "step_back",
    "query_transformations.query_decompose": "query_decompose",
    "query_transformations.discussion_fallback": "discussion_fallback",
    "context_compression.enabled": "compress_context",
    "context_compression.strategy": "compression_strategy",
    "rag.query_router": "query_router",
    "rag.contextual_retrieval": "contextual_retrieval",
    # RAPTOR
    "rag.raptor": "raptor",
    "rag.raptor_min_source_size_mb": "raptor_min_source_size_mb",
    "rag.raptor_chunk_group_size": "raptor_chunk_group_size",
    "rag.raptor_max_levels": "raptor_max_levels",
    "rag.raptor_cache_summaries": "raptor_cache_summaries",
    "rag.raptor_retrieval_mode": "raptor_retrieval_mode",
    "rag.raptor_drilldown": "raptor_drilldown",
    "rag.raptor_drilldown_top_k": "raptor_drilldown_top_k",
    # GraphRAG
    "rag.graph_rag": "graph_rag",
    "rag.graph_rag_mode": "graph_rag_mode",
    "rag.graph_rag_depth": "graph_rag_depth",
    "rag.graph_rag_budget": "graph_rag_budget",
    "rag.graph_rag_relations": "graph_rag_relations",
    "rag.graph_rag_community": "graph_rag_community",
    "rag.graph_rag_community_async": "graph_rag_community_async",
    "rag.graph_rag_ner_backend": "graph_rag_ner_backend",
    "rag.graph_rag_entity_min_frequency": "graph_rag_entity_min_frequency",
    "rag.graph_rag_auto_route": "graph_rag_auto_route",
    "rag.graph_rag_canonicalize": "graph_rag_canonicalize",
    # Code graph
    "rag.code_graph": "code_graph",
    "rag.code_graph_bridge": "code_graph_bridge",
    "rag.code_lexical_boost": "code_lexical_boost",
    "rag.code_top_k": "code_top_k",
    "rag.code_top_k_multiplier": "code_top_k_multiplier",
    # Output
    "rag.cite": "cite",
    "web_search.enabled": "truth_grounding",
    "web_search.brave_api_key": "brave_api_key",
    # Performance
    "rag.query_cache": "query_cache",
    "rag.query_cache_size": "query_cache_size",
    "rag.dedup_on_ingest": "dedup_on_ingest",
    "rag.smart_ingest": "smart_ingest",
    "rag.ingest_batch_mode": "ingest_batch_mode",
    "rag.max_chunks_per_source": "max_chunks_per_source",
    "max_workers": "max_workers",
    # REPL
    "repl.shell_passthrough": "repl_shell_passthrough",
    # Offline
    "offline.enabled": "offline_mode",
    "offline.local_assets_only": "local_assets_only",
    "offline.local_models_dir": "local_models_dir",
    "offline.embedding_models_dir": "embedding_models_dir",
    "offline.hf_models_dir": "hf_models_dir",
    "offline.tokenizer_cache_dir": "tokenizer_cache_dir",
    # Store & paths
    "store.base": "axon_store_base",
    "projects_root": "projects_root",
}


class ConfigSetRequest(BaseModel):
    key: str  # dot-notation: "chunk.strategy", "llm.model"

    value: Any

    persist: bool = True


def _reinitialize_runtime_components(brain, changed_keys: set[str]) -> None:
    """Apply runtime side effects for config changes that affect live services."""

    if changed_keys & {"llm_provider", "llm_model"}:
        from axon.llm import OpenLLM

        brain.llm = OpenLLM(brain.config)

    if changed_keys & {"embedding_provider", "embedding_model"}:
        from axon.embeddings import OpenEmbedding

        brain.embedding = OpenEmbedding(brain.config)

    if changed_keys & {"rerank", "reranker_model"} and getattr(brain.config, "rerank", False):
        from axon.rerank import OpenReranker

        brain.reranker = OpenReranker(brain.config)


# ---------------------------------------------------------------------------


# Routes


# ---------------------------------------------------------------------------


@router.get("/config/validate")
async def validate_config():
    """Validate the current config.yaml and return a structured list of issues."""

    from axon import api as _api
    from axon.config import AxonConfig

    brain = _api.brain

    # Determine path: use brain's loaded path when available

    config_path: str | None = None

    if brain is not None and brain.config is not None:
        config_path = getattr(brain.config, "_loaded_path", None)

    issues = AxonConfig.validate(config_path)

    issue_dicts = [i.to_dict() for i in issues]

    has_errors = any(i.level == "error" for i in issues)

    return {
        "valid": not has_errors,
        "issue_count": len(issues),
        "issues": issue_dicts,
    }


@router.post("/config/reset")
async def reset_config():
    """Reset config.yaml to built-in defaults.


    Writes the default config template to the user config path and returns the


    path that was written.  The running brain is NOT reloaded — restart or call


    POST /config/update to apply the new values.


    """

    import os
    from pathlib import Path

    from axon.config import _DEFAULT_CONFIG_YAML, _USER_CONFIG_PATH

    target = str(_USER_CONFIG_PATH)

    os.makedirs(os.path.dirname(target), exist_ok=True)

    Path(target).write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")

    logger.info("Config reset to defaults at %s", target)

    return {"written_to": target}


@router.post("/config/set")
async def set_config_field(request: ConfigSetRequest):
    """Set a single config field using dot-notation (e.g. chunk.strategy).


    The dot key is mapped to the flat dataclass attribute name via


    ``_DOT_TO_FLAT``.  Pass ``persist: true`` to also save the change to


    config.yaml on disk.


    """

    from axon import api as _api

    brain = _api.brain

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    flat_key = _DOT_TO_FLAT.get(request.key, request.key)

    if not hasattr(brain.config, flat_key):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown config key '{request.key}'. "
                f"Known dot-notation keys: {sorted(_DOT_TO_FLAT.keys())}"
            ),
        )

    old_value = getattr(brain.config, flat_key)

    setattr(brain.config, flat_key, request.value)

    _reinitialize_runtime_components(brain, {flat_key})

    if request.persist:
        brain.config.save()

    return {
        "status": "success",
        "key": request.key,
        "flat_key": flat_key,
        "old_value": old_value,
        "new_value": request.value,
        "persisted": request.persist,
    }
