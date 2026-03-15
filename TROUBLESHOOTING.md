# Troubleshooting Guide

Common issues and fixes for Axon.

---

## ChromaDB: `InvalidDimensionException`

**Error:**
```
chromadb.errors.InvalidDimensionException: Embedding dimension 384 does not match collection dimensionality 768
```

**Cause:** You switched the embedding model (e.g., from `nomic-embed-text` at 768d to `all-MiniLM-L6-v2` at 384d) but the existing ChromaDB collection was created with the old dimension.

**Fix:** Stop all containers and delete the vector store:
```bash
docker compose stop
rm -rf ./chroma_data ./bm25_index
docker compose up -d
```
> ⚠️ This deletes all indexed documents. Re-ingest your files after restarting.

---

## ChromaDB: `Could not connect to tenant default_tenant`

**Error:**
```
ValueError: Could not connect to tenant default_tenant. Are you sure it exists?
```

**Cause:** The ChromaDB data directory was partially deleted while containers were still running, corrupting the SQLite metadata.

**Fix:** Fully stop Docker, delete, and restart:
```bash
docker compose down
rm -rf ./chroma_data
docker compose up -d
```

---

## Ollama: LLM Forgets Conversation History

**Symptom:** The assistant doesn't remember what you discussed earlier in the same session, even though sessions are enabled.

**Cause:** Ollama defaults to a 2048-token context window (`num_ctx`). When RAG context documents fill this window, conversation history gets silently truncated.

**Fix:** This is now handled automatically — `num_ctx` is set to 8192 in all Ollama calls. If you still experience issues with very long conversations, you can increase it further in `main.py` (`OpenLLM` class, search for `num_ctx`).

---

## SentenceTransformers: `unexpected keyword argument 'convert_to_list'`

**Error:**
```
TypeError: SentenceTransformer.encode() got an unexpected keyword argument 'convert_to_list'
```

**Cause:** Some versions of `sentence-transformers` don't support the `convert_to_list` parameter.

**Fix:** Already patched — the code now uses `.tolist()` on the numpy array instead.

---

## Docker: File Changes Not Reflected in Web UI

**Symptom:** You edited source files but the Streamlit UI still runs old code.

**Cause:** Docker Desktop on Windows sometimes fails to propagate filesystem events through volume mounts, so Streamlit's auto-reloader doesn't trigger.

**Fix:**
```bash
docker compose restart axon-ui
```

---

## Brave Search: `truth_grounding` Enabled But No Web Results

**Symptom:** Web search toggle is on but no 🌐 sources appear.

**Cause:** Missing or invalid Brave API key.

**Fix:**
1. Get a free API key at [https://brave.com/search/api/](https://brave.com/search/api/)
2. Either set `BRAVE_API_KEY` in your `.env` file, or enter it in the Web UI sidebar under "🌐 Web Search"

---

## General: `.env` File

Create a `.env` file in the project root for API keys:
```env
GEMINI_API_KEY=your-gemini-key
BRAVE_API_KEY=your-brave-key
OLLAMA_CLOUD_KEY=your-ollama-cloud-key
OLLAMA_CLOUD_URL=https://your-endpoint
```

The `.env` file is optional — Docker Compose won't fail if it's missing.

---

## Answer Quality Variability with Advanced RAG Features

**Symptom:** Features like `multi_query`, inline citations, GraphRAG entity extraction, or RAPTOR summarisation produce inconsistent or degraded results.

**Cause:** These features rely on the LLM following structured instructions. Smaller models (e.g., `llama3.2:1b`, `gemma:2b`) may ignore instructions, produce malformed output, or refuse to answer when context is complex. This is not a bug — it reflects the capability limits of the model.

**Guidance:**
- Use a capable model (7B+ parameters) for best results with advanced RAG features.
- If `multi_query` degrades answer quality, disable it: `/rag multi` or set `multi_query: false` in `config.yaml`.
- If citations are missing or the model refuses, try a larger model or disable citation mode: `/rag cite`.
- GraphRAG entity extraction requires an LLM that can follow extraction instructions reliably. If the entity graph remains empty after ingestion, check the server logs for a zero-entity warning and consider switching to a larger model.

---

## GraphRAG: Entity Graph Empty After Ingestion

**Symptom:** GraphRAG is enabled but retrieval does not expand with entity-connected documents. Logs show: `GraphRAG: entity extraction returned 0 entities across all chunks.`

**Cause:** The LLM failed to extract any entities. Common reasons:
- Model is too small or not instruction-tuned.
- The model returned bullets or lists despite the "no bullets" instruction (now stripped automatically), but returned empty output entirely.
- LLM request timed out during entity extraction.

**Fix:** Switch to a larger or more capable model. A 7B+ instruction-tuned model (e.g., `llama3.1:8b`, `mistral:7b`) reliably extracts entities.

---

## Provider Auto-Detection for vLLM / HuggingFace-style Model Names

**Symptom:** You set `llm_model: meta-llama/Llama-3.1-8B-Instruct` but the model is served via vLLM, not Ollama.

**Cause:** Axon infers `llm_provider` from the model name when the provider is not explicitly set. HuggingFace-style names (e.g., `org/model-name`) are inferred as `ollama` by default, since Ollama also accepts many such names.

**Fix:** Explicitly set the provider in `config.yaml`:
```yaml
llm:
  provider: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
```

---

## `top_k` and Raw Retrieval Count

**Symptom:** The API or internal retrieval returns more chunks than the configured `top_k` value.

**Cause:** When `hybrid_search` or `rerank` is enabled, Axon internally fetches `top_k × 3` candidates to allow for score merging and re-ranking. The final result passed to the LLM is capped at `top_k` after all processing. Internal retrieval methods (used in debugging or qualification scripts) may show the pre-cap candidate set.

**Guidance:** `top_k` controls how many chunks the LLM receives as context. The pre-cap overfetch is intentional and improves hybrid/reranked result quality.
