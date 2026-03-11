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
