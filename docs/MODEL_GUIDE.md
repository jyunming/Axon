# Lightweight Open-Source Models for Local RAG

Research report on embedding models and LLMs suitable for local deployment without heavy compute requirements.

## Executive Summary

For a lightweight local RAG system, we recommend:

**Embeddings:** `all-MiniLM-L6-v2` (22M params, 384-dim) or `BAAI/bge-small-en-v1.5` (FastEmbed)

**LLM:** `Llama 3.1 8B Instruct` (default; best balance of quality and speed) or `Phi-3 Mini` (minimal footprint)

Total memory footprint: 2–8 GB RAM/VRAM depending on model selection.

---

## Embedding Models

### 🏆 Recommended: all-MiniLM-L6-v2

| Metric | Value |
|--------|-------|
| **Parameters** | 22M |
| **Dimensions** | 384 |
| **Model Size** | ~80 MB |
| **Speed (CPU)** | 5,000-14,000 sentences/sec |
| **Speed vs MPNet** | 4-5x faster |
| **Quality** | Good for most RAG applications |

**Pros:**

- Extremely fast on CPU

- Small memory footprint

- Good balance of speed vs quality

- Works well for short texts (128-256 tokens)

- Easy deployment via sentence-transformers

**Cons:**

- Truncates longer inputs

- Not state-of-the-art accuracy (but good enough)

**Installation:**

```bash

pip install sentence-transformers

```

**Usage:**

```python

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts)

```

### Alternative: BAAI/bge-small-en-v1.5 (via FastEmbed)

| Metric | Value |
|--------|-------|
| **Parameters** | ~33M |
| **Dimensions** | 384 |
| **Model Size** | ~130 MB |
| **Speed** | Fast (optimized for batch processing) |
| **Quality** | Better than MiniLM on benchmarks |

**Pros:**

- Better retrieval accuracy than MiniLM

- Optimized batch processing

- Quantized versions available

**Cons:**

- Slightly larger than MiniLM

- FastEmbed dependency

**Installation:**

```bash

pip install fastembed

```

### Alternative: nomic-embed-text (via Ollama)

| Metric | Value |
|--------|-------|
| **Parameters** | 137M |
| **Dimensions** | 768 |
| **Model Size** | ~550 MB |
| **Context Length** | 8192 tokens |
| **Quality** | Excellent, trained specifically for retrieval |

**Pros:**

- Excellent retrieval performance

- Long context support (8192 tokens)

- Trained specifically for semantic search

**Cons:**

- Larger model (550MB)

- Slower than MiniLM

- Requires Ollama

**Installation:**

```bash

ollama pull nomic-embed-text

```

### Comparison Table

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 80MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Fast prototyping, high throughput |
| BGE-small | 130MB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Better accuracy, still fast |
| nomic-embed-text | 550MB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Production RAG, long documents |
| all-mpnet-base-v2 | 420MB | ⭐⭐ | ⭐⭐⭐⭐⭐ | Maximum accuracy (slower) |
| **BAAI/bge-m3** | ~1.1GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Multilingual, 8192-token context, SOTA MTEB |

**BGE-M3** (via FastEmbed) is the highest-quality supported embedding. Use it for multilingual corpora

or documents with long passages. Requires `pip install 'axon[fastembed]'`:

```yaml

embedding:

  provider: fastembed

  model: BAAI/bge-m3

```

---

## Large Language Models (LLMs)

> **Default model:** `llama3.1:8b` (via Ollama). Change with `llm.model` in `config.yaml` or live from the UI sidebar.

### Cloud LLM Providers

The system supports cloud providers in addition to local Ollama models. Set `llm.provider` in `config.yaml`:

| Provider | `llm.provider` value | Example models | Notes |
|---|---|---|---|
| **Ollama (local)** | `ollama` | `gemma`, `llama3.1:8b`, `phi3:mini` | Default; fully offline |
| **Google Gemini** | `gemini` | `gemini-2.0-flash`, `gemini-1.5-pro` | Requires `GEMINI_API_KEY` |
| **OpenAI** | `openai` | `gpt-4o`, `gpt-4.1`, `o3-mini` | Requires `OPENAI_API_KEY` |
| **xAI Grok** | `grok` | `grok-3`, `grok-3-fast`, `grok-2` | Requires `XAI_API_KEY`; OpenAI-compatible endpoint (`api.x.ai/v1`) |
| **Ollama Cloud** | `ollama_cloud` | Any Ollama-hosted model | Requires `OLLAMA_CLOUD_URL` + `OLLAMA_CLOUD_KEY` |
| **vLLM** | `vllm` | Any vLLM-served model | Self-hosted OpenAI-compatible endpoint; set `vllm_base_url` in `config.yaml` |
| **GitHub Copilot** | `copilot` | Any active Copilot model | VS Code only — routes LLM calls through the Copilot extension bridge; no Ollama required. Enable via `axon.useCopilotLlm: true` in VS Code settings or `provider: copilot` in `config.yaml`. |

### Per-provider config.yaml examples

**Ollama (local, default — no API key needed):**

```yaml

llm:

  provider: ollama

  model: llama3.1:8b

```

**OpenAI:**

```yaml

llm:

  provider: openai

  model: gpt-4o          # or gpt-4.1, o3-mini, gpt-3.5-turbo

  openai_api_key: sk-... # or export OPENAI_API_KEY=sk-...

```

**xAI Grok:**

```yaml

llm:

  provider: grok

  model: grok-3          # or grok-3-fast, grok-2

  grok_api_key: xai-...  # or export XAI_API_KEY=xai-...

```

**Google Gemini:**

```yaml

llm:

  provider: gemini

  model: gemini-2.0-flash  # or gemini-1.5-pro

  gemini_api_key: AIza...  # or export GEMINI_API_KEY=AIza...

```

**vLLM (self-hosted):**

```yaml

llm:

  provider: vllm

  model: mistral-7b-instruct

  vllm_base_url: http://localhost:8000/v1

```

**GitHub Copilot (VS Code bridge):**

```yaml

llm:

  provider: copilot        # alias: "github_copilot" also accepted

  model: gpt-4o            # any model available in your Copilot subscription

```

> **`copilot` vs `github_copilot`:** These are two distinct providers.
> - `copilot` — routes LLM calls through the Axon VS Code extension bridge. Requires VS Code running with the extension active. No API key needed.
> - `github_copilot` — calls the Copilot API directly via a PAT/OAuth token. Works from CLI, API server, or MCP without VS Code.

**GitHub Copilot API (PAT-based, no VS Code required):**

```yaml

llm:

  provider: github_copilot

  model: gpt-4o

  # export GITHUB_COPILOT_PAT=<oauth-token>  (use device flow: axon /keys set github_copilot)

```

**Ollama Cloud (remote Ollama-compatible endpoint):**

```yaml

llm:

  provider: ollama_cloud

  model: llama3.1:8b          # any model available on your hosted endpoint

  # export OLLAMA_CLOUD_URL=https://your-ollama-host

  # export OLLAMA_CLOUD_KEY=your-api-key

```

Use `ollama_cloud` when your Ollama instance is running on a remote server rather than localhost.

The provider is identical to `ollama` except it reads `OLLAMA_CLOUD_URL` and `OLLAMA_CLOUD_KEY`

instead of the default `OLLAMA_HOST`. Both env vars are required when using this provider.

> **Note:** Gemma models (Google) do not support `system_instruction` in the Gemini SDK. The system automatically prepends the system prompt to the user message for Gemma model names.

### Environment variables quick reference

| Provider | Environment variable | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | Preferred over `API_KEY` (legacy) |
| xAI Grok | `XAI_API_KEY` | Also accepted: `GROK_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` | — |
| GitHub Copilot | `GITHUB_COPILOT_PAT` | OAuth token; also accepted: `GITHUB_TOKEN` |
| Ollama Cloud | `OLLAMA_CLOUD_URL` + `OLLAMA_CLOUD_KEY` | — |
| Brave Search | `BRAVE_API_KEY` | For `web_search.enabled: true` |

---

### 🏆 Recommended Local: Llama 3.1 8B Instruct

| Metric | Value |
|--------|-------|
| **Parameters** | 8B |
| **Model Size** | ~4.7 GB (Q4_K_M quantized) |
| **Context Length** | 128K tokens |
| **Memory Required** | ~6-8 GB RAM/VRAM |
| **Speed (RTX 4090)** | ~50-80 tokens/sec |
| **Speed (CPU)** | ~5-10 tokens/sec |

**Pros:**

- Excellent instruction following

- Large context window (128K)

- Strong benchmark performance

- Widely supported ecosystem

- Good for RAG with long contexts

**Cons:**

- Needs 6-8GB RAM minimum

- Slower than smaller models

**Ollama Installation:**

```bash

ollama pull llama3.1:8b

```

### Alternative: Qwen2.5-7B-Instruct

| Metric | Value |
|--------|-------|
| **Parameters** | 7B |
| **Model Size** | ~4.5 GB (Q4_K_M quantized) |
| **Context Length** | 128K tokens |
| **Memory Required** | ~6-8 GB RAM/VRAM |

**Pros:**

- Excellent multilingual support

- Strong coding capabilities

- Good for technical documents

- Comparable to Llama 3.1

**Cons:**

- Slightly smaller ecosystem than Llama

**Ollama Installation:**

```bash

ollama pull qwen2.5:7b

```

### Alternative: Phi-3 Mini (3.8B)

| Metric | Value |
|--------|-------|
| **Parameters** | 3.8B |
| **Model Size** | ~2.3 GB (Q4_K_M quantized) |
| **Context Length** | 128K tokens |
| **Memory Required** | ~4-6 GB RAM/VRAM |

**Pros:**

- Smallest viable model (only 3.8B params)

- State-of-the-art for its size

- Fast inference

- Good for resource-constrained environments

**Cons:**

- Less capable than 7B/8B models

- May struggle with complex reasoning

**Ollama Installation:**

```bash

ollama pull phi3:mini

```

### Alternative: Mistral 7B Instruct

| Metric | Value |
|--------|-------|
| **Parameters** | 7B |
| **Model Size** | ~4.1 GB (Q4_K_M quantized) |
| **Context Length** | 32K tokens |
| **Memory Required** | ~6-8 GB RAM/VRAM |

**Pros:**

- Fast inference

- Good balance of speed and quality

- Sliding window attention (efficient)

**Cons:**

- Smaller context than Llama 3.1

- Older architecture

**Ollama Installation:**

```bash

ollama pull mistral:7b

```

### Comparison Table

| Model | Size | Speed | Quality | Context | Best For |
|-------|------|-------|---------|---------|----------|
| Phi-3 Mini | 2.3GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 128K | Resource-constrained, fast responses |
| Mistral 7B | 4.1GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 32K | Balanced speed/quality |
| Qwen2.5 7B | 4.5GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | Technical docs, multilingual |
| Llama 3.1 8B | 4.7GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | Best overall, long contexts |

---

## Hardware Requirements

Axon's background thread pool uses up to `max_workers` CPU cores at once (default: **8**).

Set `max_workers` in `config.yaml` to match your hardware — see

[Concurrency & Performance Tuning](ADMIN_REFERENCE.md#610-concurrency--performance-tuning) for guidance.

### Minimum Configuration

- **CPU:** 8+ cores (modern x86_64)

- **RAM:** 16 GB

- **Storage:** 10 GB free

- **OS:** Linux/macOS/Windows with WSL

**Recommended Setup:**

- Embedding: all-MiniLM-L6-v2 (CPU)

- LLM: Phi-3 Mini (CPU) or Llama 3.1 8B (if GPU available)

- `max_workers: 4`

### Recommended Configuration

- **CPU:** 16+ cores

- **RAM:** 32 GB

- **GPU:** RTX 3060 12GB or better

- **Storage:** 20 GB free (for multiple models)

**Recommended Setup:**

- Embedding: nomic-embed-text or BGE-small

- LLM: Llama 3.1 8B (GPU) or Qwen2.5 7B

- `max_workers: 12`

### Optimal Configuration

- **CPU:** 32+ cores

- **RAM:** 64 GB

- **GPU:** RTX 4090 24GB

- **Storage:** 50 GB free

**Optimal Setup:**

- Embedding: nomic-embed-text

- LLM: Llama 3.1 8B (GPU-accelerated)

- Vector Store: Qdrant (in-memory)

- `max_workers: 24`, `graph_rag_map_workers: 8`

---

## Performance Benchmarks

### Embedding Speed (1,000 documents)

| Model | CPU (ms) | GPU (ms) |
|-------|----------|----------|
| all-MiniLM-L6-v2 | 100 | 20 |
| BGE-small | 150 | 30 |
| nomic-embed-text | 500 | 100 |

### LLM Inference Speed (tokens/sec)
| Model | CPU (16 cores) | RTX 3060 | RTX 4090 |
|-------|----------------|----------|----------|
| Phi-3 Mini | 15 | 40 | 100 |
| Mistral 7B | 8 | 30 | 70 |
| Llama 3.1 8B | 5 | 25 | 60 |

---

## Recommendations for Local RAG

### Option 1: Maximum Speed (Low Resource)

```yaml

embedding:

  provider: sentence_transformers

  model: all-MiniLM-L6-v2

llm:

  provider: ollama

  model: phi3:mini

vector_store:

  provider: lancedb

```

**Total Memory:** ~3-4 GB

**Best For:** Development, testing, low-traffic deployments

### Option 2: Balanced (Recommended)

```yaml

embedding:

  provider: sentence_transformers

  model: all-MiniLM-L6-v2

llm:

  provider: ollama

  model: llama3.1:8b

vector_store:

  provider: lancedb

```

**Total Memory:** ~6-8 GB

**Best For:** Production use, good balance of speed and quality

### Option 3: Maximum Quality

```yaml

embedding:

  provider: ollama

  model: nomic-embed-text

llm:

  provider: ollama

  model: qwen2.5:7b

vector_store:

  provider: qdrant

```

**Total Memory:** ~10-12 GB

**Best For:** Production with GPU, maximum accuracy

---

## References

1. **Sentence Transformers Documentation:** https://www.sbert.net/

2. **Ollama Library:** https://ollama.com/library

3. **FastEmbed GitHub:** https://github.com/qdrant/fastembed

4. **Llama 3.1 Model Card:** https://github.com/meta-llama/llama-models

5. **MTEB Leaderboard:** https://huggingface.co/spaces/mteb/leaderboard

---

*Research compiled: February 26, 2026*

*Recommended configuration updated for local deployment*

