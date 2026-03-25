# Lightweight Open-Source Models for Local RAG

Research report on embedding models and LLMs suitable for local deployment without heavy compute requirements.

## Executive Summary

For a lightweight local RAG system, we recommend:

**Embeddings:** `all-MiniLM-L6-v2` (22M params, 384-dim) or `BAAI/bge-small-en-v1.5` (FastEmbed)
**LLM:** `Gemma 2B` (default; smallest performant model) or `Llama 3.1 8B Instruct` or `Qwen2.5-7B-Instruct` for better quality

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
| **OpenAI** | `openai` | `gpt-4o`, `gpt-3.5-turbo` | Requires `OPENAI_API_KEY` |
| **Ollama Cloud** | `ollama_cloud` | Any Ollama-hosted model | Requires `OLLAMA_CLOUD_URL` + `OLLAMA_CLOUD_KEY` |
| **vLLM** | `vllm` | Any vLLM-served model | Self-hosted OpenAI-compatible endpoint; set `vllm_base_url` in `config.yaml` |
| **GitHub Copilot** | `copilot` | Any active Copilot model | VS Code only — routes LLM calls through the Copilot extension bridge; no Ollama required. Enable via `axon.useCopilotLlm: true` in VS Code settings or `provider: copilot` in `config.yaml`. |

Example `config.yaml` for Gemini:
```yaml
llm:
  provider: gemini
  model: gemini-2.0-flash
```

> **Note:** Gemma models (Google) do not support `system_instruction` in the Gemini SDK. The system automatically prepends the system prompt to the user message for Gemma model names.

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

### Minimum Configuration
- **CPU:** 8+ cores (modern x86_64)
- **RAM:** 16 GB
- **Storage:** 10 GB free
- **OS:** Linux/macOS/Windows with WSL

**Recommended Setup:**
- Embedding: all-MiniLM-L6-v2 (CPU)
- LLM: Phi-3 Mini (CPU) or Llama 3.1 8B (if GPU available)

### Recommended Configuration
- **CPU:** 16+ cores
- **RAM:** 32 GB
- **GPU:** RTX 3060 12GB or better
- **Storage:** 20 GB free (for multiple models)

**Recommended Setup:**
- Embedding: nomic-embed-text or BGE-small
- LLM: Llama 3.1 8B (GPU) or Qwen2.5 7B

### Optimal Configuration
- **CPU:** 32+ cores
- **RAM:** 64 GB
- **GPU:** RTX 4090 24GB
- **Storage:** 50 GB free

**Optimal Setup:**
- Embedding: nomic-embed-text
- LLM: Llama 3.1 8B (GPU-accelerated)
- Vector Store: Qdrant (in-memory)

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
  provider: chroma
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
  provider: chroma
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

## Migration Guide

### From Legacy RAG Systems

1. **Export existing documents:**
```bash
python migrate.py export --output rag_export.json
```

2. **Setup Axon with lightweight models:**
```bash
# Install dependencies
pip install -r requirements.txt

# Pull models
ollama pull llama3.1:8b
ollama pull nomic-embed-text  # Optional, for better quality
```

3. **Import and test:**
```python
from axon.main import AxonBrain, AxonConfig

config = AxonConfig(
    embedding_provider="sentence_transformers",
    embedding_model="all-MiniLM-L6-v2",
    llm_provider="ollama",
    llm_model="llama3.1:8b"
)

brain = AxonBrain(config)
response = brain.query("What are the key themes in my documents?")
```

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
