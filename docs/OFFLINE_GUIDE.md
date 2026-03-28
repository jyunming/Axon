# Offline / Air-Gap Guide

> **What does "air-gapped" mean?** An air-gapped machine is a computer that has no internet connection at all — it is physically or logically isolated from any external network. This is common in secure government, financial, or enterprise environments where data cannot leave the building.
>
> **What does "HF Hub" mean?** HuggingFace Hub (`huggingface.co`) is the main online repository where AI models are hosted and downloaded from. When Axon loads a model like `BAAI/bge-m3`, it normally downloads it from there on first use. In offline mode this download is blocked.

Axon supports two modes for restricted-network and air-gapped environments:

| Mode | What it blocks | RAPTOR / GraphRAG | Use when |
|------|---------------|-------------------|----------|
| **`offline_mode`** | All outbound (outgoing) network calls — HuggingFace Hub, web search, cloud LLMs | ❌ Disabled automatically | Fully air-gapped machines with zero internet access |
| **`local_assets_only`** | HuggingFace Hub downloads only — local Ollama/vLLM still works normally | ✅ Stays enabled | Secure environments where all model files must be pre-staged, but a local LLM is allowed |

> **Both modes set `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `HF_HUB_OFFLINE=1`
> before any model is loaded.** These are environment variables that tell the HuggingFace libraries
> not to attempt any downloads. Axon will fail at startup with a clear error if a required model
> file is missing — it never silently attempts a network call.

---

## 1. Offline Mode — Strict No-Egress

### What it does

- Blocks all outgoing network calls (HuggingFace Hub, cloud LLMs, Brave web search)
- Sets the three HuggingFace offline environment variables automatically
- Automatically disables `truth_grounding` (Brave web search fallback — requires internet)
- Automatically disables `raptor` and `graph_rag` — both need to send text to your LLM for summarisation and entity extraction; even a local Ollama LLM counts as an outgoing call from Axon's perspective, so these features are conservatively turned off. Use `local_assets_only` instead if you need them.
- Resolves model names like `BAAI/bge-m3` to local folder paths using `local_models_dir`

### Minimum config.yaml

```yaml
llm:
  provider: ollama
  model: llama3.1:8b

embedding:
  provider: sentence_transformers
  model: /mnt/aimodels/all-MiniLM-L6-v2   # absolute path to pre-downloaded model

offline:
  enabled: true
  local_models_dir: /mnt/aimodels          # root for HF model directories

vector_store:
  provider: chroma                          # chroma or lancedb (both local)
```

### Compatible providers

| Layer | Compatible | Incompatible |
|-------|-----------|-------------|
| **LLM** | `ollama`, `vllm` | `openai`, `gemini`, `grok`, `github_copilot`, `ollama_cloud` |
| **Embedding** | `sentence_transformers`, `fastembed`, `ollama` | `openai` |
| **Vector store** | `chroma`, `lancedb` | `qdrant` (remote) |
| **Web search** | — | Brave (disabled automatically) |

---

## 2. Local-Assets-Only Mode — Strict HF Enforcement

### What it does

- Sets the same HF offline env vars as `offline_mode`
- Resolves all model names to local folder paths: the embedding model, reranker, and optional NLP helper models (GLiNER for entity recognition, REBEL for relationship extraction, LLMLingua for compression)
- Runs a **preflight check** at startup — prints a table showing whether each model file is `[local]`, in the HuggingFace cache (`[hf_cache]`), missing locally but downloadable (`[remote]`), or `[MISSING]`
- **Stops immediately** with a clear error message if any required model file is not found — no silent network attempts
- RAPTOR and GraphRAG **remain enabled** — they call your local LLM (Ollama/vLLM), not HuggingFace Hub

### Minimum config.yaml

```yaml
llm:
  provider: ollama
  model: llama3.1:8b

embedding:
  provider: sentence_transformers
  model: BAAI/bge-m3                        # resolved via embedding_models_dir below

offline:
  local_assets_only: true

local_dirs:
  embedding_models_dir: /mnt/aimodels/embedding   # sentence-transformers / fastembed models
  hf_models_dir: /mnt/aimodels/hf                 # reranker, GLiNER, REBEL, LLMLingua
  tokenizer_cache_dir: /mnt/aimodels/tiktoken      # tiktoken BPE cache
  ollama_models_dir: /mnt/aimodels/ollama          # Ollama model blobs (sets OLLAMA_MODELS)
```

### Preflight audit output

On startup Axon logs a model asset table:

```
Model asset audit:
  [local]         embedding     /mnt/aimodels/embedding/BAAI--bge-m3
  [local]         reranker      /mnt/aimodels/hf/BAAI--bge-reranker-base
  [n/a]           gliner        (disabled)
  [n/a]           rebel         (disabled)
  [n/a]           llmlingua     (disabled)
  [local]         tokenizer     /mnt/aimodels/tiktoken
```

If any active model shows `[MISSING]` or `[remote]`, startup raises `RuntimeError` with the list of missing assets.

---

## 3. Model Directory Layout

### Model name resolution

When you write a model name like `BAAI/bge-m3` in `config.yaml`, Axon tries two folder name patterns inside each configured root directory:

```
<root>/bge-m3              # short name — just the part after the slash
<root>/BAAI--bge-m3        # HuggingFace slug — slashes replaced with double dashes
```

The first folder that exists on disk is used. If you give an absolute path (starting with `/` on Linux/Mac, or a drive letter like `C:\` on Windows), it is used directly with no lookup.

### Example layout

```
/mnt/aimodels/
├── embedding/
│   ├── BAAI--bge-m3/            # resolved by embedding_models_dir
│   └── all-MiniLM-L6-v2/
├── hf/
│   ├── BAAI--bge-reranker-base/ # cross-encoder reranker
│   ├── urchade--gliner_medium/  # GLiNER NER (if graph_rag_ner_backend: gliner)
│   └── Babelscape--rebel-large/ # REBEL relations (if graph_rag_relation_backend: rebel)
├── tiktoken/                    # tiktoken BPE encodings
│   └── cl100k_base.tiktoken
└── ollama/                      # Ollama model blobs (set OLLAMA_MODELS or ollama_models_dir)
    └── manifests/
```

### Single-root fallback

If per-type directories are not set, `local_models_dir` is used as the single fallback root for all model types:

```yaml
offline:
  enabled: true
  local_models_dir: /mnt/aimodels    # searched for all model types
```

---

## 4. Pre-Downloading Models (Online Machine → Air-Gap Transfer)

Run these commands on a machine with internet access, then copy the output directories to the air-gapped host via USB drive or a secure file transfer.

> **How to run Python snippets below:** save the code to a `.py` file and run `python myfile.py`, or paste it line by line into a Python interactive session (`python` with no arguments).

### Embedding model (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer

# Downloads the model from HuggingFace and saves it to a local folder
model = SentenceTransformer("BAAI/bge-m3")
model.save("/mnt/aimodels/embedding/BAAI--bge-m3")        # Linux / macOS
# model.save("C:\\aimodels\\embedding\\BAAI--bge-m3")     # Windows
```

### Cross-encoder reranker

The reranker re-scores retrieved chunks to improve relevance — it is only needed if `rerank: true` in your config.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")
model.save("/mnt/aimodels/hf/BAAI--bge-reranker-base")    # Linux / macOS
# model.save("C:\\aimodels\\hf\\BAAI--bge-reranker-base") # Windows
```

### FastEmbed model (alternative to sentence-transformers)

```python
from fastembed import TextEmbedding

# cache_dir is where FastEmbed saves the model files
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/mnt/aimodels/fastembed")
# Windows: cache_dir="C:\\aimodels\\fastembed"
```

### Ollama model

Ollama does not have an export command. The simplest method is to copy the Ollama model folder directly from the online machine to the air-gapped host.

**Linux / macOS:**
```bash
# On online machine — pull the model first
ollama pull llama3.1:8b

# Then copy the entire Ollama models directory to a USB drive or network share
cp -r ~/.ollama/models /media/usb/ollama-models

# On air-gapped machine — restore to the same location
cp -r /media/usb/ollama-models/* ~/.ollama/models/
# Restart Ollama, then verify
ollama list
```

**Windows:**
```powershell
# On online machine
ollama pull llama3.1:8b

# Copy the models folder (usually located here)
Copy-Item -Recurse "$env:USERPROFILE\.ollama\models" "D:\usb\ollama-models"

# On air-gapped machine
Copy-Item -Recurse "D:\usb\ollama-models\*" "$env:USERPROFILE\.ollama\models"
# Restart Ollama, then verify
ollama list
```

### tiktoken cache (for chunk size estimation)

`tiktoken` is the tokenizer used to count how many "tokens" (roughly, word pieces) fit in a chunk. Axon uses it to measure chunk sizes even when not using OpenAI as the LLM provider.

```python
import tiktoken, shutil, os

# Download the tokenizer file from the internet (run this on the online machine)
enc = tiktoken.get_encoding("cl100k_base")

# Find where tiktoken saved it and copy to your transfer location
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tiktoken")
shutil.copytree(cache_dir, "/mnt/aimodels/tiktoken", dirs_exist_ok=True)
# Windows: shutil.copytree(cache_dir, "C:\\aimodels\\tiktoken", dirs_exist_ok=True)
```

> **Where is the tiktoken cache on Windows?** Usually `C:\Users\<you>\.cache\tiktoken\`. You can find it by running `python -c "import os; print(os.path.join(os.path.expanduser('~'), '.cache', 'tiktoken'))"` on the online machine.

---

## 5. Complete Air-Gap config.yaml

**Linux / macOS** (paths starting with `/`):

```yaml
llm:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434     # local Ollama daemon — no internet needed

embedding:
  provider: sentence_transformers
  model: BAAI/bge-m3                  # Axon resolves this name to a local folder automatically

rag:
  top_k: 10
  hybrid_search: true
  rerank: true
  rerank_model: BAAI/bge-reranker-base
  cite: true
  discuss: false                        # disable: no internet to fall back to for general knowledge
  raptor: false                         # disable in offline_mode (needs LLM calls during ingest)
  graph_rag: false                      # same

vector_store:
  provider: chroma                      # fully local; lancedb also works

offline:
  enabled: true
  local_models_dir: /mnt/aimodels      # fallback root if per-type dirs below are not set

local_dirs:
  embedding_models_dir: /mnt/aimodels/embedding
  hf_models_dir: /mnt/aimodels/hf
  tokenizer_cache_dir: /mnt/aimodels/tiktoken
  ollama_models_dir: /mnt/aimodels/ollama
```

**Windows** (use backslashes or forward slashes — both work in YAML):

```yaml
llm:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434

embedding:
  provider: sentence_transformers
  model: BAAI/bge-m3

rag:
  top_k: 10
  hybrid_search: true
  rerank: true
  rerank_model: BAAI/bge-reranker-base
  cite: true
  discuss: false
  raptor: false
  graph_rag: false

vector_store:
  provider: chroma

offline:
  enabled: true
  local_models_dir: C:/aimodels

local_dirs:
  embedding_models_dir: C:/aimodels/embedding
  hf_models_dir: C:/aimodels/hf
  tokenizer_cache_dir: C:/aimodels/tiktoken
  ollama_models_dir: C:/aimodels/ollama
```

> **Want RAPTOR and GraphRAG too?** Replace `offline: { enabled: true }` with
> `offline: { local_assets_only: true }`. This still blocks HuggingFace downloads and
> enforces that all model files are local, but it keeps RAPTOR and GraphRAG enabled
> because they call your local Ollama rather than downloading anything.

---

## 6. Environment Variables

The following environment variables are set automatically by Axon when offline or local-assets-only mode is active. You can also set them before launch to take effect before Axon initialises:

| Variable | Set by | Effect |
|----------|--------|--------|
| `TRANSFORMERS_OFFLINE` | Both modes | Blocks HuggingFace Transformers from downloading |
| `HF_DATASETS_OFFLINE` | Both modes | Blocks HuggingFace Datasets from downloading |
| `HF_HUB_OFFLINE` | Both modes | Blocks HuggingFace Hub client |
| `OLLAMA_MODELS` | `ollama_models_dir` set | Redirects Ollama blob resolution to local path |
| `TIKTOKEN_CACHE_DIR` | `tokenizer_cache_dir` set | Redirects tiktoken BPE file lookups |

---

## 7. Troubleshooting

**`RuntimeError: local_assets_only is ON but the following model assets are not available locally`**

The preflight audit could not find one or more models in the configured directories. Check:
1. The directory exists and contains either `<short-name>/` or `<org>--<name>/` subdirectories
2. `embedding_models_dir` and `hf_models_dir` are set correctly in `config.yaml`
3. Run with `LOG_LEVEL=DEBUG` to see the full path resolution trace

**`WARNING: Local model resolution: 'BAAI/bge-m3' not found in [...]`**

The model ID was not resolved to a local path. Axon will attempt a remote download (and fail if `TRANSFORMERS_OFFLINE=1`). Check the directory layout matches the [expected structure](#3-model-directory-layout) above.

**Ollama responds with `model not found`**

If `ollama_models_dir` is set, ensure the Ollama daemon was restarted after the `OLLAMA_MODELS` env var was applied, or set `OLLAMA_MODELS` before starting the daemon.

**RAPTOR / GraphRAG silently disabled**

Expected behaviour in `offline_mode`. Switch to `local_assets_only` if you need these features and have a local LLM (Ollama or vLLM) available.

---

*See [MODEL_GUIDE.md](MODEL_GUIDE.md) for per-provider `config.yaml` examples.*
*See [ADMIN_REFERENCE.md § 6.6](ADMIN_REFERENCE.md) for the full offline config reference.*
*See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for additional error fixes.*
