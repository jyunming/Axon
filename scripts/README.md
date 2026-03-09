# scripts/

## prefetch_models.py — HuggingFace model pre-fetcher

Run **once on an internet-connected machine** before moving to a confined workspace.

```bash
# defaults: downloads all-MiniLM-L6-v2 + bge-reranker-base into ./models/
python scripts/prefetch_models.py

# custom output directory (recommended: use absolute path)
python scripts/prefetch_models.py --dir C:/models
python scripts/prefetch_models.py --dir /srv/models

# add extra models
python scripts/prefetch_models.py --dir C:/models --extra BAAI/bge-large-zh-v15
```

After downloading, the script prints the exact `config.yaml` block to paste.

---

## Ollama model transfer (gemma3, gpt-oss, etc.)

Ollama stores all models as content-addressed blobs.  You cannot `ollama pull`
in the confined workspace, but you can physically copy the files.

### Where Ollama stores models

| OS | Path |
|---|---|
| Windows | `C:\Users\<you>\.ollama\models\` |
| Linux / Mac | `~/.ollama/models/` |

The directory has two subdirectories:
- `blobs/` — the actual weight files (large, named by SHA256)
- `manifests/` — tiny JSON files that map model tags to blob lists

### Steps

**1. On the internet machine — pull the models you need:**
```bash
ollama pull gemma3:27b
ollama pull gpt-oss:120b   # or whatever variant you use
```

**2. Find which blobs belong to each model:**
```bash
# Linux / Mac
cat ~/.ollama/models/manifests/registry.ollama.ai/library/gemma3/27b

# Windows PowerShell
Get-Content "$env:USERPROFILE\.ollama\models\manifests\registry.ollama.ai\library\gemma3\27b"
```
The manifest lists `digest` values like `sha256:abc123...`.
Each digest corresponds to a file in `blobs/sha256-abc123...`.

**3. Copy to the confined machine:**

Copy the entire `~/.ollama/models/` directory (or just the blobs + manifests you
need) to the **same path** on the confined machine.

```bash
# Example: rsync to a remote machine
rsync -av ~/.ollama/models/ user@confined-host:~/.ollama/models/

# Or on Windows: robocopy
robocopy "%USERPROFILE%\.ollama\models" "\\confined-host\share\.ollama\models" /MIR
```

**4. Verify on the confined machine:**
```bash
ollama list
# Should show gemma3:27b, gpt-oss:120b, etc. without any download
```

### Model sizes (approximate)

| Model | Size |
|---|---|
| gemma3:27b | ~17 GB |
| gpt-oss:120b | ~70 GB |
| all-MiniLM-L6-v2 | ~90 MB |
| bge-reranker-base | ~270 MB |

---

## Full offline config.yaml example

```yaml
embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2       # resolved to <local_models_dir>/all-MiniLM-L6-v2

llm:
  provider: ollama
  model: gemma3:27b              # served by local Ollama daemon
  base_url: http://localhost:11434

rerank:
  enabled: true
  provider: cross-encoder
  model: bge-reranker-base       # resolved to <local_models_dir>/bge-reranker-base

web_search:
  enabled: false                 # also forced off by offline mode

offline:
  enabled: true
  local_models_dir: C:/models    # absolute path
```
