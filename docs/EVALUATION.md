# Axon Evaluation Guide

Evaluation answers the question: *"How good are the answers Axon gives?"*

Axon ships with two evaluation layers:

- **Smoke tests** — fast checks that run automatically on every code change. No internet, no AI model needed. They verify the retrieval math is correct.

- **RAGAS evaluation** — a full end-to-end quality run that sends real questions through Axon and scores how good the answers are, using a local LLM (no OpenAI API key required).

> **What is RAGAS?** RAGAS (Retrieval-Augmented Generation Assessment) is an open-source Python library that measures RAG pipeline quality across four dimensions: does the answer match the retrieved text, did retrieval find the right information, and does the answer actually address the question. See [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas).

---

## 1. Smoke Tests (CI)

`tests/test_eval_smoke.py` verifies that the four core RAG quality signals are computable and produce correct values against a tiny mocked corpus. No live model or API key needed.

```bash

# Run eval smoke tests only

python -m pytest -m eval tests/test_eval_smoke.py -v --no-cov

# Run as part of the full test suite

python -m pytest tests/ -v --no-cov

```

### Metrics covered

> **What does `@k` mean?** The `k` is the number of chunks retrieved (e.g. `top_k=5`). "Precision@5" means: of the 5 chunks retrieved, what fraction were actually useful?

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Precision@k** | Of the chunks Axon retrieved, what fraction were actually relevant to the question? | ≥ 0.7 |
| **Context relevance** | Does the retrieved text contain the words/concepts needed to answer the question? | ≥ 0.6 |
| **Answer faithfulness** | Is the answer based on the retrieved text, or did the LLM make things up? | ≥ 0.8 |
| **Recall@k** | Were all the truly relevant documents included in the top-k results? | ≥ 0.7 |

These checks use simple keyword matching — no additional AI model or internet connection required. They catch regressions (things that used to work but broke) in retrieval ordering and context assembly.

---

## 2. RAGAS Evaluation (Full Quality Scoring)

`scripts/evaluate.py` runs a complete end-to-end evaluation using [RAGAS](https://github.com/explodinggradients/ragas) scored by a local Ollama LLM. No OpenAI API key is required.

### Prerequisites

```bash

pip install ragas langchain-community datasets

ollama pull llama3.1:8b

ollama pull nomic-embed-text    # embedding model used by RAGAS

```

### Run an evaluation

```bash

python scripts/evaluate.py \

  --testset examples/eval_testset.jsonl \

  --config config.yaml

```

This will:

1. Load `AxonBrain` from your `config.yaml`

2. Run each question through the full RAG pipeline (retrieval → context → synthesis)

3. Score results with RAGAS using local Ollama models

4. Write a Markdown report to `eval_report_<timestamp>.md`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--testset` | — *(required)* | Path to JSONL testset file |
| `--config` | `config.yaml` | Path to Axon config file |
| `--model` | — | Override LLM model; omit to use the value from `config.yaml` |
| `--output` | `eval_report_<ts>.md` | Output report file path |

---

## 3. Testset Format

> **What is JSONL?** JSONL (JSON Lines) is a text file where each line is a separate, valid JSON object. It is a simple way to store a list of records. You can edit it in any text editor.

Testsets are JSONL files — one JSON object per line:

```jsonl

{"question": "How does hybrid search work?", "ground_truth": "Hybrid search combines vector search with BM25 keyword search, merging results using Reciprocal Rank Fusion."}

{"question": "What embedding models are supported?", "ground_truth": "Sentence Transformers, Ollama, FastEmbed, and OpenAI embeddings are supported."}

```

| Field | Required | Description |
|-------|----------|-------------|
| `question` | yes | The query to send to the RAG pipeline |
| `ground_truth` | yes | The expected answer — used to compute `context_recall` |

A starter testset is included at `examples/eval_testset.jsonl`.

### Building a good testset

- **Cover your actual use cases** — use questions you expect real users to ask

- **Write specific ground truths** — vague ground truths hurt `context_recall` scoring

- **Include hard cases** — questions that require combining multiple documents

- **Aim for 20–50 questions** — enough for stable aggregate scores, fast enough to run regularly

---

## 4. RAGAS Metrics Explained

| Metric | Range | Good score | What a low score means |
|--------|-------|-----------|------------------------|
| **faithfulness** | 0–1 | ≥ 0.8 | Answers contain claims not supported by the retrieved context (hallucination) |
| **context_recall** | 0–1 | ≥ 0.7 | The retrieved context is missing information needed to answer correctly |
| **context_precision** | 0–1 | ≥ 0.7 | Retrieved chunks contain a lot of irrelevant noise — consider lower `top_k` or reranking |
| **answer_relevancy** | 0–1 | ≥ 0.8 | Answers don't directly address what was asked |

### How to use these scores

- **Low `context_recall`** → try increasing `top_k`, enabling `hybrid_search`, or adding HyDE

- **Low `context_precision`** → try enabling `rerank`, raising `similarity_threshold`, or lowering `top_k`

- **Low `faithfulness`** → your LLM is hallucinating beyond retrieved context; try enabling `cite` + `compress`

- **Low `answer_relevancy`** → check your system prompt and consider enabling `step_back` or `decompose`

---

## 5. Integrating Evals into CI

> **What is CI?** CI (Continuous Integration) means automatically running tests every time you push code changes — usually via GitHub Actions. "Gating on regression" means the CI job will fail if a metric gets worse than it was before.

Add this to your GitHub Actions workflow file to run smoke tests on every push:

```yaml

- name: Run eval smoke tests

  run: python -m pytest -m eval tests/test_eval_smoke.py -v --no-cov

```

For RAGAS scoring in CI (requires Ollama available on the runner):

```yaml

- name: Run RAGAS evaluation

  run: |

    python scripts/evaluate.py \

      --testset examples/eval_testset.jsonl \

      --output eval_report.md

  continue-on-error: true   # non-blocking — report is an artifact

- name: Upload eval report

  uses: actions/upload-artifact@v4

  with:

    name: eval-report

    path: eval_report.md

```

The existing CI workflow runs smoke tests automatically on every push via the `eval` marker.

---

## 6. Example Evaluation Report

```markdown

# RAG Evaluation Report

**Generated:** 2026-03-26 14:30

**Samples:** 10

## Scores

| Metric              | Score  |
|---------------------|--------|
| faithfulness        | 0.8750 |
| context_recall      | 0.7200 |
| context_precision   | 0.6900 |
| answer_relevancy    | 0.8400 |

```

---

*See [ADVANCED_RAG.md](ADVANCED_RAG.md) for tuning guidance on each RAG technique.*

*See [ADMIN_REFERENCE.md § 6.4](ADMIN_REFERENCE.md) for the full list of RAG flags and their defaults.*

