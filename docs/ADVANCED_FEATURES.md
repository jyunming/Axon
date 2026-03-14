# Axon Advanced Features Guide

Axon provides several state-of-the-art RAG capabilities to improve retrieval accuracy and answer quality. This guide explains how they work and how to configure them.

## 1. GraphRAG (Entity-Centric Retrieval)

**What it does:** Standard RAG breaks documents into chunks and searches them by semantic similarity. GraphRAG goes further by using an LLM during ingestion to extract "entities" (people, places, concepts) and their relationships, building a knowledge graph. At query time, it looks for entities in your question and retrieves interconnected chunks.

**Why use it?** Perfect for "global" questions ("What are the main themes?", "Connect the dots between Project X and Person Y") that span across multiple documents.

**How to use:**
- **CLI:** `axon --graph-rag`
- **REPL:** `/rag graph-rag`
- **config.yaml:**
  ```yaml
  rag:
    graph_rag: true
  ```

## 2. Semantic Chunking (Token-Aware)

**What it does:** Instead of splitting text arbitrarily after 1000 characters (which can cut sentences in half), Semantic Chunking uses NLP rules to identify sentence boundaries, then groups complete sentences until it hits a specific token limit (using `tiktoken`).

**Why use it?** Essential for preserving the true meaning of text, preventing context loss, and significantly improving the quality of extracted entities for GraphRAG.

**How to use:**
- **CLI:** `axon --chunk-strategy semantic`
- **config.yaml:**
  ```yaml
  chunk:
    strategy: "semantic"
    size: 500 # Note: This is in tokens now, not characters
    overlap: 50
  ```

## 3. RAPTOR (Hierarchical Tree Retrieval)

**What it does:** RAPTOR clusters similar chunks together and uses an LLM to write a summary of each cluster. It then clusters the summaries, creating a tree. You can retrieve highly specific facts (from the leaf chunks) or broad summaries (from the top nodes).

**Why use it?** Great for reading long documents like books or annual reports where you might ask both specific detail questions and broad summary questions.

**How to use:**
- **CLI:** `axon --raptor`
- **REPL:** `/rag raptor`
- **config.yaml:**
  ```yaml
  rag:
    raptor: true
  ```

## 4. Context Compression

**What it does:** After retrieving relevant chunks, it sends them to a lightweight LLM task to extract *only* the specific sentences that actually answer your query, discarding the "fluff".

**Why use it?** Reduces prompt bloat, lowers API costs, and helps the final LLM focus only on the facts, reducing hallucinations.

**How to use:**
- **CLI:** `axon --compress`
- **REPL:** `/rag compress`
- **config.yaml:**
  ```yaml
  context_compression:
    enabled: true
  ```

## 5. Query Transformations

Axon can rewrite your queries behind the scenes to find better matches:

- **HyDE (Hypothetical Document Embeddings):** Generates a fake "perfect" answer to your question, then searches the database for chunks that look like the fake answer. Best for vague questions.
  - Enable: `--hyde` or `/rag hyde`
- **Multi-Query:** Asks the LLM to rephrase your question in 3 different ways, searches all 3, and merges the results.
  - Enable: `--multi-query` or `/rag multi`
- **Step-Back:** Asks the LLM to generate a broader, higher-level version of your question (e.g., "What is the history of physics?" instead of "When did Einstein publish special relativity?").
  - Enable: `--step-back` or `/rag step-back`
- **Decompose:** Breaks a complex question with "and" into multiple simple questions, searches them independently.
  - Enable: `--decompose` or `/rag decompose`
