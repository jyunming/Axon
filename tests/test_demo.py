"""
tests/test_demo.py

End-to-end (demo) tests — real embeddings, real ChromaDB, no LLM mocked.

Each test ingests documents, runs retrieval, and verifies the correct result
surfaces.  Requires: sentence-transformers, chromadb, rank_bm25.
Marked with @pytest.mark.demo so they can be run in isolation:
    pytest tests/test_demo.py -m demo -v
"""

import json
import os

import pytest

from axon.main import AxonBrain, AxonConfig

pytestmark = pytest.mark.demo


# ---------------------------------------------------------------------------
# Shared fixture: a real brain backed by tmp ChromaDB + BM25
# ---------------------------------------------------------------------------


def _make_brain(tmp_path, **overrides) -> AxonBrain:
    """Return a real AxonBrain backed by temp directories.

    parent_chunk_size=0 disables parent-doc splitting so chunk IDs stay clean
    (no _p0_chunk_0 suffix) and the type-specific chunking path is exercised.
    """
    chroma_dir = str(tmp_path / "chroma")
    bm25_dir = str(tmp_path / "bm25")
    config = AxonConfig(
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        vector_store="chroma",
        vector_store_path=chroma_dir,
        bm25_path=bm25_dir,
        hybrid_search=True,
        rerank=False,
        raptor=False,
        graph_rag=False,
        similarity_threshold=0.0,
        top_k=5,
        parent_chunk_size=0,
        **overrides,
    )
    return AxonBrain(config)


# ---------------------------------------------------------------------------
# Demo 1: Basic ingest + retrieval (weighted fusion)
# ---------------------------------------------------------------------------


class TestDemo_BasicIngestRetrieve:
    """Ingest three clearly distinct documents; verify the right one surfaces."""

    @pytest.fixture
    def brain(self, tmp_path):
        # Use weighted fusion so vector similarity dominates for semantically clear queries
        return _make_brain(tmp_path, hybrid_mode="weighted")

    def test_retrieves_correct_document(self, brain):
        docs = [
            {
                "id": "photosynthesis",
                "text": (
                    "Photosynthesis is the process by which plants use sunlight, water, "
                    "and carbon dioxide to produce oxygen and energy in the form of glucose. "
                    "Chlorophyll in plant cells absorbs light energy to drive this reaction."
                ),
                "metadata": {"source": "biology.txt"},
            },
            {
                "id": "sorting_algorithms",
                "text": (
                    "QuickSort is a divide-and-conquer sorting algorithm. It picks a pivot "
                    "element and partitions the array into two sub-arrays around the pivot. "
                    "Average time complexity is O(n log n)."
                ),
                "metadata": {"source": "cs.txt"},
            },
            {
                "id": "roman_history",
                "text": (
                    "Julius Caesar was a Roman general and statesman. He played a critical "
                    "role in the transformation of the Roman Republic into the Roman Empire. "
                    "He was assassinated on the Ides of March, 44 BC."
                ),
                "metadata": {"source": "history.txt"},
            },
        ]
        brain.ingest(docs)

        results = brain._execute_retrieval("How do plants convert sunlight to energy?")
        top_id = results["results"][0]["id"]
        assert top_id.startswith(
            "photosynthesis"
        ), f"Expected id starting with 'photosynthesis', got '{top_id}'"

    def test_sorting_query_finds_cs_doc(self, brain):
        docs = [
            {
                "id": "bio",
                "text": "Mitosis is cell division producing two identical daughter cells.",
                "metadata": {},
            },
            {
                "id": "cs",
                "text": "MergeSort recursively divides arrays and merges sorted halves. Time: O(n log n).",
                "metadata": {},
            },
            {"id": "hist", "text": "The Battle of Hastings took place in 1066 AD.", "metadata": {}},
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("What is the time complexity of merge sort?")
        top_id = results["results"][0]["id"]
        assert top_id.startswith("cs"), f"Expected id starting with 'cs', got '{top_id}'"

    def test_retrieval_returns_nonempty_for_broad_query(self, brain):
        docs = [
            {"id": f"doc{i}", "text": f"Document {i} about topic {i}.", "metadata": {}}
            for i in range(5)
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("document")
        assert len(results["results"]) > 0


# ---------------------------------------------------------------------------
# Demo 2: RRF hybrid mode
# ---------------------------------------------------------------------------


class TestDemo_RRFHybridMode:
    """Verify RRF fusion returns results and surfaces correct docs."""

    @pytest.fixture
    def brain(self, tmp_path):
        return _make_brain(tmp_path, hybrid_mode="rrf")

    def test_rrf_returns_results(self, brain):
        docs = [
            {
                "id": "ml",
                "text": "Machine learning models learn from training data to make predictions.",
                "metadata": {},
            },
            {
                "id": "db",
                "text": "Relational databases store data in tables with rows and columns.",
                "metadata": {},
            },
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("training data for prediction models")
        assert len(results["results"]) > 0

    def test_rrf_top_result_is_relevant(self, brain):
        docs = [
            {
                "id": "python",
                "text": "Python is a high-level interpreted programming language known for its simplicity.",
                "metadata": {},
            },
            {
                "id": "java",
                "text": "Java is a statically typed compiled language running on the JVM.",
                "metadata": {},
            },
            {
                "id": "cooking",
                "text": "Pasta is cooked by boiling in salted water for 8-12 minutes.",
                "metadata": {},
            },
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("interpreted programming language")
        top_id = results["results"][0]["id"]
        assert top_id.startswith("python"), f"Expected id starting with 'python', got '{top_id}'"

    def test_rrf_vector_count_nonzero(self, brain):
        docs = [
            {
                "id": "x",
                "text": "The sky is blue due to Rayleigh scattering of sunlight.",
                "metadata": {},
            }
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("why is sky blue")
        assert results["vector_count"] > 0


# ---------------------------------------------------------------------------
# Demo 3: Dataset type detection on realistic content
# ---------------------------------------------------------------------------


class TestDemo_DatasetTypeDetection:
    """_detect_dataset_type produces correct type for realistic document content."""

    @pytest.fixture
    def brain(self, tmp_path):
        return _make_brain(tmp_path)

    def test_python_source_file(self, brain):
        doc = {
            "id": "app.py",
            "text": "import os\nimport sys\n\ndef main():\n    print('hello')\n\nclass Config:\n    debug = False\n",
            "metadata": {"source": "app.py"},
        }
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "codebase"

    def test_typescript_source_file(self, brain):
        doc = {
            "id": "index.ts",
            "text": "import { Component } from '@angular/core';\n\nclass AppComponent {\n  title = 'app';\n}\n",
            "metadata": {"source": "index.ts"},
        }
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "codebase"

    def test_csv_data_as_knowledge(self, brain):
        text = "name,age,city,salary,department\nAlice,30,NYC,90000,Engineering\nBob,25,LA,70000,Marketing\n"
        doc = {"id": "employees.csv", "text": text, "metadata": {"source": "employees.csv"}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "knowledge"

    def test_academic_paper(self, brain):
        text = (
            "Abstract\nIn this paper we present a novel approach to information retrieval. "
            "Introduction\nThe field of NLP has grown rapidly. "
            "Conclusion\nWe demonstrated improvements on the BEIR benchmark. "
            "References\n[1] Karpukhin et al., 2020. DOI: 10.1000/x. arXiv:2004.00210"
        )
        doc = {"id": "paper.pdf", "text": text, "metadata": {"source": "paper.pdf"}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "paper"

    def test_markdown_documentation(self, brain):
        text = (
            "# Getting Started\n\n"
            "## Installation\n\nStep 1: Install dependencies\n\n"
            "**Note:** Requires Python 3.10+\n\n"
            "## Configuration\n\nEdit `config.yaml` to set your preferences."
        )
        doc = {"id": "README.md", "text": text, "metadata": {"source": "README.md"}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "doc"

    def test_chat_conversation_as_discussion(self, brain):
        messages = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
        ]
        doc = {"id": "chat.json", "text": json.dumps(messages), "metadata": {}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "discussion"

    def test_dataset_type_config_override(self, tmp_path):
        """When config.dataset_type is set explicitly, it overrides detection."""
        brain = _make_brain(tmp_path, dataset_type="paper")
        doc = {"id": "code.py", "text": "def foo(): pass", "metadata": {"source": "code.py"}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "paper"


# ---------------------------------------------------------------------------
# Demo 4: Type-specific chunking stores dataset_type in metadata
# ---------------------------------------------------------------------------


class TestDemo_TypeSpecificChunking:
    """Ingest docs and verify dataset_type lands in chunk metadata."""

    @pytest.fixture
    def brain(self, tmp_path):
        return _make_brain(tmp_path)

    def test_code_doc_gets_codebase_type_in_metadata(self, brain, tmp_path):
        from unittest.mock import patch

        captured_metadata = []

        original_add = brain.vector_store.add

        def capture_add(ids, embeddings, texts, metadatas):
            captured_metadata.extend(metadatas)
            return original_add(ids, embeddings, texts, metadatas)

        doc = {
            "id": "script.py",
            "text": "import os\nimport sys\n\ndef run():\n    pass\n\nclass App:\n    debug = True\n",
            "metadata": {"source": "script.py"},
        }
        with patch.object(brain.vector_store, "add", side_effect=capture_add):
            brain.ingest([doc])

        assert len(captured_metadata) > 0, "No chunks were produced"
        assert all(
            m.get("dataset_type") == "codebase" for m in captured_metadata
        ), f"Expected all chunks to have dataset_type='codebase', got: {[m.get('dataset_type') for m in captured_metadata]}"

    def test_csv_doc_gets_knowledge_type_in_metadata(self, brain):
        from unittest.mock import patch

        captured_metadata = []
        original_add = brain.vector_store.add

        def capture_add(ids, embeddings, texts, metadatas):
            captured_metadata.extend(metadatas)
            return original_add(ids, embeddings, texts, metadatas)

        rows = ["col1,col2,col3,col4,col5"] + [f"{i},{i*2},{i*3},{i*4},{i*5}" for i in range(10)]
        doc = {"id": "data.csv", "text": "\n".join(rows), "metadata": {"source": "data.csv"}}

        with patch.object(brain.vector_store, "add", side_effect=capture_add):
            brain.ingest([doc])

        assert len(captured_metadata) > 0
        assert all(m.get("dataset_type") == "knowledge" for m in captured_metadata)


# ---------------------------------------------------------------------------
# Demo 5: Doc versions persist and reload across brain restarts
# ---------------------------------------------------------------------------


class TestDemo_DocVersions:
    def test_versions_survive_brain_restart(self, tmp_path):
        brain1 = _make_brain(tmp_path)
        brain1._doc_versions = {
            "/path/to/file.txt": {
                "content_hash": "abc123",
                "mtime": 1741234567.0,
                "dataset_type": "doc",
            }
        }
        brain1._save_doc_versions()
        brain1.close()

        brain2 = _make_brain(tmp_path)
        loaded = brain2.get_doc_versions()
        assert "/path/to/file.txt" in loaded
        assert loaded["/path/to/file.txt"]["content_hash"] == "abc123"
        brain2.close()

    def test_versions_file_is_json(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._doc_versions = {"a.txt": {"content_hash": "xyz"}}
        brain._save_doc_versions()
        brain.close()

        versions_path = os.path.join(str(tmp_path / "bm25"), ".doc_versions.json")
        assert os.path.exists(versions_path)
        with open(versions_path) as f:
            data = json.load(f)
        assert data["a.txt"]["content_hash"] == "xyz"


# ---------------------------------------------------------------------------
# Demo 6: Health endpoint — real HTTP
# ---------------------------------------------------------------------------


class TestDemo_HealthEndpoint:
    def test_health_503_before_init(self):
        from fastapi.testclient import TestClient

        import axon.api as api_module

        saved = api_module.brain
        api_module.brain = None
        client = TestClient(api_module.app, raise_server_exceptions=False)
        try:
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "initializing"
        finally:
            api_module.brain = saved

    def test_health_200_after_init(self, tmp_path):
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        import axon.api as api_module

        saved = api_module.brain
        mock_brain = MagicMock()
        mock_brain.config.project = "demo"
        api_module.brain = mock_brain
        client = TestClient(api_module.app, raise_server_exceptions=False)
        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
        finally:
            api_module.brain = saved


# ---------------------------------------------------------------------------
# Demo 7: _KNOWN_DIMS drives real embedding dimension
# ---------------------------------------------------------------------------


class TestDemo_EmbeddingDimension:
    def test_minilm_real_dimension(self, tmp_path):
        """sentence_transformers all-MiniLM-L6-v2 should give dimension=384."""
        brain = _make_brain(tmp_path)
        assert brain.embedding.dimension == 384, f"Expected 384, got {brain.embedding.dimension}"
        brain.close()

    def test_ollama_nomic_uses_known_dim(self):
        from axon.main import AxonConfig, OpenEmbedding

        config = AxonConfig(embedding_provider="ollama", embedding_model="nomic-embed-text")
        emb = OpenEmbedding.__new__(OpenEmbedding)
        emb.config = config
        emb.provider = "ollama"
        emb.model = None
        emb.dimension = 0
        emb._load_model()
        assert emb.dimension == 768

    def test_ollama_bge_large_uses_known_dim(self):
        from axon.main import AxonConfig, OpenEmbedding

        config = AxonConfig(embedding_provider="ollama", embedding_model="BAAI/bge-large-en-v1.5")
        emb = OpenEmbedding.__new__(OpenEmbedding)
        emb.config = config
        emb.provider = "ollama"
        emb.model = None
        emb.dimension = 0
        emb._load_model()
        assert emb.dimension == 1024


# ---------------------------------------------------------------------------
# Demo 8: Qdrant config parsing (local path mode — no server required)
# ---------------------------------------------------------------------------


class TestDemo_QdrantConfig:
    def test_qdrant_url_parsed_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "embedding:\n  provider: sentence_transformers\n  model: all-MiniLM-L6-v2\n"
            "vector_store:\n  provider: qdrant\n  path: /tmp/qdrant_test\n"
            "qdrant_url: http://localhost:6333\n"
            "qdrant_api_key: my-key\n",
            encoding="utf-8",
        )
        cfg = AxonConfig.load(str(cfg_file))
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.qdrant_api_key == "my-key"
        assert cfg.vector_store == "qdrant"

    def test_qdrant_url_empty_by_default(self):
        cfg = AxonConfig()
        assert cfg.qdrant_url == ""
        assert cfg.qdrant_api_key == ""


# ---------------------------------------------------------------------------
# Demo 9: max_workers parsed from YAML
# ---------------------------------------------------------------------------


class TestDemo_MaxWorkers:
    def test_max_workers_parsed_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "embedding:\n  provider: sentence_transformers\n  model: all-MiniLM-L6-v2\n"
            "max_workers: 16\n",
            encoding="utf-8",
        )
        cfg = AxonConfig.load(str(cfg_file))
        assert cfg.max_workers == 16

    def test_max_workers_default_is_8(self):
        cfg = AxonConfig()
        assert cfg.max_workers == 8

    def test_brain_executor_uses_max_workers(self, tmp_path):
        brain = _make_brain(tmp_path, max_workers=4)
        assert brain._executor._max_workers == 4
        brain.close()


# ---------------------------------------------------------------------------
# Demo 10: hybrid_mode RRF vs weighted end-to-end
# ---------------------------------------------------------------------------


class TestDemo_HybridModeConfig:
    def test_hybrid_mode_parsed_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "embedding:\n  provider: sentence_transformers\n  model: all-MiniLM-L6-v2\n"
            "rag:\n  hybrid_mode: rrf\n  hybrid_search: true\n",
            encoding="utf-8",
        )
        cfg = AxonConfig.load(str(cfg_file))
        assert cfg.hybrid_mode == "rrf"

    def test_weighted_mode_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "embedding:\n  provider: sentence_transformers\n  model: all-MiniLM-L6-v2\n"
            "rag:\n  hybrid_mode: weighted\n",
            encoding="utf-8",
        )
        cfg = AxonConfig.load(str(cfg_file))
        assert cfg.hybrid_mode == "weighted"

    def test_rrf_retrieval_end_to_end(self, tmp_path):
        brain = _make_brain(tmp_path, hybrid_mode="rrf")
        docs = [
            {
                "id": "rrf_a",
                "text": "Neural networks learn representations from data via backpropagation.",
                "metadata": {},
            },
            {
                "id": "rrf_b",
                "text": "SQL databases use structured query language for data manipulation.",
                "metadata": {},
            },
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("neural network training")
        assert len(results["results"]) > 0
        assert results["results"][0]["id"].startswith("rrf_a")
        brain.close()
