import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from axon.main import AxonBrain, AxonConfig


def _make_brain(tmp_path, crag_enabled=True, truth_grounding=True):
    config = AxonConfig(
        vector_store_path=str(tmp_path / "vs"),
        bm25_path=str(tmp_path / "bm25"),
        projects_root=str(tmp_path / "projects"),
        crag_lite=crag_enabled,
        truth_grounding=truth_grounding,
        similarity_threshold=0.5,
        hybrid_search=False,  # Default
    )
    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM") as MockLLM, patch(
        "axon.main.OpenReranker"
    ):
        brain = AxonBrain(config)
        brain.vector_store = MagicMock()
        brain.bm25 = MagicMock()
        return brain, MockLLM.return_value


class TestCRAGEvaluation:
    """
    Story 2.4: CRAG-lite evaluation set.
    """

    def test_eval_kb_has_answer_high_confidence(self, tmp_path):
        brain, mock_llm = _make_brain(tmp_path, crag_enabled=True, truth_grounding=True)

        # results need vector_score if hybrid_search=False
        results = [
            {
                "id": "d1",
                "text": "Paris is the capital.",
                "score": 0.95,
                "vector_score": 0.95,
                "metadata": {"source": "s1"},
            },
            {
                "id": "d2",
                "text": "Capital of France is Paris.",
                "score": 0.8,
                "vector_score": 0.8,
                "metadata": {"source": "s2"},
            },
            {
                "id": "d3",
                "text": "Something partially related.",
                "score": 0.6,
                "vector_score": 0.6,
                "metadata": {"source": "s3"},
            },
        ]
        brain.vector_store.search.return_value = results
        brain.bm25.search.return_value = []

        with patch.object(brain, "_execute_web_search") as mock_web:
            brain.query("What is the capital of France?")
            assert mock_web.call_count == 0
            print("\n[CRAG-Eval] KB has answer (High Conf): Fallback correctly bypassed.")

    def test_eval_kb_lacks_answer_low_confidence_triggers_fallback(self, tmp_path):
        brain, mock_llm = _make_brain(tmp_path, crag_enabled=True, truth_grounding=True)

        results = [
            {
                "id": "d1",
                "text": "Irrelevant",
                "score": 0.1,
                "vector_score": 0.1,
                "metadata": {"source": "s1"},
            }
        ]
        brain.vector_store.search.return_value = results
        brain.bm25.search.return_value = []

        with patch.object(brain, "_execute_web_search", return_value=[]) as mock_web:
            brain.query("query")
            # Score 0.1 < 0.5 threshold -> filtered_results empty -> Correction triggered
            assert mock_web.call_count == 1
            print("[CRAG-Eval] KB lacks answer (Low Conf): Fallback correctly triggered.")

    def test_eval_comparison_baseline_vs_crag(self, tmp_path):
        test_cases = [
            {
                "name": "High",
                "scores": [0.95, 0.8, 0.6],
                "sources": ["s1", "s2", "s3"],
                "fallback": False,
            },
            {"name": "Low", "scores": [0.3], "sources": ["s1"], "fallback": True},
            {"name": "Empty", "scores": [], "sources": [], "fallback": True},
        ]

        stats = {"fallback_count": 0, "total": len(test_cases)}
        brain, _ = _make_brain(tmp_path, crag_enabled=True, truth_grounding=True)

        for case in test_cases:
            results = [
                {
                    "id": f"id{i}",
                    "text": "text",
                    "score": s,
                    "vector_score": s,
                    "metadata": {"source": src},
                }
                for i, (s, src) in enumerate(zip(case["scores"], case["sources"]))
            ]

            brain.vector_store.search.return_value = results
            brain.bm25.search.return_value = []

            with patch.object(brain, "_execute_web_search", return_value=[]) as mock_web:
                # Use unique query per case to prevent cache hits between iterations
                brain.query(f"q_{case['name']}")
                if mock_web.call_count > 0:
                    stats["fallback_count"] += 1
                assert (mock_web.call_count > 0) == case["fallback"], f"Failed case {case['name']}"

        print(f"[CRAG-Eval] Benchmark stats: {stats}")
        output_path = tmp_path / "crag_eval_results.json"
        output_path.write_text(json.dumps(stats, indent=2))


def run_crag_benchmark():
    import subprocess

    subprocess.run(["pytest", "tests/test_crag_eval.py", "-v", "-s"])


if __name__ == "__main__":
    run_crag_benchmark()
