import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

from axon.main import AxonBrain, AxonConfig

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BenchmarkCompression")


def _make_brain(tmp_path, strategy="none"):
    config = AxonConfig(
        vector_store_path=str(tmp_path / "vs"),
        bm25_path=str(tmp_path / "bm25"),
        projects_root=str(tmp_path / "projects"),
        compress_context=True if strategy != "none" else False,
        compression_strategy=strategy,
        top_k=5,
        hybrid_search=False,
        similarity_threshold=0.1,  # inclusive
    )

    # Mock components
    with patch("axon.main.OpenEmbedding") as MockEmb, patch("axon.main.OpenLLM") as MockLLM, patch(
        "axon.main.OpenReranker"
    ):
        mock_emb = MockEmb.return_value
        # Return a fixed-size vector for each input text
        mock_emb.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_query.return_value = [0.1] * 384

        mock_llm = MockLLM.return_value

        def mock_complete(prompt, system_prompt=None, chat_history=None):
            # If it's a compression prompt (Sentence strategy)
            if "Extract only the sentences" in prompt:
                # Just return the first 200 chars as a "summary"
                passage = prompt.split("Passage:\n")[-1]
                return passage[:200] + "..."

            # If it's a final query prompt
            return "Mocked answer based on context."

        mock_llm.complete.side_effect = mock_complete

        brain = AxonBrain(config)
        return brain, mock_llm


def run_benchmark():
    root = Path(__file__).parent.parent
    fixture_path = root / "tests" / "fixtures" / "ai_article.txt"
    if not fixture_path.exists():
        # Fallback to a synthetic doc if fixture missing
        text = "Artificial Intelligence (AI) is intelligence demonstrated by machines... " * 100
    else:
        text = fixture_path.read_text(encoding="utf-8")

    tmp_dir = root / ".test_tmp_benchmark"
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    queries = [
        "What are the main risks of AI discussed in the article?",
        "How does the author suggest mitigating bias in LLMs?",
        "What is the future outlook for local-first retrieval systems?",
    ]

    strategies = ["none", "sentence", "llmlingua"]
    results = {s: [] for s in strategies}

    for strategy in strategies:
        logger.info(f"🚀 Benchmarking strategy: {strategy}")
        brain, mock_llm = _make_brain(tmp_dir / strategy, strategy=strategy)

        # Ingest the article
        doc = {"id": "ai_article", "text": text, "metadata": {"source": "ai_article.txt"}}
        brain.ingest([doc])

        for q in queries:
            t0 = time.time()
            answer = brain.query(q)
            elapsed = time.time() - t0

            diag = brain._last_diagnostics
            results[strategy].append(
                {
                    "query": q,
                    "latency_sec": elapsed,
                    "pre_tokens": getattr(diag, "compression_pre_tokens", 0),
                    "post_tokens": getattr(diag, "compression_post_tokens", 0),
                    "ratio": getattr(diag, "compression_ratio", 1.0),
                    "fallback": getattr(diag, "compression_fallback_reason", ""),
                    "answer_preview": answer[:50],
                }
            )

        brain.close()

    # Aggregate results
    summary = []
    summary.append("# Compression Benchmark Report (Story 3.4)")
    summary.append("")
    summary.append("| Strategy | Avg Latency (s) | Avg Reduction | Success Rate |")
    summary.append("|----------|-----------------|---------------|--------------|")

    for s in strategies:
        data = results[s]
        avg_latency = sum(r["latency_sec"] for r in data) / len(data)
        avg_ratio = sum(r["ratio"] for r in data) / len(data)
        reduction = f"{(1 - avg_ratio) * 100:.1f}%"
        success_rate = sum(1 for r in data if not r["fallback"]) / len(data)
        summary.append(f"| {s} | {avg_latency:.3f} | {reduction} | {success_rate*100:.0f}% |")

    summary.append("")
    summary.append("## Detailed Results")
    for s in strategies:
        summary.append(f"### {s}")
        for r in results[s]:
            summary.append(f"- **Q:** {r['query']}")
            summary.append(f"  - Latency: {r['latency_sec']:.3f}s")
            summary.append(
                f"  - Tokens: {r['pre_tokens']} -> {r['post_tokens']} ({r['ratio']:.2f} ratio)"
            )
            if r["fallback"]:
                summary.append(f"  - **Fallback:** {r['fallback']}")

    report_path = root / "Qualification" / "GeminiQual" / "compression_benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(summary))

    # Also save raw JSON
    json_path = root / "Qualification" / "GeminiQual" / "compression_benchmark_raw.json"
    json_path.write_text(json.dumps(results, indent=2))

    print(f"✅ Benchmark complete. Report saved to {report_path}")


if __name__ == "__main__":
    run_benchmark()
