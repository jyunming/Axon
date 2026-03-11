"""
RAGAS Evaluation Script for Axon.

Evaluates RAG pipeline quality using local LLMs via Ollama.
No OpenAI API required.

Usage:
    python scripts/evaluate.py --testset examples/eval_testset.jsonl --config config.yaml

Testset format (JSONL, one JSON object per line):
    {"question": "What is...", "ground_truth": "The answer is..."}

Output:
    eval_report_<timestamp>.md
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_testset(path: str):
    """Load JSONL testset."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_pipeline(brain, questions):
    """Run RAG pipeline on all questions, collect contexts and answers."""
    results = []
    for q in questions:
        try:
            # Get raw search results (contexts)
            query_embedding = brain.embedding.embed_query(q)
            raw_results = brain.vector_store.search(query_embedding, top_k=brain.config.top_k)
            contexts = [r['text'] for r in raw_results[:5]]
            
            # Get generated answer
            answer = brain.query(q)
            
            results.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
            })
        except Exception as e:
            results.append({"question": q, "answer": f"ERROR: {e}", "contexts": []})
    return results


def evaluate_with_ragas(results, testset, llm_model: str):
    """Score results using RAGAS with local Ollama LLM."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_recall, context_precision, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install ragas langchain-community datasets")
        return None

    # Wire up local LLM
    ollama_llm = LangchainLLMWrapper(ChatOllama(model=llm_model))
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [s.get("ground_truth", "") for s in testset],
    }
    dataset = Dataset.from_dict(data)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall, context_precision, answer_relevancy],
        llm=ollama_llm,
        embeddings=ollama_embeddings,
    )
    return scores


def write_report(scores, results, output_path: str):
    """Write markdown evaluation report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# RAG Evaluation Report",
        f"",
        f"**Generated:** {ts}",
        f"**Samples:** {len(results)}",
        f"",
        f"## Scores",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
    ]
    if scores:
        for metric in ["faithfulness", "context_recall", "context_precision", "answer_relevancy"]:
            val = scores.get(metric, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"| {metric} | {val} |")
    else:
        lines.append("| Scores not available — check RAGAS installation | — |")
    
    lines += [
        f"",
        f"## Score Interpretation",
        f"",
        f"- **faithfulness** (0-1): Fraction of answer claims supported by retrieved context. >0.8 is good.",
        f"- **context_recall** (0-1): Fraction of ground-truth info present in retrieved context. >0.7 is good.",
        f"- **context_precision** (0-1): Fraction of retrieved context that is relevant. >0.7 is good.",
        f"- **answer_relevancy** (0-1): How well the answer addresses the question. >0.8 is good.",
        f"",
        f"## Sample Outputs",
        f"",
    ]
    for i, r in enumerate(results[:5]):  # show first 5
        lines += [
            f"### Sample {i+1}",
            f"**Q:** {r['question']}",
            f"",
            f"**A:** {r['answer'][:300]}{'...' if len(r['answer']) > 300 else ''}",
            f"",
            f"**Contexts retrieved:** {len(r['contexts'])}",
            f"",
        ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"📄 Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Axon with RAGAS")
    parser.add_argument("--testset", required=True, help="Path to JSONL testset file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="LLM model to use (defaults to config value)")
    parser.add_argument("--output", default=None, help="Output report path (default: eval_report_<ts>.md)")
    args = parser.parse_args()

    from axon.main import AxonBrain, AxonConfig
    
    print("🧠 Loading Axon...")
    config = AxonConfig.load(args.config)
    if args.model:
        config.llm_model = args.model
    brain = AxonBrain(config)

    print(f"📋 Loading testset from {args.testset}...")
    testset = load_testset(args.testset)
    questions = [s["question"] for s in testset]
    print(f"   {len(questions)} questions loaded.")

    print("🔍 Running RAG pipeline...")
    results = run_pipeline(brain, questions)

    print("📊 Scoring with RAGAS (local Ollama)...")
    scores = evaluate_with_ragas(results, testset, config.llm_model)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"eval_report_{ts}.md"
    write_report(scores, results, output_path)

    if scores:
        print("\n📈 Results:")
        for k, v in scores.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
