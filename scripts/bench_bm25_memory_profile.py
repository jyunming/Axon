from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from axon.retrievers import BM25Retriever


def _make_docs(n_docs: int = 30000, n_unique_texts: int = 300) -> list[dict]:
    bases = [
        (
            f"Template {i}: graph retrieval bm25 optimization repeated payload "
            f"entity_{i%53} relation_{i%47} context_{i%41}. "
            "This body is intentionally long to amplify duplicate-text memory cost."
        )
        for i in range(n_unique_texts)
    ]
    out: list[dict] = []
    for i in range(n_docs):
        t = i % n_unique_texts
        # Force a distinct string object even when logical content is identical.
        txt = (bases[t] + " ").rstrip()
        out.append({"id": f"d{i}", "text": txt, "metadata": {"source": f"s{i%200}"}})
    return out


def _run_case(*, mode: str, docs: list[dict]) -> dict:
    os.environ["AXON_BM25_TEXT_INTERN"] = mode
    with tempfile.TemporaryDirectory(prefix="axon_bm25_mem_") as tmp_str:
        tmp = Path(tmp_str)
        r = BM25Retriever(storage_path=str(tmp), engine="python")
        r.corpus = []
        r._dirty = False

        gc.collect()
        t0 = time.perf_counter()
        r.add_documents(docs, save_deferred=True)
        dt = time.perf_counter() - t0
        text_objs = [d.get("text", "") for d in r.corpus if isinstance(d, dict)]
        unique_obj_ids = {id(t) for t in text_objs if isinstance(t, str)}
        # Map back ids to one representative object to estimate unique text storage.
        id_to_text = {}
        for t in text_objs:
            if isinstance(t, str):
                id_to_text.setdefault(id(t), t)
        unique_text_bytes = sum(sys.getsizeof(t) for t in id_to_text.values())
        logical_text_bytes = sum(sys.getsizeof(t) for t in text_objs if isinstance(t, str))
        return {
            "text_intern_mode": mode,
            "seconds_add_documents": dt,
            "corpus_docs": len(r.corpus),
            "unique_text_object_count": len(unique_obj_ids),
            "logical_text_bytes": int(logical_text_bytes),
            "unique_text_bytes_estimate": int(unique_text_bytes),
        }


def main() -> int:
    docs_dup_off = _make_docs(n_docs=30000, n_unique_texts=300)
    docs_dup_on = _make_docs(n_docs=30000, n_unique_texts=300)
    docs_dup_auto = _make_docs(n_docs=30000, n_unique_texts=300)
    docs_unique_off = _make_docs(n_docs=30000, n_unique_texts=30000)
    docs_unique_on = _make_docs(n_docs=30000, n_unique_texts=30000)
    docs_unique_auto = _make_docs(n_docs=30000, n_unique_texts=30000)
    dup_off = _run_case(mode="off", docs=docs_dup_off)
    dup_on = _run_case(mode="on", docs=docs_dup_on)
    dup_auto = _run_case(mode="auto", docs=docs_dup_auto)
    unique_off = _run_case(mode="off", docs=docs_unique_off)
    unique_on = _run_case(mode="on", docs=docs_unique_on)
    unique_auto = _run_case(mode="auto", docs=docs_unique_auto)
    result = {
        "duplicate_heavy": {
            "off": dup_off,
            "on": dup_on,
            "auto": dup_auto,
        },
        "unique_heavy": {
            "off": unique_off,
            "on": unique_on,
            "auto": unique_auto,
        },
        "summary": {
            "dataset_docs": len(docs_dup_on),
            "dup_unique_texts": len({d["text"] for d in docs_dup_on}),
            "unique_unique_texts": len({d["text"] for d in docs_unique_on}),
            "dup_on_unique_text_memory_reduction_x": dup_off["unique_text_bytes_estimate"]
            / max(dup_on["unique_text_bytes_estimate"], 1),
            "dup_auto_unique_text_memory_reduction_x": dup_off["unique_text_bytes_estimate"]
            / max(dup_auto["unique_text_bytes_estimate"], 1),
            "dup_auto_add_speedup_vs_on_x": dup_on["seconds_add_documents"]
            / max(dup_auto["seconds_add_documents"], 1e-12),
            "dup_auto_add_speedup_vs_off_x": dup_off["seconds_add_documents"]
            / max(dup_auto["seconds_add_documents"], 1e-12),
            "unique_auto_add_speedup_vs_on_x": unique_on["seconds_add_documents"]
            / max(unique_auto["seconds_add_documents"], 1e-12),
            "unique_auto_add_speedup_vs_off_x": unique_off["seconds_add_documents"]
            / max(unique_auto["seconds_add_documents"], 1e-12),
        },
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_bm25_memory_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
