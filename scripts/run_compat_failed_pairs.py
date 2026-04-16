from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:/dev/studio_brain_open")
WT_V010 = REPO / "_wt_v010"
WT_CUR = REPO / "_wt_rust"
SCRIPTS_DIR = WT_CUR / "scripts"

CASES = [
    "bench_rust_vs_python_bm25.py",
    "bench_graph_rag_perf.py",
    "bench_graphrag_cost_profile.py",
    "bench_bm25_storage_profile.py",
    "bench_bm25_python_topk_profile.py",
    "bench_bm25_dedup_profile.py",
    "bench_bm25_memory_profile.py",
    "bench_graphrag_incoming_count_cache_profile.py",
]

SHIM = r"""
import inspect, importlib, os, runpy, sys
script = sys.argv[1]
sys.argv = [script]
retr = importlib.import_module("axon.retrievers")
orig = retr.BM25Retriever.__init__
sig = inspect.signature(orig)
def patched_bm25(self,*a,**kw):
    if os.getenv("AXON_COMPAT_FORCE_PY_BM25", "0") in {"1", "true", "yes", "on"}:
        if kw.get("engine") == "rust":
            kw["engine"] = "python"
            kw["rust_fallback_enabled"] = True
    kw = {k:v for k,v in kw.items() if k in sig.parameters}
    return orig(self,*a,**kw)
retr.BM25Retriever.__init__ = patched_bm25

cfg = importlib.import_module("axon.config")
origc = cfg.AxonConfig.__init__
sigc = inspect.signature(origc)
def patched_cfg(self,*a,**kw):
    kw = {k:v for k,v in kw.items() if k in sigc.parameters}
    return origc(self,*a,**kw)
cfg.AxonConfig.__init__ = patched_cfg

runpy.run_path(script, run_name="__main__")
"""


def try_extract_json(stdout: str):
    txt = stdout.strip()
    if not txt:
        return None
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = txt[start : end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        return None


def run_one(side: str, script_name: str, log_dir: Path) -> dict:
    script = SCRIPTS_DIR / script_name
    env = os.environ.copy()
    env["PYTHONPATH"] = str((WT_V010 if side == "v010" else WT_CUR) / "src")
    env["AXON_COMPAT_FORCE_PY_BM25"] = "1" if script_name == "bench_rust_vs_python_bm25.py" else "0"

    cmd = [sys.executable, "-c", SHIM, str(script)]

    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0

    out = p.stdout or ""
    err = p.stderr or ""
    data = try_extract_json(out)

    side_log = log_dir / f"{script_name[:-3]}__{side}.log"
    side_json = log_dir / f"{script_name[:-3]}__{side}.json"
    side_log.write_text(out + ("\n[stderr]\n" + err if err else ""), encoding="utf-8")
    if data is not None:
        side_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {
        "side": side,
        "exit_code": p.returncode,
        "seconds": round(dt, 3),
        "stdout_log": str(side_log),
        "json_artifact": str(side_json) if data is not None else None,
        "summary": data.get("summary") if isinstance(data, dict) else None,
        "stderr_tail": err.strip()[-500:] if err else "",
    }


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO / f"bench_logs_compat_failed_pairs_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for script in CASES:
        b = run_one("v010", script, log_dir)
        c = run_one("current", script, log_dir)
        pair = {"script": script, "baseline": b, "current": c}
        pairs.append(pair)

        bstat = "PASS" if b["exit_code"] == 0 else "FAIL"
        cstat = "PASS" if c["exit_code"] == 0 else "FAIL"
        print(f"[{script}] baseline={bstat} ({b['seconds']}s) | current={cstat} ({c['seconds']}s)")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_dir": str(log_dir),
        "total_scripts": len(CASES),
        "baseline_success": sum(1 for p in pairs if p["baseline"]["exit_code"] == 0),
        "current_success": sum(1 for p in pairs if p["current"]["exit_code"] == 0),
    }

    out = {"summary": summary, "pairs": pairs}
    out_file = REPO / "bench_results_compat_failed_pairs_v010_vs_current.json"
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("SUMMARY_FILE:", out_file)
    print("LOG_DIR:", log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
