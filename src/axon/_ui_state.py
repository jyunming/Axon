"""Lightweight shared state for REPL status-bar updates.

Written by any axon module (e.g. embeddings, ingestion); read by the REPL toolbar.
Kept as a plain dict to avoid import cycles.
"""

state: dict = {
    "embed_progress": "",  # Current tqdm embedding progress line, or ""
}
