#!/usr/bin/env python3

"""

Pre-fetch HuggingFace models for offline / air-gapped deployment.

Run this ONCE on a machine with internet access, then copy the output

directory to your confined workspace and set in config.yaml:

    offline:

      enabled: true

      local_models_dir: /absolute/path/to/models

Usage

-----

  python scripts/prefetch_models.py

  python scripts/prefetch_models.py --dir C:/models

  python scripts/prefetch_models.py --dir /srv/models --extra BAAI/bge-large-zh-v15

Default models downloaded

-------------------------

  sentence-transformers/all-MiniLM-L6-v2   (embedding, ~90 MB)

  BAAI/bge-reranker-base                   (reranker, ~270 MB)

Ollama models (gemma3:27b, gpt-oss:120b, etc.)

----------------------------------------------

Ollama manages its own model store.  See scripts/README.md for instructions

on copying Ollama models to an offline machine.

"""

import argparse
import os
import sys

DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-reranker-base",
]


def _human_size(total_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if total_bytes < 1024:
            return f"{total_bytes:.1f} {unit}"

        total_bytes /= 1024

    return f"{total_bytes:.1f} TB"


def download_model(repo_id: str, output_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download

    except ImportError:
        print("  ❌ huggingface_hub is not installed.  Run: pip install huggingface_hub")

        sys.exit(1)

    # Store as <dir>/<short-name> so config.yaml looks clean

    short_name = repo_id.split("/")[-1]

    local_dir = os.path.join(output_dir, short_name)

    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"  ✔  Already exists, skipping: {local_dir}")

        return

    print(f"  ⬇  {repo_id}  →  {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # copy files, not symlinks — safe for transfer
    )

    # Report size

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(local_dir)
        for f in files
    )

    file_count = sum(len(files) for _, _, files in os.walk(local_dir))

    print(f"     {file_count} files, {_human_size(total)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch HuggingFace models for offline use.")

    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
        help="Destination directory (default: <project>/models)",
    )

    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="REPO_ID",
        help="Additional HuggingFace repo IDs to download",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.dir)

    os.makedirs(output_dir, exist_ok=True)

    models = DEFAULT_MODELS + (args.extra or [])

    print(f"\nDestination : {output_dir}")

    print(f"Models      : {len(models)}\n")

    for repo_id in models:
        download_model(repo_id, output_dir)

    print("\n✅ Done.  Set in config.yaml:\n")

    print("    offline:")

    print("      enabled: true")

    print(f"      local_models_dir: {output_dir}")

    print()

    print("    embedding:")

    print("      model: all-MiniLM-L6-v2")

    print()

    print("    rerank:")

    print("      model: bge-reranker-base")

    print()


if __name__ == "__main__":
    main()
