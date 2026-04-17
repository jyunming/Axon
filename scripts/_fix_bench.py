"""One-shot helper: truncate bench_stores.py at the first guard block."""
from pathlib import Path

p = Path(__file__).parent / "bench_stores.py"
content = p.read_text(encoding="utf-8")

marker = '\nif __name__ == "__main__":\n    main()\n'
idx = content.find(marker)
if idx == -1:
    print("marker not found!")
else:
    trimmed = content[: idx + len(marker)]
    p.write_text(trimmed, encoding="utf-8")
    print(f"OK — kept {len(trimmed.splitlines())} lines (cut at char {idx})")
