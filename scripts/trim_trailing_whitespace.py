"""Trim trailing whitespace from text files — local pre-commit hook replacement.

Uses system Python directly so Windows Application Control policies that block
pre-commit-hooks venv executables do not prevent the hook from running.

"""

import sys
from pathlib import Path

fixed = 0

for path_str in sys.argv[1:]:
    p = Path(path_str)

    try:
        original = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue  # skip binary or unreadable files

    lines = original.splitlines(keepends=True)

    new_lines = [
        line.rstrip(" \t\r\n")
        + ("\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else "")
        for line in lines
    ]

    updated = "".join(new_lines)

    if updated != original:
        p.write_text(updated, encoding="utf-8")
        fixed += 1

if fixed:
    print(f"Trimmed trailing whitespace in {fixed} file(s).")
    sys.exit(1)  # signal pre-commit that files were modified
