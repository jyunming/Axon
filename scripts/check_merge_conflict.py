"""Check staged files for unresolved merge conflict markers.

Replacement for the pre-commit-hooks check-merge-conflict hook that runs as a

local hook using the system Python, avoiding AppLocker / WDAC policy blocks that

occur when pre-commit tries to execute binaries from its managed venv cache.

"""

from __future__ import annotations

import sys

# Standard git conflict marker prefixes

_MARKERS = ("<<<<<<< ", "=======", ">>>>>>> ")


def _has_conflict(path: str) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(_MARKERS):
                    return True

    except OSError:
        pass

    return False


def main() -> int:
    bad = [p for p in sys.argv[1:] if _has_conflict(p)]

    for path in bad:
        print(f"ERROR Merge conflict markers found in {path}", file=sys.stderr)

    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
