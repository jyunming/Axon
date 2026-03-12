#!/usr/bin/env bash
# MCP server launcher for VS Code (WSL + Windows Python)
# Ensures PYTHONPATH is set and uses the correct Python interpreter.
set -e

PYTHON="/mnt/c/Users/jyunm/AppData/Local/Programs/Python/Python313/python.exe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
exec "$PYTHON" -m axon.mcp_server
