# Axon demo — run with: vhs docs/assets/demo.tape
#
# Prerequisites:
#   - vhs installed (https://github.com/charmbracelet/vhs)
#   - Ollama running with a model pulled (e.g. ollama pull llama3.1:8b)
#   - In repo root: pip install -e .
#
# Output: docs/assets/demo.gif
# Typical run time: ~90 seconds (LLM-dependent)

Output docs/assets/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 110
Set Height 32
Set Theme "Catppuccin Mocha"
Set PlaybackSpeed 1.0
Set TypingSpeed 60ms
Set CursorBlink false

# ── Start Axon ────────────────────────────────────────────────────────────────
Sleep 500ms
Type "axon"
Enter
# Wait for the REPL to initialize and show the welcome header
Sleep 5s

# ── Ingest sample documents ───────────────────────────────────────────────────
Type "/ingest ./sample_docs/"
Enter
Sleep 6s

# ── First query ───────────────────────────────────────────────────────────────
Type "What topics are covered in these documents?"
Enter
Sleep 20s

# ── Show current settings ─────────────────────────────────────────────────────
Type "/context"
Enter
Sleep 2s

# ── Second query — project-specific ───────────────────────────────────────────
Type "Describe the system architecture"
Enter
Sleep 20s

# ── Exit ─────────────────────────────────────────────────────────────────────
Type "/quit"
Enter
Sleep 1s
