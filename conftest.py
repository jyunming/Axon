"""
Root conftest.py — makes the src/ tree importable without a pip install.
"""
import sys
import os

# Add src/ so that `import rag_brain` works in tests without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
