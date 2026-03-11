"""
Root conftest.py — makes the src/ tree importable without a pip install.
"""
import os
import sys

# Add src/ so that `import axon` works in tests without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
