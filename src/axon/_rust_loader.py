from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

logger = logging.getLogger("Axon.RustLoader")

_DEV_ARTIFACT_NAMES = (
    "axon_rust.dll",
    "libaxon_rust.so",
    "libaxon_rust.dylib",
)


def _is_extension_artifact(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _bundled_extension_artifacts(package_dir: Path) -> list[Path]:
    try:
        children = list(package_dir.iterdir())
    except OSError:
        return []
    return sorted(
        path
        for path in children
        if path.is_file() and path.name.startswith("axon_rust") and _is_extension_artifact(path)
    )


def _preferred_dev_artifact(package_dir: Path) -> Path | None:
    target_dir = package_dir / "target" / "release"
    dev_candidates = [target_dir / name for name in _DEV_ARTIFACT_NAMES]
    dev_existing = [path for path in dev_candidates if path.is_file()]
    if not dev_existing:
        return None
    newest_dev = max(dev_existing, key=lambda item: item.stat().st_mtime)

    bundled = _bundled_extension_artifacts(package_dir)
    if not bundled:
        return newest_dev

    newest_bundled_mtime = max(path.stat().st_mtime for path in bundled)
    if newest_dev.stat().st_mtime >= newest_bundled_mtime:
        return newest_dev
    return None


def _load_extension_module(module_name: str, artifact_path: Path) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(artifact_path))
    spec = importlib.util.spec_from_file_location(module_name, str(artifact_path), loader=loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {artifact_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_loaded_module(module: ModuleType) -> ModuleType:
    """Apply narrow Python-side shims for native API drift."""
    native_run_louvain = getattr(module, "run_louvain", None)
    if callable(native_run_louvain):
        try:
            setattr(module, "_native_run_louvain", native_run_louvain)
        except Exception:
            pass

        def _run_louvain(nodes, edges, resolution: float = 1.0):
            return native_run_louvain(nodes, edges, resolution)

        try:
            _run_louvain.__name__ = native_run_louvain.__name__
            _run_louvain.__doc__ = native_run_louvain.__doc__
        except Exception:
            pass
        setattr(module, "run_louvain", _run_louvain)
    return module


def bootstrap_dev_rust_module(package_name: str, package_dir: Path) -> bool:
    """Prefer a fresh cargo-built Rust artifact over a stale bundled extension.

    In editable/source-tree workflows the checked-in ``axon_rust*.pyd`` may lag
    behind ``target/release`` after new Rust functions are added. When a newer
    cargo build artifact exists, preload it into ``sys.modules`` so subsequent
    ``import axon.axon_rust`` resolves to the current native code.
    """

    module_name = f"{package_name}.axon_rust"
    if module_name in sys.modules:
        return False

    artifact = _preferred_dev_artifact(package_dir)
    if artifact is None:
        return False

    try:
        module = _load_extension_module(module_name, artifact)
        _patch_loaded_module(module)
        logger.info("Loaded development Rust artifact for %s from %s", module_name, artifact)
        return True
    except Exception:
        sys.modules.pop(module_name, None)
        logger.debug("Could not preload development Rust artifact from %s", artifact, exc_info=True)
        return False
