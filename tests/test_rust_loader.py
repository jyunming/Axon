from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from axon import _rust_loader


def test_preferred_dev_artifact_skips_when_missing(tmp_path):
    package_dir = tmp_path / "axon"
    package_dir.mkdir()
    (package_dir / "axon_rust.cp311-win_amd64.pyd").write_bytes(b"stale")

    assert _rust_loader._preferred_dev_artifact(package_dir) is None


def test_preferred_dev_artifact_uses_newer_release_build(tmp_path):
    package_dir = tmp_path / "axon"
    target_dir = package_dir / "target" / "release"
    target_dir.mkdir(parents=True)

    bundled = package_dir / "axon_rust.cp311-win_amd64.pyd"
    bundled.write_bytes(b"bundled")
    dev_name = _rust_loader._platform_dev_artifact_name()
    dev = target_dir / dev_name
    dev.write_bytes(b"dev")
    dev.touch()

    assert _rust_loader._preferred_dev_artifact(package_dir) == dev


def test_bootstrap_dev_rust_module_registers_loaded_module(tmp_path, monkeypatch):
    package_dir = tmp_path / "axon"
    target_dir = package_dir / "target" / "release"
    target_dir.mkdir(parents=True)
    dev_name = _rust_loader._platform_dev_artifact_name()
    (target_dir / dev_name).write_bytes(b"dev")

    loaded: dict[str, Path] = {}
    module_name = "axon_test_pkg.axon_rust"

    def _fake_load(name: str, artifact_path: Path) -> ModuleType:
        loaded["name"] = name
        loaded["path"] = artifact_path
        module = ModuleType(name)
        sys.modules[name] = module
        return module

    monkeypatch.setattr(_rust_loader, "_load_extension_module", _fake_load)
    sys.modules.pop(module_name, None)
    try:
        assert _rust_loader.bootstrap_dev_rust_module("axon_test_pkg", package_dir) is True
        assert loaded["name"] == module_name
        assert loaded["path"] == target_dir / dev_name
        assert module_name in sys.modules
    finally:
        sys.modules.pop(module_name, None)
