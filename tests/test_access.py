"""tests/test_access.py — Unit tests for axon.access and axon.maintenance."""

import json

import pytest

from axon.access import check_write_allowed, is_mounted_share_path

# ---------------------------------------------------------------------------
# is_mounted_share_path
# ---------------------------------------------------------------------------


class TestIsMountedSharePath:
    def test_mounts_prefix_is_mounted(self):
        assert is_mounted_share_path("mounts/alice_proj") is True

    def test_mounts_nested_is_mounted(self):
        assert is_mounted_share_path("mounts/alice_research") is True

    def test_default_project_is_not_mounted(self):
        assert is_mounted_share_path("default") is False

    def test_regular_project_is_not_mounted(self):
        assert is_mounted_share_path("myproject") is False

    def test_mounts_not_first_segment_is_not_mounted(self):
        # Only the first path segment is checked
        assert is_mounted_share_path("projects/mounts/foo") is False


# ---------------------------------------------------------------------------
# check_write_allowed
# ---------------------------------------------------------------------------


class TestCheckWriteAllowed:
    def test_passes_for_normal_project(self, tmp_path, monkeypatch):
        import axon.projects as _proj

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "meta.json").write_text(
            json.dumps({"name": "myproj", "maintenance_state": "normal"}), encoding="utf-8"
        )
        # must not raise
        check_write_allowed("ingest", "myproj", read_only_scope=False, is_mounted=False)

    def test_raises_for_read_only_scope(self):
        with pytest.raises(PermissionError, match="read-only"):
            check_write_allowed("ingest", "myproj", read_only_scope=True, is_mounted=False)

    def test_raises_for_mounted_share(self):
        with pytest.raises(PermissionError, match="mounted share"):
            check_write_allowed(
                "ingest", "mounts/alice_proj", read_only_scope=False, is_mounted=True
            )

    def test_raises_for_readonly_maintenance(self, tmp_path, monkeypatch):
        import axon.projects as _proj

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "meta.json").write_text(
            json.dumps({"name": "myproj", "maintenance_state": "readonly"}), encoding="utf-8"
        )
        with pytest.raises(PermissionError, match="readonly.*maintenance"):
            check_write_allowed("ingest", "myproj", read_only_scope=False, is_mounted=False)

    def test_raises_for_offline_maintenance(self, tmp_path, monkeypatch):
        import axon.projects as _proj

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "meta.json").write_text(
            json.dumps({"name": "myproj", "maintenance_state": "offline"}), encoding="utf-8"
        )
        with pytest.raises(PermissionError, match="offline.*maintenance"):
            check_write_allowed("ingest", "myproj", read_only_scope=False, is_mounted=False)

    def test_read_only_scope_takes_precedence_over_mounted(self):
        """read_only_scope is checked first — error message reflects scope, not mount."""
        with pytest.raises(PermissionError, match="read-only"):
            check_write_allowed("ingest", "mounts/proj", read_only_scope=True, is_mounted=True)

    def test_default_project_skips_maintenance_check(self):
        """'default' never has a maintenance state — must not raise."""
        check_write_allowed("ingest", "default", read_only_scope=False, is_mounted=False)


# ---------------------------------------------------------------------------
# maintenance module
# ---------------------------------------------------------------------------


class TestApplyMaintenanceState:
    def test_set_readonly_returns_correct_fields(self, tmp_path, monkeypatch):
        import axon.projects as _proj
        from axon.maintenance import apply_maintenance_state

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "meta.json").write_text(json.dumps({"name": "myproj"}), encoding="utf-8")
        result = apply_maintenance_state("myproj", "readonly")
        assert result["status"] == "ok"
        assert result["project"] == "myproj"
        assert result["maintenance_state"] == "readonly"
        assert "active_leases" in result
        assert "epoch" in result

    def test_set_draining_starts_registry_drain(self, tmp_path, monkeypatch):
        import axon.projects as _proj
        from axon.maintenance import apply_maintenance_state
        from axon.runtime import get_registry

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "drainproj"
        proj.mkdir()
        (proj / "meta.json").write_text(json.dumps({"name": "drainproj"}), encoding="utf-8")
        reg = get_registry()
        reg.reset("drainproj")
        apply_maintenance_state("drainproj", "draining")
        assert reg.snapshot("drainproj")["draining"] is True
        reg.stop_drain("drainproj")
        reg.reset("drainproj")

    def test_set_normal_stops_drain(self, tmp_path, monkeypatch):
        import axon.projects as _proj
        from axon.maintenance import apply_maintenance_state
        from axon.runtime import get_registry

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "normalproj"
        proj.mkdir()
        (proj / "meta.json").write_text(json.dumps({"name": "normalproj"}), encoding="utf-8")
        reg = get_registry()
        reg.start_drain("normalproj")
        apply_maintenance_state("normalproj", "normal")
        assert reg.snapshot("normalproj")["draining"] is False
        reg.reset("normalproj")

    def test_invalid_state_raises_value_error(self, tmp_path, monkeypatch):
        import axon.projects as _proj
        from axon.maintenance import apply_maintenance_state

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        with pytest.raises(ValueError):
            apply_maintenance_state("ghost", "broken")


class TestGetMaintenanceStatus:
    def test_returns_expected_fields(self, tmp_path, monkeypatch):
        import axon.projects as _proj
        from axon.maintenance import get_maintenance_status

        monkeypatch.setattr(_proj, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "statproj"
        proj.mkdir()
        (proj / "meta.json").write_text(
            json.dumps({"name": "statproj", "maintenance_state": "draining"}), encoding="utf-8"
        )
        result = get_maintenance_status("statproj")
        assert result["project"] == "statproj"
        assert result["maintenance_state"] == "draining"
        assert "active_leases" in result
        assert "epoch" in result
        assert "draining" in result
