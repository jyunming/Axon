"""Unit tests for axon.paths — share-mount / cloud-sync path classification."""
from __future__ import annotations

from pathlib import Path

import pytest

from axon.paths import (
    cloud_sync_path_reason,
    is_cloud_sync_or_mount_path,
    is_cloud_sync_path,
    is_unc_path,
    is_wsl_windows_mount_path,
    safe_local_path,
)


class TestIsCloudSyncPath:
    @pytest.mark.parametrize(
        "p",
        [
            r"C:\Users\alice\OneDrive\Axon\project",
            r"C:\Users\alice\OneDrive - Contoso\docs",
            r"C:\Users\alice\Dropbox\Projects\axon",
            r"C:\Users\alice\Google Drive\My Drive\data",
            r"G:\My Drive\Axon\project",
            r"C:\Users\alice\iCloud Drive\axon",
            "/home/alice/Dropbox/axon",
            "/Users/alice/OneDrive/work",
            "/Users/alice/OneDrive - Personal/work",
        ],
    )
    def test_cloud_sync_paths_detected(self, p):
        assert is_cloud_sync_path(p)

    @pytest.mark.parametrize(
        "p",
        [
            r"C:\Users\alice\Documents\axon",
            r"D:\projects\axon",
            "/home/alice/.axon/projects/default",
            "/tmp/axon-tests",
            "",
            None,
            "/srv/data/axon",
        ],
    )
    def test_non_cloud_sync_paths(self, p):
        assert not is_cloud_sync_path(p)

    def test_case_insensitive(self):
        assert is_cloud_sync_path(r"C:\Users\alice\onedrive\foo")
        assert is_cloud_sync_path(r"C:\Users\alice\ONEDRIVE\foo")
        assert is_cloud_sync_path(r"C:\Users\alice\DROPBOX\foo")

    def test_pathlib_input(self):
        assert is_cloud_sync_path(Path(r"C:\Users\alice\OneDrive\foo"))

    def test_unrelated_segment_with_onedrive_substring(self):
        # "OneDriver" should NOT match — only exact "OneDrive" or the
        # "OneDrive - " business prefix.
        assert not is_cloud_sync_path(r"C:\Users\alice\OneDriver\foo")
        assert not is_cloud_sync_path(r"C:\Users\alice\NotDropbox\foo")


class TestIsUncPath:
    @pytest.mark.parametrize(
        "p",
        [
            r"\\fileserver\share\axon",
            "//fileserver/share/axon",
        ],
    )
    def test_unc_detected(self, p):
        assert is_unc_path(p)

    @pytest.mark.parametrize(
        "p",
        [
            r"C:\Users\alice\axon",
            "/home/alice/axon",
            "",
            None,
        ],
    )
    def test_non_unc(self, p):
        assert not is_unc_path(p)


class TestIsWslWindowsMountPath:
    @pytest.mark.parametrize(
        "p",
        [
            "/mnt/c/Users/alice/axon",
            "/mnt/d/projects",
            "//wsl$/Ubuntu/home/alice",
            "//wsl.localhost/Ubuntu/home/alice",
        ],
    )
    def test_wsl_mount_detected(self, p):
        assert is_wsl_windows_mount_path(p)

    @pytest.mark.parametrize(
        "p",
        [
            "/mnt",  # too short — not an actual mount root
            "/mnta/foo",  # not /mnt/
            "/home/alice/axon",
            r"C:\Users\alice\axon",
            "",
            None,
        ],
    )
    def test_not_wsl_mount(self, p):
        assert not is_wsl_windows_mount_path(p)


class TestIsCloudSyncOrMountPath:
    def test_unions_all_three(self):
        assert is_cloud_sync_or_mount_path(r"C:\Users\alice\OneDrive\foo")
        assert is_cloud_sync_or_mount_path(r"\\server\share\axon")
        assert is_cloud_sync_or_mount_path("/mnt/c/axon")

    def test_safe_paths(self):
        assert not is_cloud_sync_or_mount_path(r"C:\Users\alice\Documents\axon")
        assert not is_cloud_sync_or_mount_path("/home/alice/.axon")
        assert not is_cloud_sync_or_mount_path("")
        assert not is_cloud_sync_or_mount_path(None)


class TestCloudSyncPathReason:
    def test_onedrive_reason_mentions_cloud_sync(self):
        r = cloud_sync_path_reason(r"C:\Users\alice\OneDrive\axon")
        assert "cloud-sync" in r.lower()

    def test_unc_reason_mentions_unc(self):
        r = cloud_sync_path_reason(r"\\server\share\axon")
        assert "unc" in r.lower() or "network share" in r.lower()

    def test_wsl_reason_mentions_wsl(self):
        r = cloud_sync_path_reason("/mnt/c/foo")
        assert "wsl" in r.lower()

    def test_safe_path_returns_empty(self):
        assert cloud_sync_path_reason("/home/alice/axon") == ""
        assert cloud_sync_path_reason("") == ""
        assert cloud_sync_path_reason(None) == ""


class TestSafeLocalPath:
    def test_safe_path_returned_unchanged(self, tmp_path):
        p = tmp_path / "foo.db"
        assert safe_local_path(p) == p

    def test_unsafe_path_redirected_to_axon_home(self):
        p = Path(r"C:\Users\alice\OneDrive\axon\foo.db")
        out = safe_local_path(p)
        assert out.name == "foo.db"
        # Should land under ~/.axon/
        assert ".axon" in out.parts
