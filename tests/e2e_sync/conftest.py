"""Real-sync E2E test fixtures (Phase 7 Layer 2 — Nextcloud-in-Docker).

Spins up a Nextcloud server in a docker container, provisions two
users (``alice`` the owner + ``bob`` the grantee), and exposes pytest
fixtures that act as two grantee processes pushing/pulling via the
Nextcloud WebDAV endpoint. No real cloud account, no OAuth, no rate
limits — runs on any box with Docker.

Why this layer exists: filesystem chaos tests (``tests/sync/``) catch
~80% of cloud-sync failure modes via stat/open patching but cannot
exercise true two-writer races, real ETag-based change detection,
or eventual-consistency settle behaviour. Nextcloud's WebDAV
semantics are isomorphic to OneDrive at the file-level abstraction
Axon cares about — same conflict-copy naming, same lock contention,
same ETag mismatch detection.

Auto-skipped when:

- ``docker`` CLI is not on PATH;
- ``requests`` is not installed (we use it for OCS API calls);
- the docker daemon refuses ``docker compose up`` (no daemon, no
  permissions, etc.).

Setup is documented in ``docs/SHARE_MOUNT_SEALED_SMOKE.md``.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

_COMPOSE_FILE = Path(__file__).parent / "docker-compose.yml"
_NEXTCLOUD_BASE = "http://localhost:18080"
_OCS_API = f"{_NEXTCLOUD_BASE}/ocs/v2.php/cloud/users"
_OCS_HEADERS = {"OCS-APIRequest": "true", "Accept": "application/json"}


# ---------------------------------------------------------------------------
# Prerequisites probe — auto-skip when docker / requests missing
# ---------------------------------------------------------------------------


def _skip_reason() -> str | None:
    if shutil.which("docker") is None:
        return "Real-sync E2E needs docker on PATH (https://docs.docker.com/get-docker/)."
    try:
        import requests  # noqa: F401
    except ImportError:
        return (
            "Real-sync E2E needs the 'requests' package. "
            "Install with: pip install axon-rag[sealed-test]"
        )
    # Quick "is the daemon alive" probe — `docker info` exits non-zero
    # when the daemon isn't reachable. FileNotFoundError catches the
    # macOS GitHub Actions case where shutil.which finds a stub but
    # the actual binary isn't executable.
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return f"Docker daemon not reachable: {proc.stderr.strip()[:200]}"
    except FileNotFoundError as exc:
        return f"Docker binary not executable: {exc}"
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"Docker daemon probe failed: {exc}"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


# ---------------------------------------------------------------------------
# Compose-up / compose-down session fixture
# ---------------------------------------------------------------------------


def _compose(*args: str, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a ``docker compose`` subcommand against our compose file."""
    return subprocess.run(
        ["docker", "compose", "-f", str(_COMPOSE_FILE), *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _wait_for_nextcloud(timeout: float = 90.0) -> None:
    """Poll ``status.php`` until it returns 200 or *timeout* expires."""
    import requests

    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{_NEXTCLOUD_BASE}/status.php", timeout=3)
            if r.status_code == 200 and "installed" in r.text.lower():
                return
            last_err = f"HTTP {r.status_code}: {r.text[:120]}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(2)
    raise TimeoutError(
        f"Nextcloud at {_NEXTCLOUD_BASE} did not become healthy in "
        f"{timeout}s. Last probe: {last_err}"
    )


@pytest.fixture(scope="session")
def nextcloud_stack() -> Iterator[str]:
    """Bring up + tear down the Nextcloud container for the suite.

    Yields the base URL (``http://localhost:18080``). Skips the suite
    when ``docker compose up`` fails or when ``docker`` itself can't
    be invoked (FileNotFoundError = binary not actually executable
    even though shutil.which found it; CalledProcessError = daemon
    not running / image-pull failure; TimeoutExpired = network
    stall). Skips cleanly rather than crashing every test as an ERROR.
    """
    # Best-effort cleanup of any orphaned container from a prior run.
    # Suppress all subprocess errors here — this is a defensive cleanup
    # before the real "up" call below, which is where we surface skips.
    # ``subprocess.TimeoutExpired`` shows up routinely on Windows GitHub
    # runners because Docker Desktop's compose-down can take >30s on a
    # cold daemon; let it slide and let the real up call surface a skip
    # if the daemon is genuinely unusable.
    try:
        _compose("down", "-v", check=False, timeout=30)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    try:
        _compose("up", "-d", timeout=180)
    except FileNotFoundError as exc:
        pytest.skip(
            f"docker binary not actually executable on this runner "
            f"(shutil.which found it but exec failed): {exc}"
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(
            f"docker compose up failed (daemon installed but couldn't "
            f"start the stack): {exc.stderr.strip()[:200] if exc.stderr else exc}"
        )
    except subprocess.TimeoutExpired:
        pytest.skip(
            "docker compose up timed out — image pull may be stuck "
            "behind a slow / proxied network."
        )

    try:
        _wait_for_nextcloud()
        yield _NEXTCLOUD_BASE
    finally:
        try:
            _compose("down", "-v", check=False, timeout=60)
        except (FileNotFoundError, OSError):
            pass


# ---------------------------------------------------------------------------
# OCS user-provisioning helpers (Nextcloud's REST API)
# ---------------------------------------------------------------------------


def _create_user(username: str, password: str) -> None:
    """Create *username* with *password* via the OCS API. Idempotent.

    Nextcloud OCS quirks observed in CI:
    - First-time success returns HTTP 200 with ``statuscode: 200``.
    - "User already exists" returns HTTP 400 with ``statuscode: 102``.
    - Older OCS clients also see ``statuscode: 100`` for success.

    Treat all three of {100, 102, 200} as success regardless of the
    outer HTTP status — what matters is the user exists when we
    return.
    """
    import requests

    resp = requests.post(
        _OCS_API,
        auth=("admin", "admin"),
        headers=_OCS_HEADERS,
        data={"userid": username, "password": password},
        timeout=10,
    )
    try:
        ocs_status = resp.json()["ocs"]["meta"]["statuscode"]
    except (KeyError, ValueError):
        ocs_status = None
    if ocs_status in (100, 102, 200):
        return
    raise RuntimeError(
        f"OCS create_user({username!r}) failed: HTTP {resp.status_code} "
        f"ocs_statuscode={ocs_status} body={resp.text[:200]}"
    )


def _delete_user(username: str) -> None:
    """Delete *username*. Idempotent."""
    import requests

    try:
        requests.delete(
            f"{_OCS_API}/{username}",
            auth=("admin", "admin"),
            headers=_OCS_HEADERS,
            timeout=10,
        )
    except requests.RequestException:
        pass  # cleanup is best-effort


# ---------------------------------------------------------------------------
# WebDAV upload/download helpers (one per "machine")
# ---------------------------------------------------------------------------


def _webdav_url(username: str, path: str) -> str:
    """``http://localhost:18080/remote.php/dav/files/<user>/<path>``."""
    path = path.lstrip("/")
    return f"{_NEXTCLOUD_BASE}/remote.php/dav/files/{username}/{path}"


class WebDavClient:
    """Thin WebDAV client — upload/download/delete + read ETag.

    Each instance represents one "machine" in the test; tests that
    exercise two-writer races create two ``WebDavClient`` instances
    pointed at the same path with the same credentials.
    """

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def upload(self, remote_path: str, contents: bytes, *, if_match: str | None = None) -> str:
        """PUT *contents* to *remote_path*; return the new ETag."""
        import requests

        headers: dict[str, str] = {}
        if if_match is not None:
            headers["If-Match"] = if_match
        resp = requests.put(
            _webdav_url(self.username, remote_path),
            auth=(self.username, self.password),
            data=contents,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.headers.get("ETag", "")

    def download(self, remote_path: str) -> tuple[bytes, str]:
        """GET *remote_path*; return ``(bytes, etag)``."""
        import requests

        resp = requests.get(
            _webdav_url(self.username, remote_path),
            auth=(self.username, self.password),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.content, resp.headers.get("ETag", "")

    def list_dir(self, remote_path: str) -> list[str]:
        """PROPFIND on *remote_path* — return child names (basename only)."""
        import requests

        resp = requests.request(
            "PROPFIND",
            _webdav_url(self.username, remote_path),
            auth=(self.username, self.password),
            headers={"Depth": "1"},
            timeout=15,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        # Parse the multistatus XML for <d:href> entries — strip path
        # prefix and skip the directory itself.
        import re

        hrefs = re.findall(r"<d:href>([^<]+)</d:href>", resp.text)
        prefix = f"/remote.php/dav/files/{self.username}/{remote_path.strip('/')}/"
        return [
            h.replace(prefix, "").rstrip("/") for h in hrefs if h.startswith(prefix) and h != prefix
        ]

    def mkdir(self, remote_path: str) -> None:
        """MKCOL — idempotent (405 = already exists is treated as success)."""
        import requests

        resp = requests.request(
            "MKCOL",
            _webdav_url(self.username, remote_path),
            auth=(self.username, self.password),
            timeout=15,
        )
        if resp.status_code in (201, 405):
            return
        resp.raise_for_status()

    def delete(self, remote_path: str) -> None:
        """DELETE — idempotent (404 swallowed)."""
        import requests

        try:
            requests.delete(
                _webdav_url(self.username, remote_path),
                auth=(self.username, self.password),
                timeout=15,
            )
        except requests.RequestException:
            pass


# ---------------------------------------------------------------------------
# Per-test users + clients
# ---------------------------------------------------------------------------


@pytest.fixture
def two_users(nextcloud_stack):
    """Create alice + bob; yield ``(alice_client, bob_client)``."""
    _create_user("alice", "alice-pw-12345")
    _create_user("bob", "bob-pw-12345")
    try:
        alice = WebDavClient("alice", "alice-pw-12345")
        bob = WebDavClient("bob", "bob-pw-12345")
        yield alice, bob
    finally:
        _delete_user("alice")
        _delete_user("bob")
