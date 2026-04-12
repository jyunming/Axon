from __future__ import annotations

import getpass
from pathlib import Path
from unittest.mock import patch

import pytest

import axon.api as api_module

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_project_create_switch_isolation_and_parent_fanout(api_client, make_brain):
    make_brain()

    api_client.post(
        "/add_text",
        json={
            "text": "Default project contains deployment guidance for Axon.",
            "metadata": {"source": "default.txt"},
            "doc_id": "default_doc",
        },
    )

    create_parent = api_client.post("/project/new", json={"name": "research"})

    create_child = api_client.post("/project/new", json={"name": "research/papers"})

    assert create_parent.status_code == 200

    assert create_child.status_code == 200

    switch_child = api_client.post("/project/switch", json={"name": "research/papers"})

    assert switch_child.status_code == 200

    assert switch_child.json()["active_project"] == "research/papers"

    empty_collection = api_client.get("/collection")

    assert empty_collection.status_code == 200

    assert empty_collection.json()["total_chunks"] == 0

    api_client.post(
        "/add_text",
        json={
            "text": "Research papers discuss LightRAG and GraphRAG tradeoffs.",
            "metadata": {"source": "papers.txt"},
            "doc_id": "papers_doc",
        },
    )

    switch_parent = api_client.post("/project/switch", json={"name": "research"})

    assert switch_parent.status_code == 200

    query_parent = api_client.post(
        "/query",
        json={"query": "What do the papers discuss?"},
    )

    assert query_parent.status_code == 200

    assert "LightRAG" in query_parent.json()["response"]

    switch_default = api_client.post("/project/switch", json={"name": "default"})

    assert switch_default.status_code == 200

    query_default = api_client.post("/query", json={"query": "What guidance exists in default?"})

    assert query_default.status_code == 200

    assert "deployment guidance" in query_default.json()["response"]

    projects = api_client.get("/projects")

    assert projects.status_code == 200

    project_names = {entry["name"] for entry in projects.json()["projects"]}

    assert "default" in project_names

    assert "research" in project_names


def test_store_init_whoami_and_share_lifecycle(api_client, make_brain, tmp_path):
    make_brain()

    base_path = tmp_path / "axon_store_base"

    # Patch config.save so /store/init does not write to the real ~/.config/axon/config.yaml

    with patch("axon.main.AxonConfig.save"):
        store_init = api_client.post("/store/init", json={"base_path": str(base_path)})

    assert store_init.status_code == 200

    init_payload = store_init.json()

    user_dir = Path(init_payload["user_dir"])

    assert (user_dir / "default").exists()

    assert (user_dir / "projects").exists()

    assert (user_dir / "mounts").exists()

    assert (user_dir / ".shares").exists()

    assert api_module.brain.config.vector_store_path.endswith(
        "default/vector_data"
    ) or api_module.brain.config.vector_store_path.endswith("default\\vector_data")

    whoami = api_client.get("/store/whoami")

    assert whoami.status_code == 200

    whoami_payload = whoami.json()

    assert whoami_payload["username"] == getpass.getuser()

    assert whoami_payload["active_project"] == "default"

    create_project = api_client.post("/project/new", json={"name": "sharedproj"})

    assert create_project.status_code == 200

    share_generate = api_client.post(
        "/share/generate",
        json={
            "project": "sharedproj",
            "grantee": getpass.getuser(),
        },
    )

    assert share_generate.status_code == 200

    share_payload = share_generate.json()

    assert share_payload["project"] == "sharedproj"

    assert share_payload["share_string"]

    shares = api_client.get("/share/list")

    assert shares.status_code == 200

    shares_payload = shares.json()

    assert len(shares_payload["sharing"]) == 1

    assert shares_payload["sharing"][0]["project"] == "sharedproj"

    # Redeem is now platform-independent (descriptor model); succeeds on all platforms

    redeem = api_client.post("/share/redeem", json={"share_string": share_payload["share_string"]})

    assert redeem.status_code == 200

    redeem_payload = redeem.json()

    assert redeem_payload["mount_name"] == f"{getpass.getuser()}_sharedproj"

    assert redeem_payload["owner"] == getpass.getuser()

    assert "descriptor" in redeem_payload

    revoke = api_client.post("/share/revoke", json={"key_id": share_payload["key_id"]})

    assert revoke.status_code == 200

    assert revoke.json()["key_id"] == share_payload["key_id"]

    shares_after_revoke = api_client.get("/share/list")

    assert shares_after_revoke.status_code == 200

    assert shares_after_revoke.json()["sharing"][0]["revoked"] is True
