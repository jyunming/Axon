"""Every surface that writes config must be able to reach every config field.

`AxonConfig.save()/load()` round-tripping correctly is necessary but not
sufficient — a knob is only usable if some surface can set it. The write paths:

* REST ``POST /config/set``  — dot alias or field name, persists via save()
* REST ``POST /config/update`` — curated subset for live RAG tuning
* REPL ``/config set``       — shares the resolver with /config/set
* VS Code LM tool            — calls /config/set per key
* Web GUI settings panel     — calls the REST routes

Before ``resolve_config_key`` existed, ``_DOT_TO_FLAT`` covered 101 of 240
fields and /config/set returned 400 for the rest, so `graph_rag_depth`,
`chunk_size`, `llm_temperature` and 136 others could only be changed by editing
config.yaml by hand — on every surface.
"""

from dataclasses import fields

import pytest

from axon.api_routes.config_routes import _DOT_TO_FLAT, resolve_config_key
from axon.api_schemas import ConfigUpdateRequest
from axon.config import AxonConfig


def _settable_fields():
    return [f.name for f in fields(AxonConfig) if not f.name.startswith("_")]


class TestResolveConfigKey:
    def test_curated_alias_still_wins(self):
        assert resolve_config_key("chunk.strategy") == "chunk_strategy"
        assert resolve_config_key("llm.base_url") == "ollama_base_url"

    def test_bare_field_name_is_accepted(self):
        assert resolve_config_key("graph_rag_depth") == "graph_rag_depth"

    def test_dotted_tail_is_accepted(self):
        """`rag.graph_rag_depth` previously became `rag_graph_rag_depth`."""
        assert resolve_config_key("rag.graph_rag_depth") == "graph_rag_depth"

    def test_unknown_key_returns_none(self):
        assert resolve_config_key("not_a_real_field") is None
        assert resolve_config_key("rag.not_a_real_field") is None

    def test_private_fields_are_not_reachable(self):
        assert resolve_config_key("_loaded_path") is None

    def test_alias_is_preferred_over_same_named_field(self):
        """A curated alias must not be shadowed by the tail-segment fallback."""
        for dotted, flat in _DOT_TO_FLAT.items():
            assert resolve_config_key(dotted) == flat, dotted

    @pytest.mark.parametrize("name", _settable_fields())
    def test_every_field_is_reachable(self, name):
        """The whole point: no config knob may be unreachable from the API."""
        assert resolve_config_key(name) == name


class TestConfigUpdateHonesty:
    def test_unmodelled_keys_are_no_longer_dropped_silently(self):
        """Pydantic used to discard them before the route could report them."""
        req = ConfigUpdateRequest(**{"llm_provider": "local", "chunk_size": 1234})
        assert "chunk_size" in req.model_dump(exclude_unset=True)

    def test_declared_field_set_is_the_curated_subset(self):
        declared = set(ConfigUpdateRequest.model_fields) - {"persist"}
        assert "top_k" in declared
        # Storage/credential fields stay out of the live-tuning endpoint.
        assert "axon_store_base" not in declared
        assert "api_key" not in declared

    def test_anything_it_declares_is_a_real_config_field(self):
        real = set(_settable_fields())
        declared = set(ConfigUpdateRequest.model_fields) - {"persist"}
        assert not (declared - real)


class TestCrossSurfaceParity:
    def test_repl_and_rest_resolve_identically(self):
        """REPL /config set shares the resolver, so keys cannot drift apart."""
        import inspect

        from axon import repl

        src = inspect.getsource(repl)
        assert "resolve_config_key" in src, "REPL must use the shared resolver"

    def test_curated_aliases_all_point_at_real_fields(self):
        real = set(_settable_fields())
        broken = {k: v for k, v in _DOT_TO_FLAT.items() if v not in real}
        assert not broken, f"aliases pointing at non-existent fields: {broken}"


class TestMcpConfigTools:
    """MCP was the one surface with no config access at all.

    `surface_contract.py` already declared `config_read` and `config_update` as
    supported on ALL_SURFACES, so the contract asserted a parity that did not
    exist — nothing checked that the MCP server actually exposed the tools.
    """

    EXPECTED = ("get_config", "set_config", "update_config", "validate_config")

    def _tool_names(self):
        import inspect

        from axon import mcp_server

        src = inspect.getsource(mcp_server)
        import re

        return set(re.findall(r"@mcp\.tool\(\)\s*\nasync def (\w+)", src))

    @pytest.mark.parametrize("name", EXPECTED)
    def test_tool_exists(self, name):
        assert name in self._tool_names()

    def test_tools_are_registered_with_the_mcp_decorator(self):
        """A plain async def would import fine but never reach an agent."""
        names = self._tool_names()
        assert set(self.EXPECTED) <= names

    def test_routes_the_tools_call_exist(self):
        """Each tool proxies a REST route — a typo would 404 only at runtime."""
        from axon.api_routes import config_routes, projects

        paths = {
            r.path for router in (config_routes.router, projects.router) for r in router.routes
        }
        for path in ("/config", "/config/set", "/config/update", "/config/validate"):
            assert path in paths, f"{path} missing; MCP tool would 404"
