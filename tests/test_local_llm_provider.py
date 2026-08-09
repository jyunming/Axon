"""Tests for the `local` LLM provider — any OpenAI-compatible server on this box.

Covers the three things that made pointing Axon at llama.cpp / vLLM painful:
the reasoning_content black hole, the :8000 port collision, and having no way to
ask "is my local LLM actually up?".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig
from axon.doctor import check_local_llm_reachable
from axon.llm import OpenLLM, message_text


def _msg(content=None, **extra):
    """Build a stand-in for an OpenAI response message."""
    return SimpleNamespace(content=content, **extra)


class TestMessageText:
    def test_prefers_real_content(self):
        assert message_text(_msg(content="answer", reasoning_content="thinking")) == "answer"

    def test_falls_back_to_reasoning_content(self):
        """A truncated reasoning dump beats silently returning ''."""
        assert message_text(_msg(content="", reasoning_content="thinking")) == "thinking"

    def test_falls_back_when_content_is_none(self):
        assert message_text(_msg(content=None, reasoning_content="thinking")) == "thinking"

    def test_accepts_the_reasoning_alias(self):
        assert message_text(_msg(content=None, reasoning="thinking")) == "thinking"

    def test_reads_model_extra_when_field_is_not_an_attribute(self):
        """Some servers only surface the field through the raw payload."""
        msg = SimpleNamespace(content="", model_extra={"reasoning_content": "thinking"})
        assert message_text(msg) == "thinking"

    def test_empty_when_nothing_carries_text(self):
        assert message_text(_msg(content="")) == ""

    def test_no_crash_on_bare_message(self):
        assert message_text(SimpleNamespace()) == ""


class TestLocalProviderConfig:
    def test_default_does_not_collide_with_the_api_server(self):
        """axon-api binds :8000 — a local LLM default there would clash."""
        cfg = AxonConfig()
        assert ":8080" in cfg.local_base_url
        assert cfg.local_base_url != cfg.vllm_base_url

    def test_local_is_a_valid_provider(self):
        cfg = AxonConfig(llm_provider="local")
        assert cfg.llm_provider == "local"

    def test_max_tokens_default_fits_reasoning_models(self):
        """2048 truncates Gemma 4 / GPT-OSS mid-reasoning; 8192 is the floor."""
        assert AxonConfig().llm_max_tokens >= 8192

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("AXON_LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        assert AxonConfig().local_base_url == "http://127.0.0.1:1234/v1"

    def test_explicit_value_beats_env(self, monkeypatch):
        monkeypatch.setenv("AXON_LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        cfg = AxonConfig(local_base_url="http://127.0.0.1:9999/v1")
        assert cfg.local_base_url == "http://127.0.0.1:9999/v1"


class TestEffectiveTimeout:
    """The 60s cloud default is too short for locally served models.

    Measured on an Intel Arc iGPU serving Gemma 4 26B via llama.cpp: a
    two-character answer took 33s and a step-back reformulation took 75s. At 60s
    `step_back` and `query_decompose` timed out while `hyde` and `multi_query`
    scraped through — a partial, silent failure.
    """

    def test_local_gets_the_longer_default(self):
        assert OpenLLM(AxonConfig(llm_provider="local"))._effective_timeout() == 300

    def test_cloud_providers_still_fail_fast(self):
        for provider in ("openai", "gemini", "grok", "ollama"):
            llm = OpenLLM(AxonConfig(llm_provider=provider))
            assert llm._effective_timeout() == 60, provider

    def test_explicit_setting_wins(self):
        llm = OpenLLM(AxonConfig(llm_provider="local", llm_timeout=90))
        assert llm._effective_timeout() == 90

    def test_vllm_is_not_bumped(self):
        """Only `local` is bumped; vllm keeps prior behaviour."""
        assert OpenLLM(AxonConfig(llm_provider="vllm"))._effective_timeout() == 60

    def test_completion_uses_the_effective_timeout(self):
        llm = OpenLLM(AxonConfig(llm_provider="local"))
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=_msg(content="ok"))]
        )
        with patch.object(llm, "_local_client", return_value=client):
            llm.complete("hi")
        assert client.chat.completions.create.call_args.kwargs["timeout"] == 300


class TestGraphRagLocalWarning:
    """LLM graph extraction is one call per chunk — impractical on a slow local model.

    Measured with llama.cpp serving Gemma 4 26B on an Intel Arc iGPU (~4 tok/s):
    a single extraction call ran past 10 minutes without completing, and
    `llm.timeout` did not cut it short because that bound is per-read, not
    wall-clock. `graph_rag_depth: light` extracts the same entities in 0.6s.
    """

    def _issues(self, tmp_path, llm_provider, **rag):
        """validate() is a classmethod that reads a file — so write the YAML.

        `llm_provider` is written to `llm.provider`, the key users actually put
        in config.yaml. An earlier version accepted it via **rag and it landed
        in the `rag:` block instead; the tests still passed, but only because
        load() applies `rag:` after `llm:` and `rag.llm_provider` happened to
        overwrite the real one — exercising a config shape nobody writes.

        Deliberately NOT via AxonConfig.save(): that helper does not serialise
        graph_rag_depth or graph_rag_ner_backend, so a saved config would always
        validate against the defaults and pass for the wrong reason.
        """
        import yaml

        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump({"llm": {"provider": llm_provider}, "rag": rag}), encoding="utf-8"
        )
        return [i for i in AxonConfig.validate(path=str(path)) if i.field == "graph_rag_depth"]

    def test_helper_sets_the_real_llm_provider(self, tmp_path):
        """Guard the fixture itself: the provider must reach llm.provider."""
        import yaml

        self._issues(tmp_path, llm_provider="openai", graph_rag=True)
        raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert raw["llm"]["provider"] == "openai"
        assert "llm_provider" not in raw["rag"]

    def test_warns_on_local_plus_llm_extraction(self, tmp_path):
        issues = [
            i
            for i in self._issues(
                tmp_path, llm_provider="local", graph_rag=True, graph_rag_depth="standard"
            )
            if i.level == "warn"
        ]
        assert issues, "expected a warning for local + LLM graph extraction"
        assert "light" in issues[0].suggestion

    def test_silent_when_depth_is_light(self, tmp_path):
        assert not self._issues(
            tmp_path, llm_provider="local", graph_rag=True, graph_rag_depth="light"
        )

    def test_silent_when_ner_backend_avoids_the_llm(self, tmp_path):
        assert not self._issues(
            tmp_path,
            llm_provider="local",
            graph_rag=True,
            graph_rag_depth="standard",
            graph_rag_ner_backend="gliner",
        )

    def test_silent_when_graph_rag_is_off(self, tmp_path):
        assert not self._issues(
            tmp_path, llm_provider="local", graph_rag=False, graph_rag_depth="standard"
        )

    def test_silent_for_cloud_providers(self, tmp_path):
        """A hosted model is fast enough that per-chunk extraction is fine."""
        assert not self._issues(
            tmp_path, llm_provider="openai", graph_rag=True, graph_rag_depth="standard"
        )

    def test_warning_is_not_an_error(self, tmp_path):
        """Never block the run — a fast local model on a real GPU is fine."""
        issues = self._issues(
            tmp_path, llm_provider="local", graph_rag=True, graph_rag_depth="standard"
        )
        assert issues and all(i.level != "error" for i in issues)


class TestLocalClient:
    def test_uses_configured_base_url(self):
        cfg = AxonConfig(llm_provider="local", local_base_url="http://127.0.0.1:8080/v1")
        llm = OpenLLM(cfg)
        with patch.object(llm, "_get_openai_client") as mock_get:
            llm._local_client()
        assert mock_get.call_args.kwargs["base_url"] == "http://127.0.0.1:8080/v1"

    def test_supplies_a_dummy_key_when_unset(self):
        """The OpenAI SDK refuses to build a client without some api_key."""
        llm = OpenLLM(AxonConfig(llm_provider="local", local_api_key=""))
        with patch.object(llm, "_get_openai_client") as mock_get:
            llm._local_client()
        assert mock_get.call_args.kwargs["api_key"]

    def test_forwards_a_real_key_when_set(self):
        llm = OpenLLM(AxonConfig(llm_provider="local", local_api_key="secret"))
        with patch.object(llm, "_get_openai_client") as mock_get:
            llm._local_client()
        assert mock_get.call_args.kwargs["api_key"] == "secret"


class TestLocalCompletion:
    def _llm_with_response(self, message):
        llm = OpenLLM(AxonConfig(llm_provider="local", llm_model="gemma4-26b"))
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        return llm, client

    def test_complete_returns_content(self):
        llm, client = self._llm_with_response(_msg(content="hello"))
        with patch.object(llm, "_local_client", return_value=client):
            assert llm.complete("hi") == "hello"
        assert client.chat.completions.create.call_args.kwargs["model"] == "gemma4-26b"

    def test_complete_recovers_reasoning_only_response(self):
        """The regression this provider exists to prevent."""
        llm, client = self._llm_with_response(_msg(content=None, reasoning_content="thought"))
        with patch.object(llm, "_local_client", return_value=client):
            assert llm.complete("hi") == "thought"

    def test_vllm_still_routed_to_its_own_url(self):
        cfg = AxonConfig(llm_provider="vllm", vllm_base_url="http://localhost:8000/v1")
        llm = OpenLLM(cfg)
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=_msg(content="ok"))]
        )
        with patch.object(llm, "_get_openai_client", return_value=client) as mock_get:
            assert llm.complete("hi") == "ok"
        assert mock_get.call_args.kwargs["base_url"] == "http://localhost:8000/v1"


class TestPingLocal:
    def _llm(self):
        return OpenLLM(AxonConfig(llm_provider="local", local_base_url="http://x:8080/v1"))

    def test_lists_models_in_openai_shape(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "gemma4-26b"}, {"id": "qwen3-coder"}]}
        with patch("httpx.get", return_value=response):
            result = self._llm().ping_local()
        assert result["reachable"] is True
        assert result["models"] == ["gemma4-26b", "qwen3-coder"]
        assert result["error"] is None

    def test_accepts_a_bare_list_payload(self):
        response = MagicMock()
        response.json.return_value = [{"id": "m1"}]
        with patch("httpx.get", return_value=response):
            assert self._llm().ping_local()["models"] == ["m1"]

    def test_unreachable_reports_instead_of_raising(self):
        with patch("httpx.get", side_effect=OSError("connection refused")):
            result = self._llm().ping_local()
        assert result["reachable"] is False
        assert "connection refused" in result["error"]

    def test_empty_base_url_is_reported(self):
        llm = OpenLLM(AxonConfig(llm_provider="local", local_base_url=""))
        assert llm.ping_local()["error"] == "no base_url configured"


class TestDoctorLocalCheck:
    def test_quiet_when_provider_is_not_local(self):
        result = check_local_llm_reachable("ollama", "http://x:8080/v1")
        assert result.status == "ok"
        assert "not in use" in result.detail

    def test_error_when_base_url_missing(self):
        result = check_local_llm_reachable("local", "")
        assert result.status == "error"
        assert "AXON_LOCAL_LLM_BASE_URL" in result.hint

    def test_error_when_unreachable(self):
        with patch("httpx.get", side_effect=OSError("refused")):
            result = check_local_llm_reachable("local", "http://x:8080/v1")
        assert result.status == "error"
        assert "unreachable" in result.detail

    def test_warns_when_up_but_serving_nothing(self):
        """llama-server answers /models before any model is resident."""
        response = MagicMock()
        response.json.return_value = {"data": []}
        with patch("httpx.get", return_value=response):
            result = check_local_llm_reachable("local", "http://x:8080/v1")
        assert result.status == "warning"
        assert "no models" in result.detail

    def test_ok_lists_models(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "gemma4-26b"}]}
        with patch("httpx.get", return_value=response):
            result = check_local_llm_reachable("local", "http://x:8080/v1")
        assert result.status == "ok"
        assert "gemma4-26b" in result.detail

    def test_truncates_a_long_model_list(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": f"m{i}"} for i in range(6)]}
        with patch("httpx.get", return_value=response):
            result = check_local_llm_reachable("local", "http://x:8080/v1")
        assert "+3 more" in result.detail


class TestLocalToolCalling:
    """Function calling must work for local models, not just cloud ones."""

    TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    def _llm_and_client(self, message):
        llm = OpenLLM(AxonConfig(llm_provider="local", llm_model="gemma4-26b"))
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        return llm, client

    def test_tool_calls_are_parsed(self):
        call = SimpleNamespace(function=SimpleNamespace(name="search", arguments='{"q": "axon"}'))
        message = SimpleNamespace(content=None, tool_calls=[call])
        llm, client = self._llm_and_client(message)
        with patch.object(llm, "_local_client", return_value=client):
            result = llm.complete_with_tools("hi", tools=self.TOOLS)
        assert [(c.name, c.args) for c in result] == [("search", {"q": "axon"})]
        assert client.chat.completions.create.call_args.kwargs["tool_choice"] == "auto"

    def test_plain_text_reply_falls_back_to_reasoning(self):
        """No tool call + empty content must still yield the reasoning text."""
        message = SimpleNamespace(content=None, tool_calls=None, reasoning_content="thought")
        llm, client = self._llm_and_client(message)
        with patch.object(llm, "_local_client", return_value=client):
            assert llm.complete_with_tools("hi", tools=self.TOOLS) == "thought"


class TestLocalStreaming:
    def test_reasoning_deltas_are_not_streamed_to_the_user(self):
        """Chain-of-thought is a scratchpad, not the answer."""
        llm = OpenLLM(AxonConfig(llm_provider="local"))
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning="th"))]
            ),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))]),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(chunks)
        with patch.object(llm, "_local_client", return_value=client):
            assert "".join(llm.stream("hi")) == "Hi there"

    def test_chunk_without_choices_is_skipped(self):
        """Some servers emit a trailing usage-only chunk with choices == []."""
        llm = OpenLLM(AxonConfig(llm_provider="local"))
        chunks = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(chunks)
        with patch.object(llm, "_local_client", return_value=client):
            assert "".join(llm.stream("hi")) == "ok"


class TestReasoningFallbackAcrossProviders:
    """The reasoning_content fallback must cover every OpenAI-dialect provider.

    It was originally applied only to the `local` / `vllm` branch of complete()
    and to complete_with_tools, so `openai`, `grok` and `github_copilot` still
    returned message.content directly. Anyone pointing those providers at a
    reasoning model — a self-hosted OpenAI-compatible gateway, or a hosted model
    that emits reasoning_content — got "" back with no error.
    """

    PROVIDERS = ("openai", "grok", "github_copilot", "vllm", "local")

    def _client_returning(self, message):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        return client

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_reasoning_only_response_is_recovered(self, provider):
        llm = OpenLLM(AxonConfig(llm_provider=provider, llm_model="m"))
        client = self._client_returning(_msg(content=None, reasoning_content="thought"))
        with (
            patch.object(llm, "_get_openai_client", return_value=client),
            patch.object(llm, "_local_client", return_value=client),
            patch.object(llm, "_get_grok_client", return_value=client),
            patch.object(llm, "_get_copilot_client", return_value=client),
        ):
            assert llm.complete("hi") == "thought", provider

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_real_content_still_wins(self, provider):
        llm = OpenLLM(AxonConfig(llm_provider=provider, llm_model="m"))
        client = self._client_returning(_msg(content="answer", reasoning_content="thought"))
        with (
            patch.object(llm, "_get_openai_client", return_value=client),
            patch.object(llm, "_local_client", return_value=client),
            patch.object(llm, "_get_grok_client", return_value=client),
            patch.object(llm, "_get_copilot_client", return_value=client),
        ):
            assert llm.complete("hi") == "answer", provider


class TestDoctorSendsAuth:
    """A gated local endpoint would 401 an unauthenticated probe.

    The doctor would then report "unreachable" for an endpoint that is fine —
    it must send the same Authorization header the real client uses.
    """

    def _captured_headers(self, api_key):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "m1"}]}
        with patch("httpx.get", return_value=response) as mock_get:
            check_local_llm_reachable("local", "http://x:8080/v1", api_key)
        return mock_get.call_args.kwargs.get("headers") or {}

    def test_bearer_header_sent_when_key_set(self):
        assert self._captured_headers("secret")["Authorization"] == "Bearer secret"

    def test_no_auth_header_when_key_absent(self):
        assert "Authorization" not in self._captured_headers("")

    def test_key_is_threaded_from_config(self):
        """run_doctor must pass local_api_key through, not drop it."""
        from axon.doctor import run_doctor

        cfg = AxonConfig(
            llm_provider="local",
            local_base_url="http://x:8080/v1",
            local_api_key="from-config",
        )
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "m1"}]}
        with patch("httpx.get", return_value=response) as mock_get:
            run_doctor(cfg)
        auth = [
            c.kwargs.get("headers", {}).get("Authorization")
            for c in mock_get.call_args_list
            if c.kwargs.get("headers")
        ]
        assert "Bearer from-config" in auth
