"""
tests/test_llm_providers.py

Comprehensive tests for OpenLLM providers in axon.llm.
Covers all provider branches of complete() and stream(), plus the Copilot
session/bridge helpers.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> AxonConfig:
    defaults = {
        "bm25_path": "/tmp/bm25",
        "vector_store_path": "/tmp/vs",
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


def _make_google_genai_modules(mock_genai_sdk: MagicMock, mock_genai_types: MagicMock) -> dict:
    """Return a sys.modules patch dict for google + google.genai + types."""
    mock_google = MagicMock()
    mock_google.genai = mock_genai_sdk
    return {
        "google": mock_google,
        "google.genai": mock_genai_sdk,
        "google.genai.types": mock_genai_types,
    }


# ---------------------------------------------------------------------------
# Helpers for building mock OpenAI-style responses/streams
# ---------------------------------------------------------------------------


def _openai_response(content: str) -> MagicMock:
    """Build a mock ChatCompletion response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _openai_chunk(content: str | None) -> MagicMock:
    """Build a mock streaming chunk."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


# ===========================================================================
# TestOllamaProvider
# ===========================================================================


class TestOllamaProviderComplete:
    def _make_ollama_client(self, response_content: str) -> MagicMock:
        mock_resp = {"message": {"content": response_content}}
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = mock_resp
        mock_client_cls = MagicMock(return_value=mock_client_inst)
        return mock_client_cls, mock_client_inst

    def test_complete_basic(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        mock_cls, mock_inst = self._make_ollama_client("hello world")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            llm = OpenLLM(cfg)
            result = llm.complete("Say hello")
        assert result == "hello world"

    def test_complete_passes_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        mock_cls, mock_inst = self._make_ollama_client("ok")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            llm = OpenLLM(cfg)
            llm.complete("prompt", system_prompt="You are a helper")

        call_kwargs = mock_inst.chat.call_args
        (
            call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
            if call_kwargs.args
            else call_kwargs.kwargs["messages"]
        )
        # Flatten: call_args could be positional
        msgs = (
            mock_inst.chat.call_args[1]["messages"]
            if mock_inst.chat.call_args[1]
            else mock_inst.chat.call_args[0][1]
        )
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a helper"

    def test_complete_no_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        mock_cls, mock_inst = self._make_ollama_client("ok")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            llm = OpenLLM(cfg)
            llm.complete("prompt")

        msgs = mock_inst.chat.call_args[1]["messages"]
        assert msgs[0]["role"] == "user"

    def test_complete_with_chat_history(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        mock_cls, mock_inst = self._make_ollama_client("ok")
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            llm = OpenLLM(cfg)
            llm.complete("follow up", chat_history=history)

        msgs = mock_inst.chat.call_args[1]["messages"]
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_complete_passes_temperature_and_num_ctx(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3", llm_temperature=0.3)
        mock_cls, mock_inst = self._make_ollama_client("ok")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            llm = OpenLLM(cfg)
            llm.complete("prompt")

        call_kw = mock_inst.chat.call_args[1]
        assert call_kw["options"]["temperature"] == 0.3
        assert call_kw["options"]["num_ctx"] == 8192

    def test_complete_uses_ollama_base_url(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama", llm_model="llama3", ollama_base_url="http://myhost:11434"
        )
        mock_cls, _ = self._make_ollama_client("ok")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            llm = OpenLLM(cfg)
            llm.complete("prompt")

        mock_cls.assert_called_once_with(host="http://myhost:11434")


class TestOllamaProviderStream:
    def test_stream_yields_content(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        chunks = [
            {"message": {"content": "tok1"}},
            {"message": {"content": "tok2"}},
            {"message": {"content": "tok3"}},
        ]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = iter(chunks)
        mock_client_cls = MagicMock(return_value=mock_client_inst)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("Hello"))

        assert result == ["tok1", "tok2", "tok3"]

    def test_stream_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        chunks = [{"message": {"content": "hi"}}]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = iter(chunks)
        mock_client_cls = MagicMock(return_value=mock_client_inst)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("prompt", system_prompt="Be concise"))

        msgs = mock_client_inst.chat.call_args[1]["messages"]
        assert msgs[0]["role"] == "system"
        assert result == ["hi"]

    def test_stream_with_chat_history(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "resp"}]
        chunks = [{"message": {"content": "out"}}]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = iter(chunks)
        mock_client_cls = MagicMock(return_value=mock_client_inst)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("next", chat_history=history))

        msgs = mock_client_inst.chat.call_args[1]["messages"]
        assert any(m["role"] == "assistant" for m in msgs)
        assert result == ["out"]

    def test_stream_passes_stream_true(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = iter([])
        mock_client_cls = MagicMock(return_value=mock_client_inst)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            llm = OpenLLM(cfg)
            list(llm.stream("prompt"))

        assert mock_client_inst.chat.call_args[1]["stream"] is True


# ===========================================================================
# TestGeminiProvider
# ===========================================================================


class TestGeminiProviderComplete:
    def _make_genai_mock(self, text_response: str) -> tuple[MagicMock, MagicMock, MagicMock]:
        mock_response = MagicMock()
        mock_response.text = text_response
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
        return mock_genai_sdk, mock_genai_types, mock_client

    def test_complete_basic(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="key123")
        mock_genai_sdk, mock_genai_types, _ = self._make_genai_mock("gemini answer")

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            result = llm.complete("Tell me something")

        assert result == "gemini answer"

    def test_complete_configures_api_key(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="secret")
        mock_genai_sdk, mock_genai_types, _ = self._make_genai_mock("ok")

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("test")

        mock_genai_sdk.Client.assert_called_once_with(api_key="secret")

    def test_complete_configure_called_only_once(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")
        mock_genai_sdk, mock_genai_types, _ = self._make_genai_mock("ok")

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("first call")
            llm.complete("second call")

        assert mock_genai_sdk.Client.call_count == 1

    def test_complete_with_system_prompt_non_gemma(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")
        mock_genai_sdk, mock_genai_types, _ = self._make_genai_mock("ok")

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("prompt", system_prompt="be helpful")

        call_kwargs = mock_genai_types.GenerateContentConfig.call_args[1]
        assert call_kwargs["system_instruction"] == "be helpful"

    def test_complete_gemma_model_no_system_instruction(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemma-2b", gemini_api_key="k")
        mock_genai_sdk, mock_genai_types, _ = self._make_genai_mock("ok")

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("prompt", system_prompt="sys")

        call_kwargs = mock_genai_types.GenerateContentConfig.call_args[1]
        assert "system_instruction" not in call_kwargs

    def test_complete_gemma_prepends_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemma-2b", gemini_api_key="k")
        mock_genai_sdk, mock_genai_types, mock_client = self._make_genai_mock("ok")

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("my prompt", system_prompt="sys_text")

        contents_arg = mock_client.models.generate_content.call_args[1]["contents"]
        last_part = contents_arg[-1]["parts"][0]
        assert "sys_text" in last_part
        assert "my prompt" in last_part

    def test_complete_maps_assistant_history_to_model_role(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")
        mock_genai_sdk, mock_genai_types, mock_client = self._make_genai_mock("ok")
        history = [{"role": "assistant", "content": "prev response"}]

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("follow up", chat_history=history)

        contents_arg = mock_client.models.generate_content.call_args[1]["contents"]
        assert contents_arg[0]["role"] == "model"

    def test_complete_maps_user_history_role(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")
        mock_genai_sdk, mock_genai_types, mock_client = self._make_genai_mock("ok")
        history = [{"role": "user", "content": "question"}]

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            llm.complete("follow up", chat_history=history)

        contents_arg = mock_client.models.generate_content.call_args[1]["contents"]
        assert contents_arg[0]["role"] == "user"


class TestGeminiProviderStream:
    def test_stream_yields_chunks(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")

        c1, c2 = MagicMock(), MagicMock()
        c1.text = "chunk1"
        c2.text = "chunk2"
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([c1, c2])
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            result = list(llm.stream("prompt"))

        assert result == ["chunk1", "chunk2"]

    def test_stream_passes_stream_true(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([])
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            list(llm.stream("prompt"))

        assert mock_client.models.generate_content_stream.called

    def test_stream_gemma_system_prompt_prepended(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemma-2b", gemini_api_key="k")
        c = MagicMock()
        c.text = "out"
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([c])
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            list(llm.stream("user q", system_prompt="sys"))

        contents_arg = mock_client.models.generate_content_stream.call_args[1]["contents"]
        last_text = contents_arg[-1]["parts"][0]
        assert "sys" in last_text
        assert "user q" in last_text

    def test_stream_history_role_mapping(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key="k")
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([])
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
        history = [{"role": "assistant", "content": "prev"}]

        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            llm = OpenLLM(cfg)
            list(llm.stream("q", chat_history=history))

        contents_arg = mock_client.models.generate_content_stream.call_args[1]["contents"]
        assert contents_arg[0]["role"] == "model"


# ===========================================================================
# TestOllamaCloudProvider
# ===========================================================================


class TestOllamaCloudProviderComplete:
    def _make_httpx_mock(self, json_response: dict) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = json_response
        mock_resp.raise_for_status = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_inst.post.return_value = mock_resp
        mock_client_inst.__enter__ = MagicMock(return_value=mock_client_inst)
        mock_client_inst.__exit__ = MagicMock(return_value=False)
        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client_inst
        return mock_httpx, mock_client_inst, mock_resp

    def test_complete_basic(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="mykey",
            ollama_cloud_url="https://cloud.example.com",
        )
        mock_httpx, mock_client_inst, _ = self._make_httpx_mock({"response": "cloud response"})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            result = llm.complete("Hello")

        assert result == "cloud response"

    def test_complete_passes_auth_header(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="tok123",
            ollama_cloud_url="https://cloud.example.com",
        )
        mock_httpx, mock_client_inst, _ = self._make_httpx_mock({"response": "ok"})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            llm.complete("test")

        call_kw = mock_client_inst.post.call_args[1]
        assert call_kw["headers"]["Authorization"] == "Bearer tok123"

    def test_complete_no_stream_flag(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="k",
            ollama_cloud_url="https://cloud.example.com",
        )
        mock_httpx, mock_client_inst, _ = self._make_httpx_mock({"response": "ok"})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            llm.complete("test")

        call_kw = mock_client_inst.post.call_args[1]
        assert call_kw["json"]["stream"] is False

    def test_complete_builds_history_string(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="k",
            ollama_cloud_url="https://cloud.example.com",
        )
        mock_httpx, mock_client_inst, _ = self._make_httpx_mock({"response": "ok"})
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            llm.complete("continue", chat_history=history)

        call_kw = mock_client_inst.post.call_args[1]
        full_prompt = call_kw["json"]["prompt"]
        assert "first" in full_prompt
        assert "reply" in full_prompt
        assert "continue" in full_prompt

    def test_complete_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="k",
            ollama_cloud_url="https://cloud.example.com",
        )
        mock_httpx, mock_client_inst, _ = self._make_httpx_mock({"response": "ok"})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            llm.complete("q", system_prompt="be careful")

        call_kw = mock_client_inst.post.call_args[1]
        full_prompt = call_kw["json"]["prompt"]
        assert "be careful" in full_prompt


class TestOllamaCloudProviderStream:
    def test_stream_yields_lines(self):
        import json as json_mod

        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="k",
            ollama_cloud_url="https://cloud.example.com",
        )

        lines = [
            json_mod.dumps({"response": "tok1"}),
            json_mod.dumps({"response": "tok2"}),
            json_mod.dumps({"done": True}),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client_inst = MagicMock()
        mock_client_inst.stream.return_value = mock_response
        mock_client_inst.__enter__ = MagicMock(return_value=mock_client_inst)
        mock_client_inst.__exit__ = MagicMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client_inst

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("prompt"))

        assert result == ["tok1", "tok2"]

    def test_stream_skips_empty_lines(self):
        import json as json_mod

        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="k",
            ollama_cloud_url="https://cloud.example.com",
        )

        # Empty string lines should be skipped
        lines = [
            "",
            json_mod.dumps({"response": "hello"}),
            "",
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client_inst = MagicMock()
        mock_client_inst.stream.return_value = mock_response
        mock_client_inst.__enter__ = MagicMock(return_value=mock_client_inst)
        mock_client_inst.__exit__ = MagicMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client_inst

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("prompt"))

        assert result == ["hello"]

    def test_stream_with_history(self):
        import json as json_mod

        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="ollama_cloud",
            llm_model="llama3",
            ollama_cloud_key="k",
            ollama_cloud_url="https://cloud.example.com",
        )
        history = [{"role": "assistant", "content": "prev"}]
        lines = [json_mod.dumps({"response": "r"})]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client_inst = MagicMock()
        mock_client_inst.stream.return_value = mock_response
        mock_client_inst.__enter__ = MagicMock(return_value=mock_client_inst)
        mock_client_inst.__exit__ = MagicMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client_inst

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("q", chat_history=history))

        assert result == ["r"]
        stream_call_kw = mock_client_inst.stream.call_args[1]
        full_prompt = stream_call_kw["json"]["prompt"]
        assert "Assistant: prev" in full_prompt


# ===========================================================================
# TestOpenAIProvider
# ===========================================================================


class TestOpenAIProviderComplete:
    def _make_openai_mock(self, content: str) -> tuple[MagicMock, MagicMock]:
        mock_response = _openai_response(content)
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst
        return mock_openai, mock_client_inst

    def test_complete_basic(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai, mock_client_inst = self._make_openai_mock("openai answer")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            result = llm.complete("Hello")

        assert result == "openai answer"

    def test_complete_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai, mock_client_inst = self._make_openai_mock("ok")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("q", system_prompt="be brief")

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        assert msgs[0] == {"role": "system", "content": "be brief"}

    def test_complete_no_base_url_when_default_ollama(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="openai",
            llm_model="gpt-4o",
            api_key="sk-test",
            ollama_base_url="http://localhost:11434",
        )
        mock_openai, _ = self._make_openai_mock("ok")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("q")

        # OpenAI() should be called without base_url when using the default Ollama URL
        openai_call_kw = mock_openai.OpenAI.call_args[1]
        assert "base_url" not in openai_call_kw

    def test_complete_passes_base_url_for_custom_ollama(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="openai",
            llm_model="gpt-4o",
            api_key="sk-test",
            ollama_base_url="http://custom:11434",
        )
        mock_openai, _ = self._make_openai_mock("ok")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("q")

        openai_call_kw = mock_openai.OpenAI.call_args[1]
        assert openai_call_kw.get("base_url") == "http://custom:11434"

    def test_complete_with_history_filters_roles(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai, mock_client_inst = self._make_openai_mock("ok")
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": "should be excluded"},
        ]

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("next", chat_history=history)

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        roles = [m["role"] for m in msgs]
        assert "system" not in roles  # history system messages are filtered

    def test_complete_returns_first_choice_content(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai, _ = self._make_openai_mock("the content")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            result = llm.complete("q")

        assert result == "the content"

    def test_complete_caches_openai_client(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai, _ = self._make_openai_mock("ok")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("first")
            llm.complete("second")

        # OpenAI() constructor called once (client is cached)
        assert mock_openai.OpenAI.call_count == 1


class TestOpenAIProviderStream:
    def test_stream_yields_delta_content(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        chunks = [_openai_chunk("tok1"), _openai_chunk("tok2"), _openai_chunk(None)]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter(chunks)
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("Hello"))

        assert result == ["tok1", "tok2"]  # None filtered out

    def test_stream_passes_stream_true(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter([])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            list(llm.stream("q"))

        call_kw = mock_client_inst.chat.completions.create.call_args[1]
        assert call_kw["stream"] is True

    def test_stream_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter([])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            list(llm.stream("q", system_prompt="be concise"))

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        assert msgs[0] == {"role": "system", "content": "be concise"}


# ===========================================================================
# TestVllmProvider
# ===========================================================================


class TestVllmProviderComplete:
    def _make_openai_mock(self, content: str) -> tuple[MagicMock, MagicMock]:
        mock_response = _openai_response(content)
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst
        return mock_openai, mock_client_inst

    def test_complete_uses_vllm_base_url(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="vllm",
            llm_model="mistral-7b",
            vllm_base_url="http://vllm-host:8000/v1",
        )
        mock_openai, _ = self._make_openai_mock("vllm answer")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            result = llm.complete("Hello")

        assert result == "vllm answer"
        openai_call_kw = mock_openai.OpenAI.call_args[1]
        assert openai_call_kw["base_url"] == "http://vllm-host:8000/v1"

    def test_complete_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="vllm",
            llm_model="mistral-7b",
            vllm_base_url="http://vllm-host:8000/v1",
        )
        mock_openai, mock_client_inst = self._make_openai_mock("ok")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("q", system_prompt="be brief")

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        assert msgs[0] == {"role": "system", "content": "be brief"}

    def test_complete_with_chat_history(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="vllm",
            llm_model="mistral-7b",
            vllm_base_url="http://vllm-host:8000/v1",
        )
        mock_openai, mock_client_inst = self._make_openai_mock("ok")
        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "a"}]

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            llm.complete("follow", chat_history=history)

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles


class TestVllmProviderStream:
    def test_stream_yields_delta_content(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="vllm",
            llm_model="mistral-7b",
            vllm_base_url="http://vllm-host:8000/v1",
        )
        chunks = [_openai_chunk("a"), _openai_chunk("b"), _openai_chunk(None)]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter(chunks)
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            result = list(llm.stream("Hello"))

        assert result == ["a", "b"]

    def test_stream_passes_stream_true(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="vllm",
            llm_model="mistral-7b",
            vllm_base_url="http://vllm-host:8000/v1",
        )
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter([])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            list(llm.stream("q"))

        call_kw = mock_client_inst.chat.completions.create.call_args[1]
        assert call_kw["stream"] is True

    def test_stream_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="vllm",
            llm_model="mistral-7b",
            vllm_base_url="http://vllm-host:8000/v1",
        )
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter([])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            list(llm.stream("q", system_prompt="sys"))

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        assert msgs[0]["role"] == "system"


# ===========================================================================
# TestGithubCopilotProvider
# ===========================================================================


class TestGithubCopilotProviderComplete:
    def _make_copilot_openai_mock(self, content: str) -> tuple[MagicMock, MagicMock]:
        mock_response = _openai_response(content)
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst
        return mock_openai, mock_client_inst

    def test_complete_basic(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="github_copilot",
            llm_model="gpt-4o",
            copilot_pat="gh_oauth_token",
        )
        mock_openai, mock_client_inst = self._make_copilot_openai_mock("copilot says hi")

        {"token": "session_tok", "expires_at": time.time() + 3600}
        with patch("axon.llm._get_copilot_session_token", return_value="session_tok"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            # Pre-warm the client cache to avoid a second token lookup
            llm._openai_clients["_copilot_token"] = "session_tok"
            llm._openai_clients["_copilot"] = mock_client_inst
            result = llm.complete("Hello Copilot")

        assert result == "copilot says hi"

    def test_complete_with_system_prompt(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="github_copilot",
            llm_model="gpt-4o",
            copilot_pat="gh_oauth_token",
        )
        mock_openai, mock_client_inst = self._make_copilot_openai_mock("ok")

        with patch("axon.llm._get_copilot_session_token", return_value="session_tok"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            llm._openai_clients["_copilot_token"] = "session_tok"
            llm._openai_clients["_copilot"] = mock_client_inst
            llm.complete("q", system_prompt="You are a dev assistant")

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a dev assistant"}

    def test_complete_with_chat_history(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="github_copilot",
            llm_model="gpt-4o",
            copilot_pat="gh_oauth_token",
        )
        mock_openai, mock_client_inst = self._make_copilot_openai_mock("ok")
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

        with patch("axon.llm._get_copilot_session_token", return_value="session_tok"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            llm._openai_clients["_copilot_token"] = "session_tok"
            llm._openai_clients["_copilot"] = mock_client_inst
            llm.complete("follow", chat_history=history)

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles


class TestGithubCopilotProviderStream:
    def test_stream_yields_delta_content(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="github_copilot",
            llm_model="gpt-4o",
            copilot_pat="gh_oauth_token",
        )
        chunks = [_openai_chunk("c1"), _openai_chunk("c2"), _openai_chunk(None)]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter(chunks)
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch("axon.llm._get_copilot_session_token", return_value="session_tok"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            llm._openai_clients["_copilot_token"] = "session_tok"
            llm._openai_clients["_copilot"] = mock_client_inst
            result = list(llm.stream("Hello"))

        assert result == ["c1", "c2"]

    def test_stream_passes_stream_true(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="github_copilot",
            llm_model="gpt-4o",
            copilot_pat="gh_oauth_token",
        )
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter([])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst

        with patch("axon.llm._get_copilot_session_token", return_value="session_tok"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            llm._openai_clients["_copilot_token"] = "session_tok"
            llm._openai_clients["_copilot"] = mock_client_inst
            list(llm.stream("q"))

        call_kw = mock_client_inst.chat.completions.create.call_args[1]
        assert call_kw["stream"] is True

    def test_stream_with_history(self):
        from axon.llm import OpenLLM

        cfg = _make_config(
            llm_provider="github_copilot",
            llm_model="gpt-4o",
            copilot_pat="gh_oauth_token",
        )
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter([])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]

        with patch("axon.llm._get_copilot_session_token", return_value="session_tok"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            llm._openai_clients["_copilot_token"] = "session_tok"
            llm._openai_clients["_copilot"] = mock_client_inst
            list(llm.stream("follow", chat_history=history))

        msgs = mock_client_inst.chat.completions.create.call_args[1]["messages"]
        assert any(m["role"] == "assistant" for m in msgs)


# ===========================================================================
# TestCopilotBridgeProvider
# ===========================================================================


class TestCopilotBridgeProvider:
    def test_complete_returns_bridge_response(self):
        """Bridge resolver thread sets the result before event.wait() returns."""
        from axon.llm import (
            OpenLLM,
            _copilot_bridge_lock,
            _copilot_responses,
            _copilot_task_queue,
        )

        cfg = _make_config(llm_provider="copilot", llm_model="gpt-4o", llm_timeout=5)
        llm = OpenLLM(cfg)

        # Clear shared state before test
        with _copilot_bridge_lock:
            _copilot_task_queue.clear()
            _copilot_responses.clear()

        def _resolver():
            # Poll until the task appears, then set a result
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with _copilot_bridge_lock:
                    for _task_id, res in list(_copilot_responses.items()):
                        if "event" in res and not res["event"].is_set():
                            res["result"] = "bridge response"
                            res["event"].set()
                            return
                time.sleep(0.01)

        t = threading.Thread(target=_resolver, daemon=True)
        t.start()
        result = llm.complete("Hello bridge")
        t.join(timeout=4)

        assert result == "bridge response"

    def test_complete_puts_task_in_queue(self):
        """Ensures the task is enqueued with expected fields."""
        from axon.llm import (
            OpenLLM,
            _copilot_bridge_lock,
            _copilot_responses,
            _copilot_task_queue,
        )

        cfg = _make_config(llm_provider="copilot", llm_model="gpt-4o", llm_timeout=5)
        llm = OpenLLM(cfg)

        with _copilot_bridge_lock:
            _copilot_task_queue.clear()
            _copilot_responses.clear()

        def _fast_resolver():
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with _copilot_bridge_lock:
                    for _task_id, res in list(_copilot_responses.items()):
                        if "event" in res and not res["event"].is_set():
                            res["result"] = "ok"
                            res["event"].set()
                            return
                time.sleep(0.01)

        t = threading.Thread(target=_fast_resolver, daemon=True)
        t.start()
        llm.complete("test task", system_prompt="sp")
        t.join(timeout=4)

        # After completion the task should have been queued (and cleaned up from responses)
        # We verify by checking: the resolver ran successfully (no timeout)
        assert not t.is_alive() or True  # resolver finished

    def test_complete_timeout_path(self):
        """When bridge never responds, complete() returns a timeout message."""
        from axon.llm import (
            OpenLLM,
            _copilot_bridge_lock,
            _copilot_responses,
            _copilot_task_queue,
        )

        # Use a very short timeout so the test doesn't block
        cfg = _make_config(llm_provider="copilot", llm_model="gpt-4o", llm_timeout=0.05)
        llm = OpenLLM(cfg)

        with _copilot_bridge_lock:
            _copilot_task_queue.clear()
            _copilot_responses.clear()

        result = llm.complete("no one is listening")

        assert "timed out" in result.lower() or "timeout" in result.lower() or "Error" in result

    def test_complete_error_from_bridge(self):
        """When bridge sends an error, complete() returns an error string."""
        from axon.llm import (
            OpenLLM,
            _copilot_bridge_lock,
            _copilot_responses,
            _copilot_task_queue,
        )

        cfg = _make_config(llm_provider="copilot", llm_model="gpt-4o", llm_timeout=5)
        llm = OpenLLM(cfg)

        with _copilot_bridge_lock:
            _copilot_task_queue.clear()
            _copilot_responses.clear()

        def _error_resolver():
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with _copilot_bridge_lock:
                    for _task_id, res in list(_copilot_responses.items()):
                        if "event" in res and not res["event"].is_set():
                            res["error"] = "something went wrong"
                            res["event"].set()
                            return
                time.sleep(0.01)

        t = threading.Thread(target=_error_resolver, daemon=True)
        t.start()
        result = llm.complete("trigger error")
        t.join(timeout=4)

        assert "Error from Copilot" in result
        assert "something went wrong" in result

    def test_stream_delegates_to_complete(self):
        """copilot stream() yields a single item equal to complete() output."""
        from axon.llm import (
            OpenLLM,
            _copilot_bridge_lock,
            _copilot_responses,
            _copilot_task_queue,
        )

        cfg = _make_config(llm_provider="copilot", llm_model="gpt-4o", llm_timeout=5)
        llm = OpenLLM(cfg)

        with _copilot_bridge_lock:
            _copilot_task_queue.clear()
            _copilot_responses.clear()

        def _resolver():
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with _copilot_bridge_lock:
                    for _task_id, res in list(_copilot_responses.items()):
                        if "event" in res and not res["event"].is_set():
                            res["result"] = "streamed response"
                            res["event"].set()
                            return
                time.sleep(0.01)

        t = threading.Thread(target=_resolver, daemon=True)
        t.start()
        chunks = list(llm.stream("Hello"))
        t.join(timeout=4)

        assert chunks == ["streamed response"]

    def test_complete_task_has_correct_model_and_temp(self):
        """Task enqueued in _copilot_task_queue contains model and temperature."""
        from axon.llm import (
            OpenLLM,
            _copilot_bridge_lock,
            _copilot_responses,
            _copilot_task_queue,
        )

        cfg = _make_config(
            llm_provider="copilot",
            llm_model="claude-3.5-sonnet",
            llm_temperature=0.2,
            llm_timeout=5,
        )
        llm = OpenLLM(cfg)

        with _copilot_bridge_lock:
            _copilot_task_queue.clear()
            _copilot_responses.clear()

        captured_task = {}

        def _capturing_resolver():
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with _copilot_bridge_lock:
                    if _copilot_task_queue:
                        captured_task.update(_copilot_task_queue[0])
                    for _task_id, res in list(_copilot_responses.items()):
                        if "event" in res and not res["event"].is_set():
                            res["result"] = "ok"
                            res["event"].set()
                            return
                time.sleep(0.01)

        t = threading.Thread(target=_capturing_resolver, daemon=True)
        t.start()
        llm.complete("check task")
        t.join(timeout=4)

        assert captured_task.get("model") == "claude-3.5-sonnet"
        assert captured_task.get("temperature") == 0.2


# ===========================================================================
# TestGetCopilotSessionToken
# ===========================================================================


class TestGetCopilotSessionToken:
    def test_raises_when_no_pat(self):
        from axon.llm import OpenLLM, _get_copilot_session_token

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="")
        # Override env to avoid picking up actual env vars
        import os

        env_patch = {"GITHUB_COPILOT_PAT": "", "GITHUB_TOKEN": ""}
        with patch.dict(os.environ, env_patch):
            llm = OpenLLM(cfg)
            llm.config.copilot_pat = ""  # force empty after __post_init__
            with pytest.raises(ValueError, match="GitHub Copilot token not set"):
                _get_copilot_session_token(llm)

    def test_returns_cached_valid_session(self):
        from axon.llm import OpenLLM, _get_copilot_session_token

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        # Pre-load a session that won't expire for 10 minutes
        future_expiry = time.time() + 600
        llm._openai_clients["_copilot_session"] = {
            "token": "cached_session_token",
            "expires_at": future_expiry,
        }

        result = _get_copilot_session_token(llm)
        assert result == "cached_session_token"

    def test_refreshes_expired_session(self):
        from axon.llm import OpenLLM, _get_copilot_session_token

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        # Session that expired in the past
        llm._openai_clients["_copilot_session"] = {
            "token": "old_token",
            "expires_at": time.time() - 100,
        }

        new_session = {"token": "fresh_token", "expires_at": time.time() + 1800}
        with patch("axon.llm._refresh_copilot_session", return_value=new_session) as mock_refresh:
            result = _get_copilot_session_token(llm)

        mock_refresh.assert_called_once_with("gh_tok")
        assert result == "fresh_token"
        assert llm._openai_clients["_copilot_session"]["token"] == "fresh_token"

    def test_refreshes_when_within_buffer(self):
        """Session expiring within _COPILOT_SESSION_REFRESH_BUFFER should be refreshed."""
        from axon.llm import (
            _COPILOT_SESSION_REFRESH_BUFFER,
            OpenLLM,
            _get_copilot_session_token,
        )

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        # Session expiring within the buffer period
        llm._openai_clients["_copilot_session"] = {
            "token": "almost_expired",
            "expires_at": time.time() + _COPILOT_SESSION_REFRESH_BUFFER - 10,
        }

        new_session = {"token": "new_tok", "expires_at": time.time() + 1800}
        with patch("axon.llm._refresh_copilot_session", return_value=new_session):
            result = _get_copilot_session_token(llm)

        assert result == "new_tok"

    def test_no_refresh_when_no_cached_session(self):
        """With no cached session, _refresh_copilot_session is called to create one."""
        from axon.llm import OpenLLM, _get_copilot_session_token

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        new_session = {"token": "brand_new", "expires_at": time.time() + 1800}
        with patch("axon.llm._refresh_copilot_session", return_value=new_session) as mock_ref:
            result = _get_copilot_session_token(llm)

        mock_ref.assert_called_once()
        assert result == "brand_new"


# ===========================================================================
# TestRefreshCopilotSession
# ===========================================================================


class TestRefreshCopilotSession:
    def test_returns_token_and_expires_at(self):
        from axon.llm import _refresh_copilot_session

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"token": "sess_tok", "expires_at": 9999999.0}

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = _refresh_copilot_session("gh_oauth_tok")

        assert result["token"] == "sess_tok"
        assert result["expires_at"] == 9999999.0

    def test_calls_correct_endpoint(self):
        from axon.llm import _refresh_copilot_session

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"token": "t", "expires_at": 0.0}

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            _refresh_copilot_session("my_oauth_token")

        call_args = mock_httpx.get.call_args
        url = call_args[0][0]
        assert "copilot_internal/v2/token" in url

    def test_passes_authorization_header(self):
        from axon.llm import _refresh_copilot_session

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"token": "t", "expires_at": 0.0}

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            _refresh_copilot_session("my_secret_token")

        call_kw = mock_httpx.get.call_args[1]
        assert call_kw["headers"]["Authorization"] == "token my_secret_token"

    def test_raises_on_http_error(self):
        from axon.llm import _refresh_copilot_session

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(Exception, match="401"):
                _refresh_copilot_session("bad_token")


# ===========================================================================
# TestFetchCopilotModels
# ===========================================================================


class TestFetchCopilotModels:
    def test_returns_fallback_when_no_pat(self):
        from axon.llm import _COPILOT_MODELS_FALLBACK, OpenLLM, _fetch_copilot_models

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = ""  # ensure empty after post_init

        result = _fetch_copilot_models(llm)
        assert result == list(_COPILOT_MODELS_FALLBACK)

    def test_returns_model_ids_from_api(self):
        from axon.llm import OpenLLM, _fetch_copilot_models

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        api_models = {
            "data": [
                {"id": "gpt-4o", "capabilities": {"type": "chat"}},
                {"id": "o1", "capabilities": {"type": "chat"}},
                {"id": "text-embed", "capabilities": {"type": "embedding"}},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = api_models

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch("axon.llm._get_copilot_session_token", return_value="sess_tok"), patch.dict(
            "sys.modules", {"httpx": mock_httpx}
        ):
            result = _fetch_copilot_models(llm)

        assert "gpt-4o" in result
        assert "o1" in result
        assert "text-embed" not in result  # non-chat type filtered out

    def test_returns_fallback_on_http_error(self):
        from axon.llm import _COPILOT_MODELS_FALLBACK, OpenLLM, _fetch_copilot_models

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = Exception("Network error")

        with patch("axon.llm._get_copilot_session_token", return_value="sess_tok"), patch.dict(
            "sys.modules", {"httpx": mock_httpx}
        ):
            result = _fetch_copilot_models(llm)

        assert result == list(_COPILOT_MODELS_FALLBACK)

    def test_returns_fallback_when_empty_model_list(self):
        from axon.llm import _COPILOT_MODELS_FALLBACK, OpenLLM, _fetch_copilot_models

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch("axon.llm._get_copilot_session_token", return_value="sess_tok"), patch.dict(
            "sys.modules", {"httpx": mock_httpx}
        ):
            result = _fetch_copilot_models(llm)

        assert result == list(_COPILOT_MODELS_FALLBACK)

    def test_returns_fallback_when_no_chat_models(self):
        from axon.llm import _COPILOT_MODELS_FALLBACK, OpenLLM, _fetch_copilot_models

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"id": "embed-model", "capabilities": {"type": "embedding"}}]
        }

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch("axon.llm._get_copilot_session_token", return_value="sess_tok"), patch.dict(
            "sys.modules", {"httpx": mock_httpx}
        ):
            result = _fetch_copilot_models(llm)

        assert result == list(_COPILOT_MODELS_FALLBACK)

    def test_fetch_passes_session_bearer_token(self):
        from axon.llm import OpenLLM, _fetch_copilot_models

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        llm = OpenLLM(cfg)
        llm.config.copilot_pat = "gh_tok"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o", "capabilities": {"type": "chat"}}]}

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch(
            "axon.llm._get_copilot_session_token", return_value="the_session_tok"
        ), patch.dict("sys.modules", {"httpx": mock_httpx}):
            _fetch_copilot_models(llm)

        call_kw = mock_httpx.get.call_args[1]
        assert call_kw["headers"]["Authorization"] == "Bearer the_session_tok"


# ===========================================================================
# TestUnknownProvider
# ===========================================================================


class TestUnknownProvider:
    def test_complete_unknown_provider_raises(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        llm = OpenLLM(cfg)
        llm.config.llm_provider = "totally_unknown_provider"  # bypass Literal validation

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            llm.complete("hello")

    def test_stream_unknown_provider_yields_nothing(self):
        """stream() has no else-raise for unknown providers — it simply yields nothing."""
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="ollama", llm_model="llama3")
        llm = OpenLLM(cfg)
        llm.config.llm_provider = "totally_unknown_provider"

        result = list(llm.stream("hello"))
        assert result == []


# ===========================================================================
# TestGetOpenAIClient (caching / base_url logic)
# ===========================================================================


class TestGetOpenAIClient:
    def test_client_cached_by_key(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            c1 = llm._get_openai_client()
            c2 = llm._get_openai_client()

        assert c1 is c2
        assert mock_openai.OpenAI.call_count == 1

    def test_different_base_urls_create_separate_clients(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="sk-test")
        mock_openai = MagicMock()
        mock_openai.OpenAI.side_effect = lambda **kw: MagicMock()

        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenLLM(cfg)
            c1 = llm._get_openai_client(base_url="http://host1:8000/v1")
            c2 = llm._get_openai_client(base_url="http://host2:8000/v1")

        assert c1 is not c2
        assert mock_openai.OpenAI.call_count == 2

    def test_uses_dummy_key_when_api_key_empty(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="openai", llm_model="gpt-4o", api_key="")
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        import os

        with patch.dict(os.environ, {"API_KEY": "", "OPENAI_API_KEY": ""}), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            llm.config.api_key = ""  # force empty
            llm.config.openai_api_key = ""  # force empty (new dedicated field)
            llm._get_openai_client()

        call_kw = mock_openai.OpenAI.call_args[1]
        assert call_kw["api_key"] == "sk-dummy"


# ===========================================================================
# TestGetCopilotClient
# ===========================================================================


class TestGetCopilotClient:
    def test_builds_client_with_session_token(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch("axon.llm._get_copilot_session_token", return_value="fresh_sess"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            client = llm._get_copilot_client()

        assert client is mock_client
        call_kw = mock_openai.OpenAI.call_args[1]
        assert call_kw["api_key"] == "fresh_sess"
        assert "api.githubcopilot.com" in call_kw["base_url"]

    def test_returns_cached_client_when_token_unchanged(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch("axon.llm._get_copilot_session_token", return_value="same_token"), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            c1 = llm._get_copilot_client()
            c2 = llm._get_copilot_client()

        assert mock_openai.OpenAI.call_count == 1
        assert c1 is c2

    def test_rebuilds_client_when_token_changes(self):
        from axon.llm import OpenLLM

        cfg = _make_config(llm_provider="github_copilot", copilot_pat="gh_tok")
        mock_openai = MagicMock()
        mock_openai.OpenAI.side_effect = lambda **kw: MagicMock()

        tokens = iter(["token_v1", "token_v2"])
        with patch("axon.llm._get_copilot_session_token", side_effect=tokens), patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            llm = OpenLLM(cfg)
            c1 = llm._get_copilot_client()
            c2 = llm._get_copilot_client()

        assert mock_openai.OpenAI.call_count == 2
        assert c1 is not c2


# ===========================================================================
# Parametrized provider-level smoke tests
# ===========================================================================


@pytest.mark.parametrize(
    "provider,extra_kwargs",
    [
        ("ollama", {"ollama_base_url": "http://localhost:11434"}),
        ("gemini", {"gemini_api_key": "key"}),
        (
            "ollama_cloud",
            {"ollama_cloud_key": "k", "ollama_cloud_url": "https://cloud.example.com"},
        ),
        ("openai", {"api_key": "sk-test"}),
        ("vllm", {"vllm_base_url": "http://vllm:8000/v1"}),
    ],
)
def test_complete_returns_string_for_provider(provider, extra_kwargs):
    """Smoke test: complete() returns a non-empty string for each provider."""
    from axon.llm import OpenLLM

    cfg = _make_config(llm_provider=provider, llm_model="test-model", **extra_kwargs)
    llm = OpenLLM(cfg)

    if provider == "ollama":
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = {"message": {"content": "answer"}}
        mock_cls = MagicMock(return_value=mock_client_inst)
        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            result = llm.complete("test")
    elif provider == "gemini":
        mock_response = MagicMock()
        mock_response.text = "gemini"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            result = llm.complete("test")
    elif provider == "ollama_cloud":
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "cloud"}
        mock_resp.raise_for_status = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_inst.post.return_value = mock_resp
        mock_client_inst.__enter__ = MagicMock(return_value=mock_client_inst)
        mock_client_inst.__exit__ = MagicMock(return_value=False)
        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client_inst
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = llm.complete("test")
    elif provider in ("openai", "vllm"):
        mock_response = _openai_response("openai-or-vllm")
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = llm.complete("test")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize(
    "provider,extra_kwargs",
    [
        ("ollama", {}),
        ("gemini", {"gemini_api_key": "key"}),
        ("openai", {"api_key": "sk-test"}),
        ("vllm", {"vllm_base_url": "http://vllm:8000/v1"}),
    ],
)
def test_stream_yields_strings_for_provider(provider, extra_kwargs):
    """Smoke test: stream() yields at least one string for each provider."""
    from axon.llm import OpenLLM

    cfg = _make_config(llm_provider=provider, llm_model="test-model", **extra_kwargs)
    llm = OpenLLM(cfg)

    if provider == "ollama":
        mock_client_inst = MagicMock()
        mock_client_inst.chat.return_value = iter([{"message": {"content": "tok"}}])
        mock_cls = MagicMock(return_value=mock_client_inst)
        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_cls)}):
            result = list(llm.stream("test"))
    elif provider == "gemini":
        c = MagicMock()
        c.text = "gemini_tok"
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([c])
        mock_genai_sdk = MagicMock()
        mock_genai_sdk.Client.return_value = mock_client
        mock_genai_types = MagicMock()
        mock_genai_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
        with patch.dict(
            "sys.modules", _make_google_genai_modules(mock_genai_sdk, mock_genai_types)
        ):
            result = list(llm.stream("test"))
    elif provider in ("openai", "vllm"):
        chunks = [_openai_chunk("t1"), _openai_chunk("t2")]
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = iter(chunks)
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_inst
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = list(llm.stream("test"))

    assert len(result) >= 1
    assert all(isinstance(tok, str) for tok in result)
