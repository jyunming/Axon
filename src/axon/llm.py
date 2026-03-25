"""
src/axon/llm.py

OpenLLM client and Copilot bridge helpers extracted from main.py for Phase 2 of the
Axon refactor.
"""

import logging
import threading
import uuid
from importlib import import_module

from axon.config import AxonConfig

logger = logging.getLogger("Axon")

# ---------------------------------------------------------------------------
# VS Code Copilot LLM bridge — shared state used by the FastAPI layer
# ---------------------------------------------------------------------------

_copilot_task_queue: list[dict] = []
_copilot_responses: dict[str, dict] = {}
_copilot_bridge_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Copilot OAuth / session helpers
# ---------------------------------------------------------------------------

_COPILOT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"  # GitHub Copilot Vim plugin client ID
_COPILOT_SESSION_REFRESH_BUFFER = 60  # seconds before expiry to pre-refresh


def _copilot_device_flow() -> str:
    """Run GitHub OAuth device flow and return the resulting GitHub OAuth token.

    Prints the user code and verification URL, then polls until the user
    completes authorisation in the browser.
    """
    import time

    import httpx

    resp = httpx.post(
        "https://github.com/login/device/code",
        json={"client_id": _COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"},
        headers={"Accept": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"  Open in your browser: {data['verification_uri']}")
    print(f"  Enter this code:      {data['user_code']}")
    print("  Waiting for authorisation", end="", flush=True)

    interval = data.get("interval", 5)
    deadline = time.time() + data.get("expires_in", 900)

    while time.time() < deadline:
        time.sleep(interval)
        print(".", end="", flush=True)
        token_resp = httpx.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": _COPILOT_OAUTH_CLIENT_ID,
                "device_code": data["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        token_data = token_resp.json()
        if "access_token" in token_data:
            print()
            return token_data["access_token"]
        err = token_data.get("error", "")
        if err == "slow_down":
            interval += 5
        elif err != "authorization_pending":
            print()
            raise RuntimeError(f"GitHub OAuth error: {err}")
    print()
    raise RuntimeError("GitHub OAuth device flow timed out.")


def _refresh_copilot_session(oauth_token: str) -> dict:
    """Exchange a GitHub OAuth token for a short-lived Copilot session token.

    Returns {"token": str, "expires_at": float} where expires_at is a Unix
    timestamp.  The session token is valid for ~30 minutes.
    """
    import httpx

    resp = httpx.get(
        "https://api.github.com/copilot_internal/v2/token",
        headers={
            "Authorization": f"token {oauth_token}",
            "Editor-Version": "axon/0.9.0",
            "Editor-Plugin-Version": "axon/0.9.0",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"token": data["token"], "expires_at": float(data["expires_at"])}


def _get_copilot_session_token(llm: "OpenLLM") -> str:
    """Return a valid Copilot session token, refreshing if needed."""
    import time

    oauth_token = llm.config.copilot_pat
    if not oauth_token:
        raise ValueError(
            "GitHub Copilot token not set. "
            "Run /keys set github_copilot or export GITHUB_COPILOT_PAT=<oauth_token>."
        )
    session = llm._openai_clients.get("_copilot_session")
    if not session or time.time() >= session["expires_at"] - _COPILOT_SESSION_REFRESH_BUFFER:
        session = _refresh_copilot_session(oauth_token)
        llm._openai_clients["_copilot_session"] = session
    return session["token"]


# Static fallback used when the Copilot token isn't set or the network call fails.
# The live list is fetched from https://api.githubcopilot.com/models at completion time.
_COPILOT_MODELS_FALLBACK: list[str] = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "o1",
    "o3-mini",
]


def _fetch_copilot_models(llm: "OpenLLM") -> list[str]:
    """Return chat model IDs from https://api.githubcopilot.com/models.

    Uses the Copilot session token (not the raw OAuth token / PAT) because the
    /models endpoint requires a session token.  Falls back to
    _COPILOT_MODELS_FALLBACK on any error.
    """
    try:
        if not llm.config.copilot_pat:
            return list(_COPILOT_MODELS_FALLBACK)
        session_token = _get_copilot_session_token(llm)
        import httpx

        resp = httpx.get(
            "https://api.githubcopilot.com/models",
            headers={
                "Authorization": f"Bearer {session_token}",
                "Editor-Version": "axon/0.9.0",
                "Copilot-Integration-Id": "axon",
            },
            timeout=10,
        )
        resp.raise_for_status()
        ids = [
            m["id"]
            for m in resp.json().get("data", [])
            if m.get("capabilities", {}).get("type") == "chat"
        ]
        return ids if ids else list(_COPILOT_MODELS_FALLBACK)
    except Exception:
        return list(_COPILOT_MODELS_FALLBACK)


# ---------------------------------------------------------------------------
# OpenLLM
# ---------------------------------------------------------------------------


class OpenLLM:
    """Unified LLM client supporting ollama, gemini, ollama_cloud, openai, vllm, and copilot.

    The ``copilot`` provider routes completions through the VS Code extension
    bridge (poll ``GET /llm/copilot/tasks``, submit results via
    ``POST /llm/copilot/result/<task_id>``).
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self._openai_clients: dict = {}

    def _get_gemini_sdk(self) -> tuple[object, object]:
        """Resolve and cache the Gemini SDK modules (google.genai)."""
        cached = self._openai_clients.get("_gemini_sdk")
        if cached:
            return cached
        genai_sdk = import_module("google.genai")
        genai_types = import_module("google.genai.types")
        sdk = (genai_sdk, genai_types)
        self._openai_clients["_gemini_sdk"] = sdk
        return sdk

    def _get_gemini_client(self, genai_sdk) -> object:
        """Return a cached ``google.genai.Client`` keyed by API key."""
        key = self.config.gemini_api_key or ""
        cache_key = "_gemini_client"
        if self._openai_clients.get("_gemini_client_api_key") != key:
            self._openai_clients[cache_key] = genai_sdk.Client(api_key=key)
            self._openai_clients["_gemini_client_api_key"] = key
        return self._openai_clients[cache_key]

    @staticmethod
    def _build_gemini_contents(
        prompt: str, system_prompt: str | None, history: list[dict[str, str]], is_gemma: bool
    ) -> list[dict]:
        """Build Gemini conversation payload from history + prompt."""
        contents: list[dict] = []
        for msg in history:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            mapped = "model" if role == "assistant" else "user"
            contents.append({"role": mapped, "parts": [msg.get("content", "")]})

        user_text = prompt
        if system_prompt and is_gemma:
            user_text = f"{system_prompt}\n\n{prompt}"
        contents.append({"role": "user", "parts": [user_text]})
        return contents

    def _get_openai_client(self, base_url: str = None):
        """Return a cached OpenAI client. Pass base_url for vLLM or custom endpoints."""
        cache_key = base_url or "default"
        if cache_key not in self._openai_clients:
            from openai import OpenAI

            api_key = self.config.api_key if self.config.api_key else "sk-dummy"
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_clients[cache_key] = OpenAI(**kwargs)
        return self._openai_clients[cache_key]

    def _get_copilot_client(self):
        """Return an OpenAI client authenticated with the current Copilot session token.

        The session token is obtained by exchanging the stored GitHub OAuth token
        (copilot_pat field) via the copilot_internal/v2/token endpoint.  It expires
        every ~30 minutes; this method refreshes it transparently and rebuilds the
        client only when the token changes.
        """
        from openai import OpenAI

        session_token = _get_copilot_session_token(self)
        if self._openai_clients.get("_copilot_token") != session_token:
            self._openai_clients["_copilot"] = OpenAI(
                base_url="https://api.githubcopilot.com",
                api_key=session_token,
                default_headers={
                    "Editor-Version": "axon/0.9.0",
                    "Editor-Plugin-Version": "axon/0.9.0",
                    "Copilot-Integration-Id": "axon",
                },
            )
            self._openai_clients["_copilot_token"] = session_token
        return self._openai_clients["_copilot"]

    def complete(
        self, prompt: str, system_prompt: str = None, chat_history: list[dict[str, str]] = None
    ) -> str:
        provider = self.config.llm_provider
        history = chat_history or []
        if provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add chat history
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            response = client.chat(
                model=self.config.llm_model,
                messages=messages,
                options={"temperature": self.config.llm_temperature, "num_ctx": 8192},
            )
            return response["message"]["content"]

        elif provider == "gemini":
            genai, genai_types = self._get_gemini_sdk()
            is_gemma = "gemma" in self.config.llm_model.lower()
            contents = self._build_gemini_contents(prompt, system_prompt, history, is_gemma)
            client = self._get_gemini_client(genai)
            cfg_kwargs = {
                "temperature": self.config.llm_temperature,
                "max_output_tokens": self.config.llm_max_tokens,
            }
            if system_prompt and not is_gemma:
                cfg_kwargs["system_instruction"] = system_prompt
            response = client.models.generate_content(
                model=self.config.llm_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**cfg_kwargs),
            )
            return getattr(response, "text", "") or ""

        elif provider == "ollama_cloud":
            import httpx

            headers = {
                "Authorization": f"Bearer {self.config.ollama_cloud_key}",
                "Content-Type": "application/json",
            }

            history_str = ""
            for msg in history:
                role = "Assistant" if msg["role"] == "assistant" else "User"
                history_str += f"{role}: {msg['content']}\n\n"

            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += history_str
            full_prompt += f"User: {prompt}\n\nAssistant:"

            payload = {
                "model": self.config.llm_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": self.config.llm_temperature},
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.config.ollama_cloud_url}/generate", json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()["response"]

        elif provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            # Pass base_url when pointing at an OpenAI-compatible local endpoint
            _openai_base = (
                self.config.ollama_base_url
                if self.config.ollama_base_url != "http://localhost:11434"
                else None
            )
            response = self._get_openai_client(base_url=_openai_base).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content

        elif provider == "vllm":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})

            response = self._get_openai_client(
                base_url=self.config.vllm_base_url
            ).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content

        elif provider == "github_copilot":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            response = self._get_copilot_client().chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content

        elif provider == "copilot":
            task_id = f"task_{uuid.uuid4().hex[:12]}"
            event = threading.Event()

            with _copilot_bridge_lock:
                _copilot_task_queue.append(
                    {
                        "id": task_id,
                        "prompt": prompt,
                        "history": history,
                        "system_prompt": system_prompt,
                        "model": self.config.llm_model,
                        "temperature": self.config.llm_temperature,
                        "max_tokens": self.config.llm_max_tokens,
                    }
                )
                _copilot_responses[task_id] = {"event": event, "result": None, "error": None}

            # Wait for extension to fulfill (timeout from config)
            if event.wait(timeout=self.config.llm_timeout):
                res = _copilot_responses.get(task_id)
                if not res:
                    return "Error: Task response lost."
                if res["error"]:
                    return f"Error from Copilot: {res['error']}"
                result = res["result"] or ""
                with _copilot_bridge_lock:
                    _copilot_responses.pop(task_id, None)
                return result
            else:
                with _copilot_bridge_lock:
                    _copilot_responses.pop(task_id, None)
                return "Error: Copilot LLM bridge timed out."

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def stream(
        self, prompt: str, system_prompt: str = None, chat_history: list[dict[str, str]] = None
    ):
        provider = self.config.llm_provider
        history = chat_history or []
        if provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            stream_resp = client.chat(
                model=self.config.llm_model,
                messages=messages,
                stream=True,
                options={"temperature": self.config.llm_temperature, "num_ctx": 8192},
            )
            for chunk in stream_resp:
                yield chunk["message"]["content"]

        elif provider == "gemini":
            genai, genai_types = self._get_gemini_sdk()
            is_gemma = "gemma" in self.config.llm_model.lower()
            contents = self._build_gemini_contents(prompt, system_prompt, history, is_gemma)
            client = self._get_gemini_client(genai)
            cfg_kwargs = {
                "temperature": self.config.llm_temperature,
                "max_output_tokens": self.config.llm_max_tokens,
            }
            if system_prompt and not is_gemma:
                cfg_kwargs["system_instruction"] = system_prompt
            response = client.models.generate_content_stream(
                model=self.config.llm_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**cfg_kwargs),
            )
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text

        elif provider == "ollama_cloud":
            import json

            import httpx

            headers = {
                "Authorization": f"Bearer {self.config.ollama_cloud_key}",
                "Content-Type": "application/json",
            }

            history_str = ""
            for msg in history:
                role = "Assistant" if msg["role"] == "assistant" else "User"
                history_str += f"{role}: {msg['content']}\n\n"

            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += history_str
            full_prompt += f"User: {prompt}\n\nAssistant:"

            payload = {
                "model": self.config.llm_model,
                "prompt": full_prompt,
                "stream": True,
                "options": {"temperature": self.config.llm_temperature},
            }
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{self.config.ollama_cloud_url}/generate",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]

        elif provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            # Pass base_url when pointing at an OpenAI-compatible local endpoint
            _openai_base = (
                self.config.ollama_base_url
                if self.config.ollama_base_url != "http://localhost:11434"
                else None
            )
            stream = self._get_openai_client(base_url=_openai_base).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
                timeout=self.config.llm_timeout,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        elif provider == "vllm":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})

            stream = self._get_openai_client(
                base_url=self.config.vllm_base_url
            ).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
                timeout=self.config.llm_timeout,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        elif provider == "github_copilot":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            stream = self._get_copilot_client().chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
                timeout=self.config.llm_timeout,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        elif provider == "copilot":
            # For streaming, we'll just use the blocking complete() logic but yield it as one chunk.
            # True streaming via the bridge would require a more complex WebSocket setup.
            yield self.complete(prompt, system_prompt, chat_history)
