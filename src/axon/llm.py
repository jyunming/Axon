"""


src/axon/llm.py


OpenLLM client and Copilot bridge helpers extracted from main.py for Phase 2 of the


Axon refactor.


"""


import logging
import threading
import uuid
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    _AXON_VERSION = _pkg_version("axon-rag")
except PackageNotFoundError:
    _AXON_VERSION = "0.0.0+dev"

from axon.config import AxonConfig

logger = logging.getLogger("Axon")


def _openai_tool_to_gemini_declaration(tool_dict: dict, genai_types) -> object:
    """Convert an OpenAI-format tool schema dict to a Gemini FunctionDeclaration."""
    fn = tool_dict.get("function", {})

    def _schema(s: dict) -> object:
        if not isinstance(s, dict):
            return genai_types.Schema(type=genai_types.Type.STRING)
        _type_map = {
            "string": genai_types.Type.STRING,
            "integer": genai_types.Type.INTEGER,
            "number": genai_types.Type.NUMBER,
            "boolean": genai_types.Type.BOOLEAN,
            "array": genai_types.Type.ARRAY,
            "object": genai_types.Type.OBJECT,
        }
        t = _type_map.get((s.get("type") or "string").lower(), genai_types.Type.STRING)
        kwargs: dict = {"type": t}
        if s.get("description"):
            kwargs["description"] = s["description"]
        if s.get("enum"):
            kwargs["enum"] = list(s["enum"])
        if s.get("properties"):
            kwargs["properties"] = {k: _schema(v) for k, v in s["properties"].items()}
        if s.get("items"):
            kwargs["items"] = _schema(s["items"])
        if s.get("required"):
            kwargs["required"] = list(s["required"])
        return genai_types.Schema(**kwargs)

    params = fn.get("parameters", {})
    param_schema = _schema(params) if params else None
    return genai_types.FunctionDeclaration(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=param_schema,
    )


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
            "Editor-Version": f"axon/{_AXON_VERSION}",
            "Editor-Plugin-Version": f"axon/{_AXON_VERSION}",
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
                "Editor-Version": f"axon/{_AXON_VERSION}",
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

    """Unified LLM client supporting ollama, gemini, ollama_cloud, openai, grok, vllm, and copilot.


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
        last_was_fn_response = False

        for msg in history:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            # Tool-call rounds stored by run_agent_loop — convert to native Gemini parts.
            # After adding function_response the model responds automatically; adding another
            # user-text turn here would create two consecutive "user" messages which Gemini rejects.
            if "__tool_calls__" in msg:
                calls = msg["__tool_calls__"]  # list of (name, args, result[, thought_sig])
                fc_parts = []
                for call in calls:
                    n, a = call[0], call[1]
                    thought_sig = call[3] if len(call) > 3 else None
                    part: dict = {"function_call": {"name": n, "args": a}}
                    if thought_sig is not None:
                        # Gemini thinking models require the original thought_signature to be
                        # echoed back verbatim in the function_call history part.
                        part["thought_signature"] = thought_sig
                    fc_parts.append(part)
                contents.append({"role": "model", "parts": fc_parts})
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": call[0],
                                    "response": {"result": call[2]},
                                }
                            }
                            for call in calls
                        ],
                    }
                )
                last_was_fn_response = True
                continue

            last_was_fn_response = False
            mapped = "model" if role == "assistant" else "user"
            contents.append({"role": mapped, "parts": [{"text": msg.get("content", "")}]})

        # When the last history entry was a function_response, Gemini responds to it automatically.
        # Adding another user message would produce invalid consecutive user turns.
        if not last_was_fn_response:
            user_text = prompt
            if system_prompt and is_gemma:
                user_text = f"{system_prompt}\n\n{prompt}"
            contents.append({"role": "user", "parts": [{"text": user_text}]})

        return contents

    def _get_openai_client(self, base_url: str = None, api_key: str = None):
        """Return a cached OpenAI client. Pass base_url for vLLM or custom endpoints."""

        cache_key = base_url or "default"

        if cache_key not in self._openai_clients:
            from openai import OpenAI

            resolved_key = (
                api_key or self.config.openai_api_key or self.config.api_key or "sk-dummy"
            )

            kwargs = {"api_key": resolved_key}

            if base_url:
                kwargs["base_url"] = base_url

            self._openai_clients[cache_key] = OpenAI(**kwargs)

        return self._openai_clients[cache_key]

    def _get_grok_client(self):
        """Return a cached OpenAI-compatible client for xAI Grok."""

        cache_key = "_grok"

        if cache_key not in self._openai_clients:
            from openai import OpenAI

            if not self.config.grok_api_key:
                raise ValueError(
                    "Grok API key not set. "
                    "Export XAI_API_KEY=<key> or set llm.grok_api_key in config.yaml."
                )

            self._openai_clients[cache_key] = OpenAI(
                api_key=self.config.grok_api_key,
                base_url="https://api.x.ai/v1",
            )

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
                    "Editor-Version": f"axon/{_AXON_VERSION}",
                    "Editor-Plugin-Version": f"axon/{_AXON_VERSION}",
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

        elif provider == "grok":
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            response = self._get_grok_client().chat.completions.create(
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

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict] | None = None,
        system_prompt: str = None,
        chat_history: list[dict[str, str]] = None,
    ):
        """Return a list of ToolCall namedtuples or a plain text string.

        Implements native function calling for Gemini, Ollama, and OpenAI-compatible
        providers (openai, vllm, github_copilot, grok).  Falls back to plain text
        completion for providers that do not support function calling.
        """
        from axon.agent import ToolCall  # local import to avoid circular

        provider = self.config.llm_provider

        # ------------------------------------------------------------------ Gemini
        if provider == "gemini":
            genai, genai_types = self._get_gemini_sdk()
            is_gemma = "gemma" in self.config.llm_model.lower()
            history = chat_history or []
            contents = self._build_gemini_contents(prompt, system_prompt, history, is_gemma)
            client = self._get_gemini_client(genai)
            cfg_kwargs: dict = {
                "temperature": self.config.llm_temperature,
                "max_output_tokens": self.config.llm_max_tokens,
            }
            if system_prompt and not is_gemma:
                cfg_kwargs["system_instruction"] = system_prompt
            # Register tools with the Gemini API so it follows the function-calling protocol
            if tools and not is_gemma:
                try:
                    decls = [_openai_tool_to_gemini_declaration(t, genai_types) for t in tools]
                    cfg_kwargs["tools"] = [genai_types.Tool(function_declarations=decls)]
                except Exception:
                    pass
            response = client.models.generate_content(
                model=self.config.llm_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**cfg_kwargs),
            )
            # Extract function_call parts before touching .text (which raises SDK warning)
            try:
                parts = response.candidates[0].content.parts
                calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]
                if calls:
                    result: list[ToolCall] = []
                    for part in calls:
                        fc = part.function_call
                        args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                        # thought_signature is required for Gemini thinking models when
                        # replaying function_call parts in the conversation history.
                        thought_sig = getattr(part, "thought_signature", None)
                        result.append(
                            ToolCall(name=fc.name, args=args, thought_signature=thought_sig)
                        )
                    return result
            except Exception:
                pass
            return getattr(response, "text", "") or ""

        # ------------------------------------------------------------------ Ollama
        if provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in chat_history or []:
                if "__tool_calls__" in msg:
                    for _name, _args, result_str in msg["__tool_calls__"]:
                        messages.append({"role": "tool", "content": result_str})
                    continue
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg.get("content", "")})
            messages.append({"role": "user", "content": prompt})
            kwargs: dict = {
                "model": self.config.llm_model,
                "messages": messages,
                "options": {"temperature": self.config.llm_temperature, "num_ctx": 8192},
            }
            if tools:
                kwargs["tools"] = tools
            response = client.chat(**kwargs)
            msg_obj = response["message"]
            if msg_obj.get("tool_calls"):
                return [
                    ToolCall(
                        name=tc["function"]["name"],
                        args=tc["function"].get("arguments") or {},
                    )
                    for tc in msg_obj["tool_calls"]
                ]
            return msg_obj.get("content", "") or ""

        # ------------------------------------------------------------------ OpenAI-compatible
        if provider in ("openai", "vllm", "github_copilot", "grok"):
            import json as _json

            _client = {
                "openai": lambda: self._get_openai_client(),
                "vllm": lambda: self._get_openai_client(base_url=self.config.vllm_base_url),
                "github_copilot": lambda: self._get_copilot_client(),
                "grok": lambda: self._get_grok_client(),
            }[provider]()
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in chat_history or []:
                if "__tool_calls__" in msg:
                    for _name, _args, result_str in msg["__tool_calls__"]:
                        messages.append(
                            {"role": "tool", "tool_call_id": "0", "content": result_str}
                        )
                    continue
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg.get("content", "")})
            messages.append({"role": "user", "content": prompt})
            kwargs: dict = {
                "model": self.config.llm_model,
                "messages": messages,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "timeout": self.config.llm_timeout,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = _client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.message.tool_calls:
                return [
                    ToolCall(
                        name=tc.function.name,
                        args=_json.loads(tc.function.arguments or "{}"),
                    )
                    for tc in choice.message.tool_calls
                ]
            return choice.message.content or ""

        # ------------------------------------------------------------------ Fallback
        return self.complete(prompt, system_prompt=system_prompt, chat_history=chat_history)

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

        elif provider == "grok":
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            stream = self._get_grok_client().chat.completions.create(
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

        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
