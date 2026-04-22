import html
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from importlib import import_module
from pathlib import Path

try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from axon.main import AxonBrain  # noqa: E402
from axon.projects import (  # noqa: E402
    delete_project,
    ensure_project,
    get_active_project,
    get_maintenance_state,
    list_projects,
)

# Setup logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Axon",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — supplements .streamlit/config.toml theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Hide Streamlit chrome ── */
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── Sidebar: fixed width, pinned, no collapse ── */
    [data-testid="stSidebar"] {
        min-width: 272px !important;
        max-width: 272px !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    /* Tighten the default top padding */
    [data-testid="stSidebar"] > div:first-child {
        padding: 0.75rem 0.85rem 0.75rem !important;
    }
    /* Moderate vertical gaps between sidebar elements */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }

    /* ── Sidebar section headers ── */
    .sb-section {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.50);
        margin: 16px 0 6px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.10);
    }

    /* ── All sidebar buttons: list-row style ── */
    [data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        min-height: 32px !important;
        font-size: 0.83rem !important;
        color: rgba(228,228,231,0.60) !important;
        width: 100% !important;
        transition: background 0.2s, color 0.2s !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(228,228,231,0.92) !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── Active session: primary button styled as highlighted row ── */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(139,92,246,0.18) !important;
        border-left: 2px solid #a78bfa !important;
        border-radius: 0 6px 6px 0 !important;
        color: #e4e4e7 !important;
        font-weight: 500 !important;
        padding-left: 6px !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: rgba(139,92,246,0.25) !important;
    }

    /* ── New Chat: secondary button gets a subtle outline ── */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: rgba(139,92,246,0.12) !important;
        border: 1px solid rgba(139,92,246,0.35) !important;
        border-radius: 7px !important;
        color: #c4b5fd !important;
        font-weight: 600 !important;
        justify-content: center !important;
        font-size: 0.83rem !important;
        min-height: 34px !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(139,92,246,0.22) !important;
        border-color: rgba(139,92,246,0.55) !important;
        color: #ddd6fe !important;
    }

    /* ── Small action buttons (Clear, Delete, Summarize) ── */
    [data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
        font-size: 0.77rem !important;
        color: rgba(228,228,231,0.62) !important;
        padding: 3px 6px !important;
        min-height: 28px !important;
        justify-content: center !important;
        transition: background 0.2s, color 0.2s !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
        color: rgba(228,228,231,0.92) !important;
        background: rgba(255,255,255,0.07) !important;
    }

    /* ── Sidebar widget labels — compact, readable ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p {
        font-size: 0.79rem !important;
        color: rgba(228,228,231,0.65) !important;
        margin-bottom: 2px !important;
    }

    /* ── Sidebar expanders ── */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        color: rgba(228,228,231,0.62) !important;
        padding: 4px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        color: rgba(228,228,231,0.92) !important;
    }

    /* ── Sidebar widgets: match sidebar background (#1c1c2a) ── */
    [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
        background-color: #1c1c2a !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background-color: #1c1c2a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] {
        background-color: #1c1c2a !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] > div {
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
        background-color: #1c1c2a !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
    }

    /* ── Chat messages — glassmorphism card style ── */
    [data-testid="stChatMessage"] {
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(167, 139, 250, 0.12) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        background: rgba(255,255,255,0.02) !important;
        box-shadow: 0 2px 12px rgba(139, 92, 246, 0.06) !important;
    }
    [data-testid="stChatMessage"] p { line-height: 1.7 !important; }

    /* ── Code colour ── */
    code { color: #a78bfa !important; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #71717a; }

    /* ── Status bar padding — avoid overlap ── */
    .main .block-container { padding-bottom: 48px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fallback model lists
# ---------------------------------------------------------------------------
_FALLBACK_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
]


@st.cache_data(ttl=3600)
def _list_gemini_models(api_key: str) -> list:
    """Fetch available Gemini/Gemma models via google.genai. Cached 1 h."""
    if not api_key:
        return _FALLBACK_GEMINI_MODELS
    try:
        genai_sdk = import_module("google.genai")
        client = genai_sdk.Client(api_key=api_key)
        models = []
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = set(getattr(m, "supported_actions", []) or [])
            if name and ("generateContent" in actions or "generate_content" in actions):
                models.append(name)
        return sorted(models) if models else _FALLBACK_GEMINI_MODELS
    except Exception:
        return _FALLBACK_GEMINI_MODELS


def _list_ollama_cloud_models(cloud_key: str, cloud_url: str) -> list:
    """Fetch models from the Ollama Cloud endpoint."""
    try:
        import httpx

        headers = {"Authorization": f"Bearer {cloud_key}"}
        resp = httpx.get(f"{cloud_url}/tags", headers=headers, timeout=8.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


@st.cache_data(ttl=5)
def _cached_list_projects() -> list:
    """List projects with a short 5-second TTL so new projects appear quickly."""
    return list_projects()


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
SESSIONS_FILE = "sessions.json"


def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sessions(sessions_dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions_dict, f, indent=4)


# ---------------------------------------------------------------------------
# Source card renderer
# ---------------------------------------------------------------------------
def _render_source_card(i: int, doc: dict):
    """Render a rich source card with icon, score badge, and text preview."""
    if doc.get("is_web"):
        title = doc.get("metadata", {}).get("title", doc["id"])
        st.markdown(f"**🌐 [{i + 1}] [{title}]({doc['id']})**")
    else:
        src = doc.get("id", "")
        ext = Path(src).suffix.lower() if src else ""
        icon = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "📜",
            ".md": "📝",
            ".txt": "📄",
            ".pdf": "📕",
            ".csv": "📊",
            ".json": "🔧",
            ".html": "🌐",
            ".png": "🖼",
            ".jpg": "🖼",
            ".jpeg": "🖼",
        }.get(ext, "📄")
        score = doc.get("vector_score", doc.get("score", 0))
        if score >= 0.8:
            score_color = "#4ade80"
        elif score >= 0.5:
            score_color = "#fb923c"
        else:
            score_color = "#f87171"
        score_badge = (
            f"<span style='background:rgba(0,0,0,0.3);border:1px solid {score_color}33;"
            f"color:{score_color};border-radius:4px;padding:1px 6px;font-size:0.75rem;"
            f"font-weight:600;'>{score:.3f}</span>"
        )
        st.markdown(
            f"**{icon} [{i + 1}] `{html.escape(src)}`** {score_badge}",
            unsafe_allow_html=True,
        )
    text = doc["text"][:600] + ("…" if len(doc["text"]) > 600 else "")
    st.code(text, language=None)
    st.divider()


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()

    if not st.session_state.sessions:
        default_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.sessions = {
            default_id: {
                "name": f"Session {timestamp}",
                "messages": [],
                "created_at": timestamp,
            }
        }
        st.session_state.current_session_id = default_id
    else:
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]

if "brain" not in st.session_state:
    try:
        with st.spinner("Initializing Brain…"):
            st.session_state.brain = AxonBrain()
        # Seed embedding baseline so the hot-swap warning only fires on user changes
        st.session_state["_emb_provider"] = st.session_state.brain.config.embedding_provider
        st.session_state["_emb_model"] = st.session_state.brain.config.embedding_model
        # Store active project
        st.session_state["active_project"] = get_active_project()
    except Exception as e:
        st.error(
            f"Failed to initialize Axon Brain: {e}\n\n"
            "Please check your `config.yaml` and ensure all dependencies are installed. "
            "Try running `axon` from the terminal to diagnose the issue."
        )
        st.stop()

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

if "show_new_project" not in st.session_state:
    st.session_state.show_new_project = False

if "confirm_delete_project" not in st.session_state:
    st.session_state.confirm_delete_project = False

current_session_id = st.session_state.current_session_id
current_session = st.session_state.sessions[current_session_id]
messages = current_session["messages"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if "brain" not in st.session_state:
    st.error("Axon Brain failed to initialize. Please check your configuration and restart.")
    st.stop()

with st.sidebar:
    config = st.session_state.brain.config

    # ── App title ──
    st.markdown(
        '<div style="font-size:1rem;font-weight:700;color:#e4e4e7;margin-bottom:8px;">🧠 Axon</div>',
        unsafe_allow_html=True,
    )

    # =========================================================================
    # PROJECT HUB
    # =========================================================================
    st.markdown('<div class="sb-section">Projects</div>', unsafe_allow_html=True)

    # Load project list (cached, TTL=5s)
    all_projects = _cached_list_projects()
    project_names = [p["name"] for p in all_projects]
    if not project_names:
        project_names = ["default"]
    if "default" not in project_names:
        project_names = ["default"] + project_names

    active_proj = st.session_state.get("active_project", "default")
    active_proj_idx = project_names.index(active_proj) if active_proj in project_names else 0

    selected_proj = st.selectbox(
        "Project",
        options=project_names,
        index=active_proj_idx,
        label_visibility="collapsed",
        key="project_selectbox",
    )

    # Switch project on change
    if selected_proj != active_proj:
        try:
            st.session_state.brain.switch_project(selected_proj)
            st.session_state["active_project"] = selected_proj
            st.toast(f"📁 Switched to '{selected_proj}'")
            active_proj = selected_proj
            st.rerun()
        except Exception as e:
            st.error(f"Failed to switch project: {e}")

    # Maintenance state banner
    if active_proj != "default":
        try:
            _maint_state = get_maintenance_state(active_proj)
        except Exception:
            _maint_state = "normal"
        if _maint_state == "draining":
            st.warning(
                "⏳ **Draining** — this project is draining for maintenance. "
                "New writes are blocked; in-flight writes are completing.",
                icon=None,
            )
        elif _maint_state == "readonly":
            st.warning(
                "🔒 **Read-only** — writes are disabled for maintenance.",
                icon=None,
            )
        elif _maint_state == "offline":
            st.error(
                "🚫 **Offline** — this project is under maintenance. " "All operations are blocked.",
                icon=None,
            )

    # New / Delete project buttons
    ph_col1, ph_col2 = st.columns(2)
    with ph_col1:
        if st.button("＋ New", use_container_width=True, type="tertiary", key="toggle_new_project"):
            st.session_state.show_new_project = not st.session_state.show_new_project
            st.session_state.confirm_delete_project = False

    with ph_col2:
        delete_disabled = active_proj == "default"
        if st.button(
            "🗑 Delete",
            use_container_width=True,
            type="tertiary",
            key="toggle_delete_project",
            disabled=delete_disabled,
        ):
            st.session_state.confirm_delete_project = not st.session_state.confirm_delete_project
            st.session_state.show_new_project = False

    # New project form
    if st.session_state.show_new_project:
        new_proj_name = st.text_input(
            "Project name",
            placeholder="e.g. my-research",
            key="new_proj_name_input",
            label_visibility="collapsed",
        )
        new_proj_desc = st.text_input(
            "Description (optional)",
            placeholder="Short description",
            key="new_proj_desc_input",
            label_visibility="collapsed",
        )
        if st.button("Create", use_container_width=True, type="tertiary", key="create_project_btn"):
            if new_proj_name:
                try:
                    ensure_project(new_proj_name, new_proj_desc)
                    st.session_state.brain.switch_project(new_proj_name)
                    st.session_state["active_project"] = new_proj_name
                    st.session_state.show_new_project = False
                    _cached_list_projects.clear()
                    st.toast(f"✅ Project '{new_proj_name}' created")
                    st.rerun()
                except ValueError as e:
                    st.error(f"Invalid project name: {e}")
                except Exception as e:
                    st.error(f"Could not create project: {e}")
            else:
                st.warning("Please enter a project name.")

    # Delete project confirmation
    if st.session_state.confirm_delete_project and active_proj != "default":
        st.warning(f"Delete project '{active_proj}' and all its data?")
        dc1, dc2 = st.columns(2)
        if dc1.button("Yes", use_container_width=True, type="tertiary", key="confirm_del_yes"):
            try:
                delete_project(active_proj)
                st.session_state.brain.switch_project("default")
                st.session_state["active_project"] = "default"
                st.session_state.confirm_delete_project = False
                _cached_list_projects.clear()
                st.toast(f"🗑 Project '{active_proj}' deleted")
                st.rerun()
            except ValueError as e:
                st.error(f"Cannot delete: {e}")
                st.session_state.confirm_delete_project = False
            except Exception as e:
                st.error(f"Delete failed: {e}")
                st.session_state.confirm_delete_project = False
        if dc2.button("No", use_container_width=True, type="tertiary", key="confirm_del_no"):
            st.session_state.confirm_delete_project = False
            st.rerun()

    # =========================================================================
    # CONVERSATIONS
    # =========================================================================
    st.markdown('<div class="sb-section">Conversations</div>', unsafe_allow_html=True)

    if st.button("＋  New Chat", use_container_width=True, type="secondary", key="new_chat"):
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {
            "name": "New Chat",
            "messages": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        save_sessions(st.session_state.sessions)
        st.session_state.current_session_id = new_id
        st.rerun()

    # Session list — newest first
    for sid, sess in reversed(list(st.session_state.sessions.items())):
        label = sess["name"] if len(sess["name"]) <= 27 else sess["name"][:24] + "…"
        is_active = sid == current_session_id
        btn_type = "primary" if is_active else "tertiary"
        if st.button(label, use_container_width=True, type=btn_type, key=f"sess_{sid}"):
            if not is_active:
                st.session_state.current_session_id = sid
                st.rerun()

    # Active session actions — Clear, Summarize, Delete
    if st.session_state.confirm_clear:
        st.caption("Clear all messages?")
        ca, cb = st.columns(2)
        if ca.button("Yes", use_container_width=True, type="tertiary", key="confirm_yes"):
            st.session_state.sessions[current_session_id]["messages"] = []
            save_sessions(st.session_state.sessions)
            st.session_state.confirm_clear = False
            st.rerun()
        if cb.button("No", use_container_width=True, type="tertiary", key="confirm_no"):
            st.session_state.confirm_clear = False
            st.rerun()
    else:
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            if st.button("🧹 Clear", use_container_width=True, type="tertiary", key="clear_chat"):
                st.session_state.confirm_clear = True
                st.rerun()
        with ac2:
            if st.button(
                "✦ Sum", use_container_width=True, type="tertiary", key="summarize_history"
            ):
                if messages:
                    formatted = "\n".join(
                        f"{m['role'].capitalize()}: {m['content']}" for m in messages
                    )
                    try:
                        summary = st.session_state.brain.llm.complete(
                            f"Summarize this conversation in 2-3 sentences:\n\n{formatted}"
                        )
                        st.session_state.sessions[current_session_id]["messages"] = [
                            {"role": "assistant", "content": f"[Summary] {summary}"}
                        ]
                        save_sessions(st.session_state.sessions)
                        st.toast("🗜 History summarized")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Summarize failed: {e}")
        with ac3:
            if st.button("🗑 Delete", use_container_width=True, type="tertiary", key="delete_sess"):
                if len(st.session_state.sessions) > 1:
                    del st.session_state.sessions[current_session_id]
                    save_sessions(st.session_state.sessions)
                    st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]
                    st.rerun()
                else:
                    st.warning("Cannot delete the only session.")

    # =========================================================================
    # MODEL & SETTINGS (collapsed — show active model as status line)
    # =========================================================================
    provider_labels = {
        "ollama": "🤖 Ollama",
        "gemini": "✨ Gemini",
        "ollama_cloud": "☁️ Ollama Cloud",
        "openai": "🔑 OpenAI",
        "vllm": "⚡ vLLM",
        "github_copilot": "🐙 GitHub Copilot",
    }
    _model_short = config.llm_model[:22] + "…" if len(config.llm_model) > 22 else config.llm_model
    st.caption(f"{provider_labels.get(config.llm_provider, config.llm_provider)} · {_model_short}")

    with st.expander("⚙ Model & Settings"):
        st.caption("LLM")
        llm_providers = list(provider_labels.keys())
        current_provider_idx = (
            llm_providers.index(config.llm_provider) if config.llm_provider in llm_providers else 0
        )
        config.llm_provider = st.selectbox(
            "Provider",
            options=llm_providers,
            format_func=lambda x: provider_labels[x],
            index=current_provider_idx,
            label_visibility="collapsed",
        )

        if config.llm_provider == "ollama":
            config.ollama_base_url = st.text_input(
                "Ollama URL",
                value=config.ollama_base_url,
                label_visibility="collapsed",
                placeholder="http://localhost:11434",
            )
            ollama_models = []
            ollama_conn_error = None
            try:
                from ollama import Client

                client = Client(host=config.ollama_base_url)
                ollama_models = [
                    m.model
                    for m in client.list().models
                    if not m.model.startswith("embeddinggemma")
                ]
            except Exception as exc:
                ollama_conn_error = str(exc)

            if ollama_models:
                current_idx = (
                    ollama_models.index(config.llm_model)
                    if config.llm_model in ollama_models
                    else 0
                )
                config.llm_model = st.selectbox(
                    "Model", ollama_models, index=current_idx, label_visibility="collapsed"
                )
            else:
                config.llm_model = st.text_input(
                    "Model",
                    config.llm_model,
                    label_visibility="collapsed",
                    placeholder="model name",
                )
                if ollama_conn_error:
                    st.warning(
                        f"Cannot connect to Ollama at `{config.ollama_base_url}`. "
                        "Run `ollama serve` to start the server, then pull a model with "
                        "`ollama pull llama3.2:3b`."
                    )
                else:
                    st.warning("No models found. Pull a model with: `ollama pull llama3.2:3b`")

            with st.expander("⬇ Pull model"):
                pull_name = st.text_input(
                    "Model name",
                    placeholder="e.g. llama3.2:3b",
                    key="pull_model_name",
                    label_visibility="collapsed",
                )
                if st.button("Pull", type="tertiary", key="do_pull"):
                    try:
                        from ollama import Client as OllamaClient

                        pull_client = OllamaClient(host=config.ollama_base_url)
                        prog = st.progress(0, text=f"Pulling {pull_name}...")
                        for resp in pull_client.pull(pull_name, stream=True):
                            if hasattr(resp, "completed") and hasattr(resp, "total") and resp.total:
                                pct = min(int(resp.completed / resp.total * 100), 100)
                                mb_done = resp.completed // 1024 // 1024
                                mb_total = resp.total // 1024 // 1024
                                prog.progress(
                                    pct,
                                    text=f"Pulling {pull_name}... {mb_done}MB/{mb_total}MB",
                                )
                        prog.progress(100, text="Done!")
                        st.toast(f"⬇ Model '{pull_name}' ready")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pull failed: {e}")

        elif config.llm_provider == "gemini":
            config.gemini_api_key = st.text_input(
                "API Key",
                value=config.gemini_api_key,
                type="password",
                label_visibility="collapsed",
                placeholder="Gemini API key",
            )
            gemini_models = _list_gemini_models(config.gemini_api_key)
            config.llm_model = st.selectbox(
                "Model",
                gemini_models,
                label_visibility="collapsed",
                index=(
                    0
                    if config.llm_model not in gemini_models
                    else gemini_models.index(config.llm_model)
                ),
            )

        elif config.llm_provider == "ollama_cloud":
            config.ollama_cloud_key = st.text_input(
                "API Key",
                value=config.ollama_cloud_key,
                type="password",
                label_visibility="collapsed",
                placeholder="Ollama Cloud key",
            )
            col_m, col_f = st.columns([4, 1])
            with col_m:
                config.llm_model = st.text_input(
                    "Model",
                    config.llm_model,
                    label_visibility="collapsed",
                    placeholder="model name",
                )
            with col_f:
                if st.button("↻", type="tertiary", help="Fetch models from Ollama Cloud"):
                    fetched = _list_ollama_cloud_models(
                        config.ollama_cloud_key, config.ollama_cloud_url
                    )
                    if fetched:
                        st.session_state["_ollama_cloud_models"] = fetched
                        st.rerun()
                    else:
                        st.warning("Could not fetch models.")
            if st.session_state.get("_ollama_cloud_models"):
                config.llm_model = st.selectbox(
                    "Available",
                    st.session_state["_ollama_cloud_models"],
                    label_visibility="collapsed",
                )

        elif config.llm_provider == "openai":
            config.api_key = st.text_input(
                "API Key",
                value=config.api_key,
                type="password",
                label_visibility="collapsed",
                placeholder="OpenAI API key",
            )
            config.llm_model = st.text_input(
                "Model",
                config.llm_model,
                label_visibility="collapsed",
                placeholder="e.g. gpt-4o",
            )

        elif config.llm_provider == "vllm":
            config.vllm_base_url = st.text_input(
                "Base URL",
                value=config.vllm_base_url,
                label_visibility="collapsed",
                placeholder="http://localhost:8000/v1",
            )
            config.llm_model = st.text_input(
                "Model",
                config.llm_model,
                label_visibility="collapsed",
                placeholder="e.g. mistralai/Mistral-7B-Instruct-v0.2",
            )

        elif config.llm_provider == "github_copilot":
            _pat = st.text_input(
                "GitHub OAuth Token",
                value=config.copilot_pat,
                type="password",
                help="GitHub OAuth token (not a PAT). "
                "Obtain it by running '/keys set github_copilot' in the REPL "
                "or set GITHUB_COPILOT_PAT env var.",
            )
            if _pat != config.copilot_pat:
                config.copilot_pat = _pat
                _brain = st.session_state.brain
                if hasattr(_brain.llm, "_openai_clients"):
                    _brain.llm._openai_clients.pop("_copilot", None)
                # Invalidate cached model list so we re-fetch with the new token
                st.session_state.pop("_copilot_model_list", None)

            # Fetch the live model list once per session (or after PAT change).
            if "_copilot_model_list" not in st.session_state:
                if config.copilot_pat:
                    with st.spinner("Fetching Copilot model list…"):
                        try:
                            from axon.main import _fetch_copilot_models

                            st.session_state["_copilot_model_list"] = _fetch_copilot_models(
                                st.session_state.brain.llm
                            )
                        except Exception:
                            st.session_state["_copilot_model_list"] = None

            from axon.main import _COPILOT_MODELS_FALLBACK

            _copilot_models = st.session_state.get("_copilot_model_list") or list(
                _COPILOT_MODELS_FALLBACK
            )
            _copilot_model = st.selectbox(
                "Model",
                options=_copilot_models
                + ([config.llm_model] if config.llm_model not in _copilot_models else []),
                index=(
                    _copilot_models.index(config.llm_model)
                    if config.llm_model in _copilot_models
                    else 0
                ),
                label_visibility="collapsed",
            )
            if _copilot_model != config.llm_model:
                config.llm_model = _copilot_model
                from axon.main import OpenLLM

                st.session_state.brain.llm = OpenLLM(config)

        # ── Embedding ──
        st.caption("Embedding")
        _EMB_PROVIDER_LABELS = {
            "sentence_transformers": "🤗 Sentence Transformers",
            "ollama": "🤖 Ollama (local)",
            "fastembed": "⚡ FastEmbed",
            "openai": "🔑 OpenAI-compatible",
        }
        _EMB_MODELS: dict = {
            "sentence_transformers": [
                "all-MiniLM-L6-v2",
                "BAAI/bge-large-en",
                "all-mpnet-base-v2",
            ],
            "ollama": ["nomic-embed-text", "mxbai-embed-large"],
            "fastembed": ["BAAI/bge-small-en-v1.5", "BAAI/bge-m3"],
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
        }
        emb_providers = list(_EMB_PROVIDER_LABELS.keys())
        config.embedding_provider = st.selectbox(
            "Embedding provider",
            options=emb_providers,
            format_func=lambda x: _EMB_PROVIDER_LABELS[x],
            index=(
                emb_providers.index(config.embedding_provider)
                if config.embedding_provider in emb_providers
                else 0
            ),
            label_visibility="collapsed",
        )
        if config.embedding_provider == "ollama" and config.llm_provider != "ollama":
            config.ollama_base_url = st.text_input(
                "Ollama URL (embed)",
                value=config.ollama_base_url,
                label_visibility="collapsed",
                placeholder="http://localhost:11434",
            )
        suggestions = _EMB_MODELS.get(config.embedding_provider, [])
        if suggestions and config.embedding_model not in suggestions:
            config.embedding_model = suggestions[0]
        if suggestions:
            config.embedding_model = st.selectbox(
                "Embedding model",
                options=suggestions,
                index=suggestions.index(config.embedding_model),
                label_visibility="collapsed",
            )
        else:
            config.embedding_model = st.text_input(
                "Embedding model", value=config.embedding_model, label_visibility="collapsed"
            )

        # Hot-swap brain.embedding when provider or model changes
        _prev_emb_provider = st.session_state.get("_emb_provider", config.embedding_provider)
        _prev_emb_model = st.session_state.get("_emb_model", config.embedding_model)
        if (
            config.embedding_provider != _prev_emb_provider
            or config.embedding_model != _prev_emb_model
        ):
            if st.session_state.get("ingested_files"):
                st.caption(
                    "⚠️ Changing embedding model after ingestion may cause dimension errors. "
                    "Clear ChromaDB data or start fresh if queries fail."
                )
            with st.spinner("Loading embedding model…"):
                try:
                    from axon.main import OpenEmbedding

                    st.session_state.brain.embedding = OpenEmbedding(config)
                    st.session_state["_emb_provider"] = config.embedding_provider
                    st.session_state["_emb_model"] = config.embedding_model
                except Exception as e:
                    st.error(f"Failed to load embedding model: {e}")
                    config.embedding_provider = _prev_emb_provider
                    config.embedding_model = _prev_emb_model

    # ── Temperature & Web search — visible below model status ──
    _t_col, _t_val = st.columns([3, 1])
    with _t_col:
        config.llm_temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            config.llm_temperature,
            step=0.05,
            label_visibility="collapsed",
        )
    with _t_val:
        st.caption(f"{config.llm_temperature:.2f}")

    _ws_col, _ws_lbl = st.columns([1, 4])
    with _ws_col:
        config.truth_grounding = st.checkbox(
            "Truth grounding", config.truth_grounding, key="ws_toggle", label_visibility="collapsed"
        )
    with _ws_lbl:
        st.caption("🌐 Web search")
    if config.truth_grounding:
        config.brave_api_key = st.text_input(
            "Brave API Key",
            value=config.brave_api_key,
            type="password",
            label_visibility="collapsed",
            placeholder="Brave API key",
        )

    # =========================================================================
    # RAG INTELLIGENCE
    # =========================================================================
    st.markdown('<div class="sb-section">⚡ RAG Intelligence</div>', unsafe_allow_html=True)

    # Always-visible basics
    config.hybrid_search = st.checkbox("Hybrid search", config.hybrid_search)
    config.top_k = st.slider("Top-K", 1, 20, config.top_k)
    config.similarity_threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, config.similarity_threshold, step=0.05
    )
    config.discussion_fallback = st.checkbox("Discussion fallback", config.discussion_fallback)
    config.rerank = st.checkbox("Re-ranking", config.rerank)
    if config.rerank:
        config.reranker_provider = st.selectbox(
            "Re-ranker",
            ["cross-encoder", "llm"],
            index=0 if config.reranker_provider == "cross-encoder" else 1,
        )

    # Advanced query transformations — collapsed by default
    with st.expander("Advanced retrieval"):
        config.multi_query = st.checkbox(
            "Multi-query",
            config.multi_query,
            help="Generate 3 rephrased queries and merge their results for broader recall",
        )
        config.hyde = st.checkbox(
            "HyDE",
            config.hyde,
            help="Generate a hypothetical answer document and embed that instead of the raw query",
        )
        config.step_back = st.checkbox(
            "Step-back prompting",
            config.step_back,
            help="Abstract the query to a more general form before retrieval to surface background knowledge",
        )
        config.query_decompose = st.checkbox(
            "Query decomposition",
            config.query_decompose,
            help="Break complex questions into simpler sub-questions for independent retrieval",
        )
        config.compress_context = st.checkbox(
            "Context compression",
            config.compress_context,
            help="Use the LLM to extract only query-relevant sentences from each retrieved chunk",
        )
        config.raptor = st.checkbox(
            "RAPTOR",
            config.raptor,
            help="Generate hierarchical summary nodes during ingest for multi-hop question answering",
        )
        config.graph_rag = st.checkbox(
            "GraphRAG",
            config.graph_rag,
            help="Extract entities during ingest and expand retrieval results via entity-linked documents",
        )

    # =========================================================================
    # KNOWLEDGE HUB
    # =========================================================================
    st.markdown('<div class="sb-section">📚 Knowledge Hub</div>', unsafe_allow_html=True)

    # ── DOCUMENTS (collapsed) ──
    with st.expander("📥 Documents"):
        if "ingested_files" not in st.session_state:
            st.session_state.ingested_files = []
        if st.session_state.ingested_files:
            st.caption(
                "Recent: " + ", ".join(f"`{f}`" for f in st.session_state.ingested_files[-3:])
            )

        uploaded_file = st.file_uploader(
            "File",
            type=["txt", "md", "pdf", "csv", "json"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            if st.button("⬆ Ingest", use_container_width=True, type="tertiary"):
                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_{uploaded_file.name}"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                try:
                    from axon.loaders import DirectoryLoader

                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    loader_mgr = DirectoryLoader()
                    if ext in loader_mgr.loaders:
                        with st.status("Ingesting...", expanded=True) as ingest_status:
                            st.write("📂 Loading file...")
                            docs = loader_mgr.loaders[ext].load(tmp_path)
                            for d in docs:
                                d["metadata"]["source"] = uploaded_file.name
                            st.write("✂️ Chunking...")
                            # chunking already done by the loader above
                            st.write("🔢 Generating embeddings & indexing...")
                            st.session_state.brain.ingest(docs)
                            n = len(docs)
                            st.write(f"✅ {n} chunks ingested")
                            ingest_status.update(
                                label=f"✅ Ingested {n} chunks from {uploaded_file.name}",
                                state="complete",
                            )
                        st.session_state.ingested_files.append(uploaded_file.name)
                        st.toast(f"✅ {n} chunks ingested from {uploaded_file.name}")
                    else:
                        st.error(f"Unsupported file type: {ext}")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        ingest_dir = st.text_input(
            "Directory", placeholder="/my/docs", label_visibility="collapsed"
        )
        if st.button("📂 Ingest Directory", use_container_width=True, type="tertiary"):
            if os.path.isdir(ingest_dir):
                allowed_base = os.path.abspath(os.getenv("RAG_INGEST_BASE", "."))
                abs_path = os.path.abspath(ingest_dir)
                try:
                    common = os.path.commonpath([allowed_base, abs_path])
                    if common != allowed_base:
                        st.error(f"Access denied: path outside allowed base '{allowed_base}'.")
                        st.stop()
                except ValueError:
                    st.error(f"Access denied: path outside allowed base '{allowed_base}'.")
                    st.stop()
                from axon.loaders import DirectoryLoader

                loader = DirectoryLoader()
                all_docs = []
                with st.status("Ingesting directory...", expanded=True) as dir_status:
                    st.write("📂 Scanning directory...")
                    files = list(Path(abs_path).rglob("*"))
                    supported = [f for f in files if f.suffix.lower() in loader.loaders]
                    st.write(f"✂️ Loading {len(supported)} file(s)...")
                    for f in supported:
                        try:
                            docs = loader.loaders[f.suffix.lower()].load(str(f))
                            all_docs.extend(docs)
                        except Exception:
                            pass
                    st.write(f"🔢 Indexing {len(all_docs)} chunk(s)...")
                    if all_docs:
                        st.session_state.brain.ingest(all_docs)
                    dir_status.update(
                        label=f"✅ Done — {len(all_docs)} chunks",
                        state="complete",
                    )
                st.toast(f"✅ {len(all_docs)} chunks ingested")
            else:
                st.error("Invalid directory path.")

    # ── KNOWLEDGE BASE VIEWER (collapsed) ──
    with st.expander("📚 Knowledge Base"):
        brain_obj = st.session_state.get("brain")
        if brain_obj is None:
            st.caption("Brain not initialised yet.")
        else:
            if st.button("🔄 Refresh", use_container_width=True, type="tertiary", key="kb_refresh"):
                st.session_state.pop("kb_docs_cache", None)

            if "kb_docs_cache" not in st.session_state:
                with st.spinner("Loading…"):
                    try:
                        st.session_state.kb_docs_cache = brain_obj.list_documents()
                    except Exception as e:
                        st.error(f"Could not load knowledge base: {e}")
                        st.session_state.kb_docs_cache = []

            docs_cache = st.session_state.get("kb_docs_cache", [])
            total_chunks = sum(d["chunks"] for d in docs_cache)

            if not docs_cache:
                st.caption("📭 Knowledge base is empty.")
            else:
                st.caption(f"{len(docs_cache)} file(s) · {total_chunks} chunk(s)")
                for d in docs_cache:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:0.78rem;padding:2px 0;'>"
                        f"<span>📄 {html.escape(d['source'])}</span>"
                        f"<span style='color:rgba(200,200,200,0.6);'>{d['chunks']} chunks</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown(
    f'<h3 style="margin:0 0 1rem;font-size:1.1rem;color:rgba(228,228,231,0.85);">'
    f'{html.escape(current_session["name"])}</h3>',
    unsafe_allow_html=True,
)

# Display existing messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Timing caption for assistant messages
        if message["role"] == "assistant":
            ttft = message.get("ttft")
            elapsed = message.get("elapsed")
            if ttft is not None and elapsed is not None:
                st.caption(f"⏱ {ttft:.1f}s TTFT · {elapsed:.1f}s total")

        # Sources expander
        if message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])})"):
                for i, doc in enumerate(message["sources"]):
                    _render_source_card(i, doc)

        # Brain Thoughts expander
        if message.get("thoughts"):
            with st.expander("🧠 Brain Thoughts"):
                thoughts = message["thoughts"]
                active_flags = thoughts.get("active_flags", [])
                num_sources = thoughts.get("num_sources", 0)
                gen_time = thoughts.get("elapsed")
                if active_flags:
                    st.caption("Active RAG features: " + ", ".join(active_flags))
                else:
                    st.caption("No advanced RAG features active.")
                st.caption(f"Sources retrieved: {num_sources}")
                if gen_time is not None:
                    st.caption(f"Generation time: {gen_time:.1f}s")

# Chat input
if prompt := st.chat_input("Ask me anything about your documents…"):
    chat_history_snapshot = messages.copy()

    if len(messages) == 0:
        session_title = prompt[:30] + ("…" if len(prompt) > 30 else "")
        st.session_state.sessions[current_session_id]["name"] = session_title

    st.session_state.sessions[current_session_id]["messages"].append(
        {"role": "user", "content": prompt}
    )
    save_sessions(st.session_state.sessions)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        thoughts_placeholder = st.empty()
        timing_placeholder = st.empty()
        full_response = ""
        sources = []

        # Snapshot active RAG flags for Brain Thoughts
        _active_flags = []
        for flag_name, flag_label in [
            ("hybrid_search", "Hybrid Search"),
            ("multi_query", "Multi-Query"),
            ("hyde", "HyDE"),
            ("step_back", "Step-Back"),
            ("query_decompose", "Query Decompose"),
            ("compress_context", "Context Compression"),
            ("raptor", "RAPTOR"),
            ("graph_rag", "GraphRAG"),
            ("rerank", "Re-ranking"),
            ("truth_grounding", "Web Search"),
            ("discussion_fallback", "Discussion Fallback"),
        ]:
            if getattr(config, flag_name, False):
                _active_flags.append(flag_label)

        t_start = time.time()
        first_token = True
        ttft = None

        with st.spinner("Thinking…"):
            for chunk in st.session_state.brain.query_stream(
                prompt, chat_history=chat_history_snapshot
            ):
                if isinstance(chunk, dict) and chunk.get("type") == "sources":
                    sources = chunk.get("sources", [])
                    continue
                if first_token and not isinstance(chunk, dict):
                    ttft = time.time() - t_start
                    first_token = False
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

        elapsed = time.time() - t_start
        response_placeholder.markdown(full_response)

        # Timing caption
        if ttft is not None:
            timing_placeholder.caption(f"⏱ {ttft:.1f}s TTFT · {elapsed:.1f}s total")

        # Sources
        if sources:
            with sources_placeholder.expander(f"📚 Sources ({len(sources)})"):
                for i, doc in enumerate(sources):
                    _render_source_card(i, doc)

        # Brain Thoughts
        thoughts = {
            "active_flags": _active_flags,
            "num_sources": len(sources),
            "elapsed": elapsed,
        }
        with thoughts_placeholder.expander("🧠 Brain Thoughts"):
            if _active_flags:
                st.caption("Active RAG features: " + ", ".join(_active_flags))
            else:
                st.caption("No advanced RAG features active.")
            st.caption(f"Sources retrieved: {len(sources)}")
            st.caption(f"Generation time: {elapsed:.1f}s")

    # Persist message with timing and thoughts metadata
    st.session_state.sessions[current_session_id]["messages"].append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "ttft": ttft,
            "elapsed": elapsed,
            "thoughts": thoughts,
        }
    )
    save_sessions(st.session_state.sessions)

# ---------------------------------------------------------------------------
# Persistent status bar (fixed, bottom of viewport)
# ---------------------------------------------------------------------------
_active_proj_display = st.session_state.get("active_project", "default")
_llm_model_display = config.llm_model if "brain" in st.session_state else "—"
_emb_model_display = config.embedding_model if "brain" in st.session_state else "—"

st.markdown(
    f"""
    <style>
    #axon-status-bar {{
        position: fixed;
        bottom: 0; left: 0; right: 0;
        height: 28px;
        background: rgba(15,15,15,0.92);
        backdrop-filter: blur(8px);
        border-top: 1px solid rgba(255,255,255,0.07);
        display: flex;
        align-items: center;
        padding: 0 16px;
        font-size: 0.72rem;
        color: rgba(228,228,231,0.45);
        gap: 16px;
        z-index: 9999;
    }}
    #axon-status-bar .dot {{ color: rgba(139,92,246,0.7); }}
    #axon-status-bar .val {{ color: rgba(228,228,231,0.75); font-weight: 500; }}
    </style>
    <div id="axon-status-bar">
        <span>📁 <span class="val">{html.escape(_active_proj_display)}</span></span>
        <span class="dot">•</span>
        <span>🤖 <span class="val">{html.escape(_llm_model_display)}</span></span>
        <span class="dot">•</span>
        <span>⚡ <span class="val">{html.escape(_emb_model_display)}</span></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main_ui():
    """Entry point for the axon-ui command."""
    import subprocess

    if not _STREAMLIT_AVAILABLE:
        print(
            "ERROR: streamlit is not installed.\n" "Install it with:  pip install axon-rag[ui]",
            file=sys.stderr,
        )
        sys.exit(1)

    app_path = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", app_path])


if __name__ == "__main__":
    pass
