import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from axon.main import AxonBrain

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
# The theme handles: backgrounds, text, inputs, sliders, checkboxes, buttons.
# Here we only add what the theme system cannot express.
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
    /* Collapse vertical gaps between all sidebar elements */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.15rem !important;
    }

    /* ── Sidebar section headers ── */
    .sb-section {
        font-size: 0.63rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.28);
        margin: 10px 0 4px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.07);
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
        transition: background 0.12s, color 0.12s !important;
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

    /* ── Small action buttons (Clear, Delete) ── */
    [data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
        font-size: 0.75rem !important;
        color: rgba(228,228,231,0.40) !important;
        padding: 2px 6px !important;
        min-height: 26px !important;
        justify-content: center !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
        color: rgba(228,228,231,0.80) !important;
        background: rgba(255,255,255,0.05) !important;
    }

    /* ── Sidebar widget labels — compact, muted ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p {
        font-size: 0.77rem !important;
        color: rgba(228,228,231,0.50) !important;
        margin-bottom: 2px !important;
    }

    /* ── Sidebar expanders ── */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: rgba(228,228,231,0.45) !important;
        padding: 3px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        color: rgba(228,228,231,0.85) !important;
    }

    /* ── Sidebar widgets: match sidebar background (#1c1c2a) ── */
    /* Selectbox trigger */
    [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
        background-color: #1c1c2a !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    /* Text / password inputs */
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background-color: #1c1c2a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] {
        background-color: #1c1c2a !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    /* Slider track fill */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div {
        background-color: transparent !important;
    }
    /* File uploader drop zone */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
        background-color: #1c1c2a !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    /* Expander inner content area */
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
    }

    /* ── Chat messages — card style ── */
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        background: rgba(255,255,255,0.02) !important;
    }
    [data-testid="stChatMessage"] p { line-height: 1.7 !important; }

    /* ── Code colour ── */
    code { color: #a78bfa !important; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #71717a; }
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
    """Fetch available Gemini/Gemma models via google.generativeai. Cached 1 h."""
    if not api_key:
        return _FALLBACK_GEMINI_MODELS
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
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
    with st.spinner("Initializing Brain…"):
        st.session_state.brain = AxonBrain()
    # Seed embedding baseline so the hot-swap warning only fires on user changes
    st.session_state["_emb_provider"] = st.session_state.brain.config.embedding_provider
    st.session_state["_emb_model"] = st.session_state.brain.config.embedding_model

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

current_session_id = st.session_state.current_session_id
current_session = st.session_state.sessions[current_session_id]
messages = current_session["messages"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    config = st.session_state.brain.config

    # ── App title ──
    st.markdown(
        '<div style="font-size:1rem;font-weight:700;color:#e4e4e7;margin-bottom:8px;">🧠 Axon</div>',
        unsafe_allow_html=True,
    )

    # ── CONVERSATIONS ──
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

    # Session list — newest first, one button per session
    for sid, sess in reversed(list(st.session_state.sessions.items())):
        label = sess["name"] if len(sess["name"]) <= 27 else sess["name"][:24] + "…"
        is_active = sid == current_session_id
        btn_type = "primary" if is_active else "tertiary"
        if st.button(label, use_container_width=True, type=btn_type, key=f"sess_{sid}"):
            if not is_active:
                st.session_state.current_session_id = sid
                st.rerun()

    # Active session actions (always below session list, compact)
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
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.button("🧹 Clear", use_container_width=True, type="tertiary", key="clear_chat"):
                st.session_state.confirm_clear = True
                st.rerun()
        with ac2:
            if st.button("🗑 Delete", use_container_width=True, type="tertiary", key="delete_sess"):
                if len(st.session_state.sessions) > 1:
                    del st.session_state.sessions[current_session_id]
                    save_sessions(st.session_state.sessions)
                    st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]
                    st.rerun()
                else:
                    st.warning("Cannot delete the only session.")

    # ── MODEL ──
    st.markdown('<div class="sb-section">Model</div>', unsafe_allow_html=True)

    provider_labels = {
        "ollama": "🤖 Ollama (local)",
        "gemini": "✨ Gemini",
        "ollama_cloud": "☁️ Ollama Cloud",
        "openai": "🔑 OpenAI-compatible",
        "vllm": "⚡ vLLM (local)",
    }
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
        try:
            from ollama import Client

            client = Client(host=config.ollama_base_url)
            ollama_models = [
                m.model for m in client.list().models if not m.model.startswith("embeddinggemma")
            ]
        except Exception:
            pass
        if ollama_models:
            current_idx = (
                ollama_models.index(config.llm_model) if config.llm_model in ollama_models else 0
            )
            config.llm_model = st.selectbox(
                "Model", ollama_models, index=current_idx, label_visibility="collapsed"
            )
        else:
            config.llm_model = st.text_input(
                "Model", config.llm_model, label_visibility="collapsed", placeholder="model name"
            )
            st.caption("No models found — run `ollama pull <model>`")

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
                "Model", config.llm_model, label_visibility="collapsed", placeholder="model name"
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
                "Available", st.session_state["_ollama_cloud_models"], label_visibility="collapsed"
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
            "Model", config.llm_model, label_visibility="collapsed", placeholder="e.g. gpt-4o"
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

    # ── EMBEDDING ──
    st.markdown('<div class="sb-section">Embedding</div>', unsafe_allow_html=True)

    _EMB_PROVIDER_LABELS = {
        "sentence_transformers": "🤗 Sentence Transformers",
        "ollama": "🤖 Ollama (local)",
        "fastembed": "⚡ FastEmbed",
        "openai": "🔑 OpenAI-compatible",
    }
    _EMB_MODELS: dict = {
        "sentence_transformers": ["all-MiniLM-L6-v2", "BAAI/bge-large-en", "all-mpnet-base-v2"],
        "ollama": ["nomic-embed-text", "mxbai-embed-large"],
        "fastembed": ["BAAI/bge-small-en-v1.5", "BAAI/bge-m3"],
        "openai": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
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

    # Ollama base URL (shared with LLM when embedding also uses Ollama)
    if config.embedding_provider == "ollama" and config.llm_provider != "ollama":
        config.ollama_base_url = st.text_input(
            "Ollama URL (embed)",
            value=config.ollama_base_url,
            label_visibility="collapsed",
            placeholder="http://localhost:11434",
        )

    suggestions = _EMB_MODELS.get(config.embedding_provider, [])
    # If current model isn't in the suggestion list, reset to the first suggestion
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
    if config.embedding_provider != _prev_emb_provider or config.embedding_model != _prev_emb_model:
        # Warn about potential ChromaDB dimension mismatch when docs already ingested
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
                # Revert to previous working values
                config.embedding_provider = _prev_emb_provider
                config.embedding_model = _prev_emb_model

    # ── SETTINGS ──
    st.markdown('<div class="sb-section">Settings</div>', unsafe_allow_html=True)

    # Temperature on one row with its value
    config.llm_temperature = st.slider(
        "Temperature", 0.0, 1.0, config.llm_temperature, step=0.05, label_visibility="collapsed"
    )
    st.caption(f"Temperature: {config.llm_temperature:.2f}")

    # Hybrid + Web on same row
    tc1, tc2 = st.columns(2)
    with tc1:
        config.hybrid_search = st.checkbox("Hybrid search", config.hybrid_search)
    with tc2:
        config.truth_grounding = st.checkbox("Web search", config.truth_grounding)

    if config.truth_grounding:
        config.brave_api_key = st.text_input(
            "Brave API Key",
            value=config.brave_api_key,
            type="password",
            label_visibility="collapsed",
            placeholder="Brave API key",
        )

    # ── ADVANCED (collapsed) ──
    with st.expander("⚙ Advanced"):
        config.top_k = st.slider("Top K results", 1, 20, config.top_k)
        config.similarity_threshold = st.slider(
            "Similarity threshold", 0.0, 1.0, config.similarity_threshold, step=0.05
        )
        config.discussion_fallback = st.checkbox("Discussion fallback", config.discussion_fallback)
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
        config.cite_sources = st.checkbox(
            "Inline citations",
            config.cite_sources,
            help="Instruct the LLM to cite [Doc N] inline when using information from retrieved documents",
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
        config.rerank = st.checkbox("Re-ranking", config.rerank)
        if config.rerank:
            config.reranker_provider = st.selectbox(
                "Re-ranker",
                ["cross-encoder", "llm"],
                index=0 if config.reranker_provider == "cross-encoder" else 1,
            )

    # ── DOCUMENTS (collapsed) ──
    with st.expander("📥 Documents"):
        if "ingested_files" not in st.session_state:
            st.session_state.ingested_files = []
        if st.session_state.ingested_files:
            st.caption(
                "Recent: " + ", ".join(f"`{f}`" for f in st.session_state.ingested_files[-3:])
            )

        uploaded_file = st.file_uploader(
            "File", type=["txt", "md", "pdf", "csv", "json"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            if st.button("⬆ Ingest", use_container_width=True, type="tertiary"):
                import tempfile

                with st.spinner(f"Ingesting {uploaded_file.name}…"):
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
                            docs = loader_mgr.loaders[ext].load(tmp_path)
                            for d in docs:
                                d["metadata"]["source"] = uploaded_file.name
                            st.session_state.brain.ingest(docs)
                            st.caption(f"✅ {len(docs)} chunk(s) ingested")
                            st.session_state.ingested_files.append(uploaded_file.name)
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
                if not abs_path.startswith(allowed_base):
                    st.error(f"Access denied: path outside allowed base '{allowed_base}'.")
                    st.stop()
                with st.spinner(f"Ingesting {abs_path}…"):
                    import asyncio

                    asyncio.run(st.session_state.brain.load_directory(abs_path))
                    st.success("Done!")
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
                        f"<span>📄 {d['source']}</span>"
                        f"<span style='color:rgba(200,200,200,0.6);'>{d['chunks']} chunks</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown(
    f'<h3 style="margin:0 0 1rem;font-size:1.1rem;color:rgba(228,228,231,0.85);">'
    f'{current_session["name"]}</h3>',
    unsafe_allow_html=True,
)

# Display existing messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])})"):
                for i, doc in enumerate(message["sources"]):
                    if doc.get("is_web"):
                        title = doc.get("metadata", {}).get("title", doc["id"])
                        st.markdown(f"**🌐 [{i+1}] [{title}]({doc['id']})**")
                    else:
                        display_score = doc.get("vector_score", doc.get("score", 0))
                        st.markdown(
                            f"**📄 [{i+1}] `{doc['id']}`** — similarity: `{display_score:.3f}`"
                        )
                    st.code(
                        doc["text"][:500] + ("…" if len(doc["text"]) > 500 else ""),
                        language=None,
                    )
                    st.divider()

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
        full_response = ""
        sources = []

        with st.spinner("Thinking…"):
            for chunk in st.session_state.brain.query_stream(
                prompt, chat_history=chat_history_snapshot
            ):
                if isinstance(chunk, dict) and chunk.get("type") == "sources":
                    sources = chunk.get("sources", [])
                    continue
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            if sources:
                with sources_placeholder.expander(f"📚 Sources ({len(sources)})"):
                    for i, doc in enumerate(sources):
                        if doc.get("is_web"):
                            title = doc.get("metadata", {}).get("title", doc["id"])
                            st.markdown(f"**🌐 [{i+1}] [{title}]({doc['id']})**")
                        else:
                            display_score = doc.get("vector_score", doc.get("score", 0))
                            st.markdown(
                                f"**📄 [{i+1}] `{doc['id']}`** — similarity: `{display_score:.3f}`"
                            )
                        st.code(
                            doc["text"][:500] + ("…" if len(doc["text"]) > 500 else ""),
                            language=None,
                        )
                        st.divider()

    st.session_state.sessions[current_session_id]["messages"].append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
    save_sessions(st.session_state.sessions)


def main_ui():
    """Entry point for studio-brain-ui command."""
    import subprocess

    app_path = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", app_path])


if __name__ == "__main__":
    pass
