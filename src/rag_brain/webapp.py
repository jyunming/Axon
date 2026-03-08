import streamlit as st
import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_brain.main import OpenStudioBrain, OpenStudioConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Local RAG Brain",
    page_icon="🧠",
    layout="wide"
)

SESSIONS_FILE = "sessions.json"

def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sessions(sessions_dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions_dict, f, indent=4)

# Initialize Session State
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
    
    if not st.session_state.sessions:
        default_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.sessions = {
            default_id: {
                "name": f"Session {timestamp}",
                "messages": [],
                "created_at": timestamp
            }
        }
        st.session_state.current_session_id = default_id
    else:
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]

if "brain" not in st.session_state:
    with st.spinner("Initializing Brain..."):
        st.session_state.brain = OpenStudioBrain()

current_session_id = st.session_state.current_session_id
current_session = st.session_state.sessions[current_session_id]
messages = current_session["messages"]

# Sidebar
with st.sidebar:
    st.title("💬 Chat Sessions")
    
    # Session Management UI
    session_options = {sid: sess["name"] for sid, sess in st.session_state.sessions.items()}
    
    selected_session = st.selectbox(
        "Select Session",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x],
        index=list(session_options.keys()).index(current_session_id)
    )
    
    if selected_session != current_session_id:
        st.session_state.current_session_id = selected_session
        st.rerun()
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Session", use_container_width=True):
            new_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.sessions[new_id] = {
                "name": f"Session {timestamp}",
                "messages": [],
                "created_at": timestamp
            }
            save_sessions(st.session_state.sessions)
            st.session_state.current_session_id = new_id
            st.rerun()
            
    with col2:
        if st.button("🗑️ Delete", use_container_width=True):
            if len(st.session_state.sessions) > 1:
                del st.session_state.sessions[current_session_id]
                save_sessions(st.session_state.sessions)
                st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]
                st.rerun()
            else:
                st.warning("Cannot delete the last session.")
                
    st.divider()

    st.title("⚙️ Settings")
    
    config = st.session_state.brain.config
    
    st.subheader("RAG Parameters")
    config.top_k = st.slider("Top K", 1, 20, config.top_k)
    config.similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, config.similarity_threshold)
    config.hybrid_search = st.checkbox("Enable Hybrid Search", config.hybrid_search)
    config.discussion_fallback = st.checkbox("Enable Discussion Fallback", config.discussion_fallback)
    
    st.subheader("LLM Parameters")
    llm_providers = ["ollama", "gemini", "ollama_cloud", "openai"]
    config.llm_provider = st.selectbox("LLM Provider", llm_providers, index=llm_providers.index(config.llm_provider) if config.llm_provider in llm_providers else 0)
    
    if config.llm_provider == "ollama":
        # Dynamically fetch available Ollama models
        ollama_models = []
        try:
            from ollama import Client
            client = Client(host=config.ollama_base_url)
            models_resp = client.list()
            ollama_models = [m.model for m in models_resp.models if not m.model.startswith("embeddinggemma")]
        except Exception:
            ollama_models = []
        
        if ollama_models:
            current_idx = ollama_models.index(config.llm_model) if config.llm_model in ollama_models else 0
            config.llm_model = st.selectbox("Local Model", ollama_models, index=current_idx)
        else:
            config.llm_model = st.text_input("Local Model (no models found — pull one with `ollama pull`)", config.llm_model)
            st.warning("No Ollama models detected. Pull a model first: `docker exec <ollama-container> ollama pull gemma`")
    elif config.llm_provider == "gemini":
        config.gemini_api_key = st.text_input("Gemini API Key", value=config.gemini_api_key, type="password")
        gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemma-3-27b-it",
            "gemma-3-12b-it",
            "gemma-3-4b-it"
        ]
        config.llm_model = st.selectbox("Gemini Model", gemini_models, index=0 if config.llm_model not in gemini_models else gemini_models.index(config.llm_model))
    elif config.llm_provider == "ollama_cloud":
        config.ollama_cloud_key = st.text_input("Ollama Cloud Key", value=config.ollama_cloud_key, type="password")
        config.llm_model = st.text_input("Ollama Cloud Model", config.llm_model)
        
    config.llm_temperature = st.slider("Temperature", 0.0, 1.0, config.llm_temperature)
    
    st.subheader("Re-ranking")
    config.rerank = st.checkbox("Enable Re-ranking", config.rerank)
    config.reranker_provider = st.selectbox("Re-ranking Provider", ["cross-encoder", "llm"], index=0 if config.reranker_provider == "cross-encoder" else 1)
    
    st.subheader("🌐 Web Search")
    config.truth_grounding = st.checkbox("Enable Truth Grounding", config.truth_grounding,
                                          help="Augment answers with live web results from Brave Search")
    if config.truth_grounding:
        config.brave_api_key = st.text_input("Brave API Key", value=config.brave_api_key, type="password",
                                              help="Get your key at https://brave.com/search/api/")
    
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []
        
    st.subheader("📥 Ingest Data")
    
    # Show recently ingested files
    if st.session_state.ingested_files:
        with st.expander("Recently Ingested Files"):
            for file_name in st.session_state.ingested_files:
                st.text(f"✅ {file_name}")
                
    uploaded_file = st.file_uploader("Upload a Single File", type=["txt", "md", "pdf", "csv", "json"])
    if uploaded_file is not None:
        if st.button("Ingest Uploaded File"):
            import tempfile
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    from rag_brain.loaders import DirectoryLoader
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    loader_mgr = DirectoryLoader()
                    if ext in loader_mgr.loaders:
                        docs = loader_mgr.loaders[ext].load(tmp_path)
                        # Fix metadata source to show original filename instead of tmp path
                        for d in docs:
                            d["metadata"]["source"] = uploaded_file.name
                        st.session_state.brain.ingest(docs)
                        st.session_state.ingested_files.append(uploaded_file.name)
                        st.success(f"Successfully ingested {uploaded_file.name}!")
                    else:
                        st.error(f"Unsupported file type: {ext}")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
    st.markdown("---")
    st.markdown("**Batch Upload**")
    
    ingest_dir = st.text_input("Directory Path", placeholder="C:/my_documents")
    if st.button("Ingest Directory"):
        if os.path.isdir(ingest_dir):
            allowed_base = os.path.abspath(os.getenv("RAG_INGEST_BASE", "."))
            abs_path = os.path.abspath(ingest_dir)
            if not abs_path.startswith(allowed_base):
                st.error(
                    f"Access denied: '{abs_path}' is outside the allowed base "
                    f"directory '{allowed_base}'. Set RAG_INGEST_BASE to change it."
                )
                st.stop()
            with st.spinner(f"Ingesting {abs_path}..."):
                import asyncio
                asyncio.run(st.session_state.brain.load_directory(abs_path))
                st.success("Ingestion complete!")
        else:
            st.error("Invalid directory path.")
            
    st.divider()
    
    if st.button("Clear Chat History (Current Session)"):
        st.session_state.sessions[current_session_id]["messages"] = []
        save_sessions(st.session_state.sessions)
        st.rerun()

# Main Chat Interface
st.title("🧠 Local RAG Brain")
st.caption(f"General purpose local RAG interface - **{current_session['name']}**")

# Display chat messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
             with st.expander(f"📚 Sources ({len(message['sources'])})"):
                 for i, doc in enumerate(message['sources']):
                     if doc.get('is_web'):
                         st.markdown(f"**🌐 [{i+1}] [{doc.get('metadata', {}).get('title', doc['id'])}]({doc['id']})**")
                     else:
                         st.markdown(f"**📄 [{i+1}] ID: {doc['id']}** (Score: {doc.get('score', 0):.3f})")
                     st.text(doc['text'][:500] + ("..." if len(doc['text']) > 500 else ""))
                     st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    
    # Store history snapshot (excluding the new prompt)
    chat_history_snapshot = messages.copy()
    
    # Update Session Name based on first prompt
    if len(messages) == 0:
        session_title = prompt[:25] + "..." if len(prompt) > 25 else prompt
        st.session_state.sessions[current_session_id]["name"] = session_title
    
    # Add user message
    new_user_msg = {"role": "user", "content": prompt}
    st.session_state.sessions[current_session_id]["messages"].append(new_user_msg)
    save_sessions(st.session_state.sessions)
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        full_response = ""
        sources = []
        
        with st.spinner("Searching and synthesizing..."):
            # Use query_stream with injected chat history
            for chunk in st.session_state.brain.query_stream(prompt, chat_history=chat_history_snapshot):
                if isinstance(chunk, dict) and chunk.get("type") == "sources":
                    sources = chunk.get("sources", [])
                    continue
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            if sources:
                with sources_placeholder.expander(f"📚 Sources ({len(sources)})"):
                    for i, doc in enumerate(sources):
                        if doc.get('is_web'):
                            st.markdown(f"**🌐 [{i+1}] [{doc.get('metadata', {}).get('title', doc['id'])}]({doc['id']})**")
                        else:
                            st.markdown(f"**📄 [{i+1}] ID: {doc['id']}** (Score: {doc.get('score', 0):.3f})")
                        st.text(doc['text'][:500] + ("..." if len(doc['text']) > 500 else ""))
                        st.divider()
            
    # Add assistant message and save
    new_asst_msg = {"role": "assistant", "content": full_response, "sources": sources}
    st.session_state.sessions[current_session_id]["messages"].append(new_asst_msg)
    save_sessions(st.session_state.sessions)

def main_ui():
    """Entry point for studio-brain-ui command."""
    import subprocess
    import sys
    
    app_path = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    pass
