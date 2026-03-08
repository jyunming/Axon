import streamlit as st
import os
import sys
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

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "brain" not in st.session_state:
    with st.spinner("Initializing Brain..."):
        st.session_state.brain = OpenStudioBrain()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    config = st.session_state.brain.config
    
    st.subheader("RAG Parameters")
    config.top_k = st.slider("Top K", 1, 20, config.top_k)
    config.similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, config.similarity_threshold)
    config.hybrid_search = st.checkbox("Enable Hybrid Search", config.hybrid_search)
    config.discussion_fallback = st.checkbox("Enable Discussion Fallback", config.discussion_fallback)
    
    st.subheader("LLM Parameters")
    config.llm_provider = st.selectbox("LLM Provider", ["ollama", "gemini", "ollama_cloud"], index=["ollama", "gemini", "ollama_cloud"].index(config.llm_provider) if config.llm_provider in ["ollama", "gemini", "ollama_cloud"] else 0)
    
    if config.llm_provider == "ollama":
        config.llm_model = st.text_input("Local Model", config.llm_model)
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
        ollama_cloud_models = [
            "gpt-oss:120b",
            "kimi-k2.5",
            "cogito-2.1:671b",
            "mistral-large-3:675b",
            "gpt-oss:20b",
            "deepseek-v3.2",
            "qwen3-coder:480b",
            "gemini-3-flash-preview"
        ]
        config.llm_model = st.selectbox("Ollama Cloud Model", ollama_cloud_models, index=0 if config.llm_model not in ollama_cloud_models else ollama_cloud_models.index(config.llm_model))
        
    config.llm_temperature = st.slider("Temperature", 0.0, 1.0, config.llm_temperature)
    
    st.subheader("Re-ranking")
    config.rerank = st.checkbox("Enable Re-ranking", config.rerank)
    config.reranker_provider = st.selectbox("Re-ranking Provider", ["cross-encoder", "llm"], index=0 if config.reranker_provider == "cross-encoder" else 1)
    
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
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🧠 Local RAG Brain")
st.caption("General purpose local RAG interface for private knowledge bases")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
             with st.expander(f"📚 Sources ({len(message['sources'])})"):
                 for i, doc in enumerate(message['sources']):
                     st.markdown(f"**[{i+1}] ID: {doc['id']}** (Score: {doc.get('score', 0):.3f})")
                     st.text(doc['text'][:500] + ("..." if len(doc['text']) > 500 else ""))
                     st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        full_response = ""
        sources = []
        
        with st.spinner("Searching and synthesizing..."):
            # Use query_stream for better UX
            for chunk in st.session_state.brain.query_stream(prompt):
                if isinstance(chunk, dict) and chunk.get("type") == "sources":
                    sources = chunk.get("sources", [])
                    continue
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            if sources:
                with sources_placeholder.expander(f"📚 Sources ({len(sources)})"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**[{i+1}] ID: {doc['id']}** (Score: {doc.get('score', 0):.3f})")
                        st.text(doc['text'][:500] + ("..." if len(doc['text']) > 500 else ""))
                        st.divider()
            
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})

def main_ui():
    """Entry point for studio-brain-ui command."""
    import subprocess
    import sys
    
    app_path = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    # If running directly, we don't need main_ui() because streamlit run handles it
    pass
