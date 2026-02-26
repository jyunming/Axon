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
    config.rerank = st.checkbox("Enable Re-ranking", config.rerank)
    
    st.divider()
    
    st.subheader("📥 Ingest Data")
    ingest_dir = st.text_input("Directory Path", placeholder="C:/my_documents")
    if st.button("Ingest Directory"):
        if os.path.isdir(ingest_dir):
            with st.spinner(f"Ingesting {ingest_dir}..."):
                import asyncio
                asyncio.run(st.session_state.brain.load_directory(ingest_dir))
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

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Searching and synthesizing..."):
            # Use query_stream for better UX
            for chunk in st.session_state.brain.query_stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def main_ui():
    """Entry point for studio-brain-ui command."""
    import subprocess
    import sys
    
    app_path = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    # If running directly, we don't need main_ui() because streamlit run handles it
    pass
