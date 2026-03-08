"""
Core engine for Local RAG Brain - Open Source RAG Interface.
"""

# Suppress TensorFlow/Keras noise before any imports that might trigger them
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")  # tell transformers to skip TF backend
# Disable ChromaDB / PostHog telemetry (avoids atexit noise on Ctrl+C)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import re
import sys
import time
import yaml
import logging
import asyncio
import threading
from pathlib import Path
from typing import Literal, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables — project .env first, then user-global ~/.rag_brain/.env
load_dotenv()
_user_env = Path.home() / ".rag_brain" / ".env"
if _user_env.exists():
    load_dotenv(_user_env)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGBrain")


@dataclass
class OpenStudioConfig:
    """Configuration for Local RAG Brain."""
    # Embedding
    embedding_provider: Literal["sentence_transformers", "ollama", "fastembed", "openai"] = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    
    # LLM
    llm_provider: Literal["ollama", "gemini", "ollama_cloud", "openai"] = "ollama"
    llm_model: str = "gemma"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    api_key: str = ""
    gemini_api_key: str = ""
    ollama_cloud_key: str = ""
    ollama_cloud_url: str = ""
    
    def __post_init__(self) -> None:
        """Populate API-related fields from environment variables when not set."""
        if not self.api_key:
            self.api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.ollama_cloud_key:
            self.ollama_cloud_key = os.getenv("OLLAMA_CLOUD_KEY", "")
        if not self.ollama_cloud_url:
            self.ollama_cloud_url = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com/api")
        if not self.brave_api_key:
            self.brave_api_key = os.getenv("BRAVE_API_KEY", "")
    
    # Vector Store
    vector_store: Literal["chroma", "qdrant", "lancedb"] = "chroma"
    vector_store_path: str = "./chroma_data"
    
    # BM25 Settings
    bm25_path: str = "./bm25_index"
    
    # RAG Settings
    top_k: int = 10
    similarity_threshold: float = 0.3
    hybrid_search: bool = True
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Re-ranking
    rerank: bool = False
    reranker_provider: Literal["cross-encoder", "llm"] = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Query Transformations
    multi_query: bool = False
    hyde: bool = False
    discussion_fallback: bool = True

    # Web Search / Truth Grounding
    truth_grounding: bool = False
    brave_api_key: str = ""

    @classmethod
    def load(cls, path: str = "config.yaml") -> 'OpenStudioConfig':
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            logger.warning(f"Config file {path} not found. Using defaults.")
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten the YAML structure to match dataclass fields
        config_dict = {}
        if 'embedding' in data:
            config_dict.update({f"embedding_{k}": v for k, v in data['embedding'].items()})
        if 'llm' in data:
            config_dict.update({f"llm_{k}": v for k, v in data['llm'].items()})
        if 'vector_store' in data:
            config_dict.update({f"vector_store_{k}" if k == 'provider' else k: v for k, v in data['vector_store'].items()})
            if 'path' in data['vector_store']:
                config_dict['vector_store_path'] = data['vector_store']['path']
        if 'bm25' in data:
            if 'path' in data['bm25']:
                config_dict['bm25_path'] = data['bm25']['path']
        if 'rag' in data:
            config_dict.update(data['rag'])
        if 'chunk' in data:
            config_dict.update({f"chunk_{k}": v for k, v in data['chunk'].items()})
        if 'rerank' in data:
            if 'enabled' in data['rerank']:
                config_dict['rerank'] = data['rerank']['enabled']
            if 'provider' in data['rerank']:
                config_dict['reranker_provider'] = data['rerank']['provider']
            if 'model' in data['rerank']:
                config_dict['reranker_model'] = data['rerank']['model']
                
        if 'query_transformations' in data:
            if 'multi_query' in data['query_transformations']:
                config_dict['multi_query'] = data['query_transformations']['multi_query']
            if 'hyde' in data['query_transformations']:
                config_dict['hyde'] = data['query_transformations']['hyde']
            if 'discussion_fallback' in data['query_transformations']:
                config_dict['discussion_fallback'] = data['query_transformations']['discussion_fallback']
        if 'web_search' in data:
            ws = data['web_search']
            config_dict['truth_grounding'] = ws.get('enabled', False)
            if ws.get('brave_api_key'):
                config_dict['brave_api_key'] = ws['brave_api_key']
        
        # Map some specific names if they don't match exactly
        if 'ollama_base_url' not in config_dict and 'llm_base_url' in config_dict:
            config_dict['ollama_base_url'] = config_dict['llm_base_url']
            
        if 'api_key' not in config_dict and 'llm_api_key' in config_dict:
            config_dict['api_key'] = config_dict['llm_api_key']
            
        # Environment Variable Overrides (High Priority for Docker)
        env_ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
        if env_ollama_host:
            config_dict['ollama_base_url'] = env_ollama_host

        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)


class OpenReranker:
    """
    Open-source reranking using Cross-Encoders.
    """
    
    def __init__(self, config: OpenStudioConfig):
        self.config = config
        self.model = None
        self.llm = None
        if self.config.rerank:
            if self.config.reranker_provider == "cross-encoder":
                try:
                    from sentence_transformers import CrossEncoder
                    logger.info(f"📊 Loading Reranker: {self.config.reranker_model}")
                    self.model = CrossEncoder(self.config.reranker_model)
                except ImportError:
                    logger.error("sentence-transformers not installed. Reranking disabled.")
                    self.config.rerank = False
            elif self.config.reranker_provider == "llm":
                logger.info(f"📊 Using LLM for Re-ranking (RankGPT)")
                self.llm = OpenLLM(self.config)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on a query.
        """
        if not self.config.rerank or (not self.model and not self.llm) or not documents:
            return documents
        
        logger.info(f"🔄 Reranking {len(documents)} documents...")

        if self.config.reranker_provider == "llm" and self.llm:
            return self._llm_rerank(query, documents)
        
        # Cross-encoder pointwise scoring
        # Prepare pairs: (query, doc_text)
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score descending
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs

    def _llm_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """RankGPT pointwise scoring implementation."""
        system_prompt = "You are an expert relevance ranker. Rate the relevance of the document to the query on a scale from 1 to 10. Output ONLY the integer score."
        
        from concurrent.futures import ThreadPoolExecutor
        
        def score_doc(doc):
            prompt = f"Query: {query}\n\nDocument: {doc['text']}\n\nScore (1-10):"
            try:
                response = self.llm.complete(prompt, system_prompt=system_prompt).strip()
                return float(response) if response.replace('.','',1).isdigit() else 0.0
            except Exception:
                return 0.0
                
        # Batch concurrent requests to reduce overall latency
        with ThreadPoolExecutor(max_workers=min(10, len(documents))) as executor:
            scores = list(executor.map(score_doc, documents))
            
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
            
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs


class OpenEmbedding:
    """
    Open-source embedding provider.
    """
    
    def __init__(self, config: OpenStudioConfig):
        self.config = config
        self.provider = config.embedding_provider
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            logger.info(f"📊 Loading Sentence Transformers: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
        elif self.provider == "ollama":
            import ollama
            logger.info(f"📊 Using Ollama Embedding: {self.config.embedding_model}")
            self.dimension = 768  # nomic-embed-text dimension
            
        elif self.provider == "fastembed":
            from fastembed import TextEmbedding
            logger.info(f"📊 Loading FastEmbed: {self.config.embedding_model}")
            self.model = TextEmbedding(model_name=self.config.embedding_model)
            if "bge-m3" in self.config.embedding_model.lower():
                self.dimension = 1024
            else:
                self.dimension = 384
                
        elif self.provider == "openai":
            from openai import OpenAI
            logger.info(f"📊 Using OpenAI API Embedding: {self.config.embedding_model}")
            kwargs = {"api_key": self.config.api_key} if self.config.api_key else {"api_key": "sk-dummy"}
            if self.config.ollama_base_url and self.config.ollama_base_url != "http://localhost:11434":
                kwargs["base_url"] = self.config.ollama_base_url
            self.model = OpenAI(**kwargs)
            self.dimension = 1536
            
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "sentence_transformers":
            embeddings = self.model.encode(texts, show_progress_bar=False)
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return list(embeddings)
            
        elif self.provider == "ollama":
            from ollama import Client
            client = Client(host=self.config.ollama_base_url)
            embeddings = []
            for text in texts:
                response = client.embeddings(model=self.config.embedding_model, prompt=text)
                embeddings.append(response['embedding'])
            return embeddings
            
        elif self.provider == "fastembed":
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]
            
        elif self.provider == "openai":
            response = self.model.embeddings.create(input=texts, model=self.config.embedding_model)
            return [data.embedding for data in response.data]
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]


class OpenLLM:
    """
    Open-source LLM provider.
    """
    
    def __init__(self, config: OpenStudioConfig):
        self.config = config
        self._openai_client = None
    
    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.config.api_key} if self.config.api_key else {"api_key": "sk-dummy"}
            if self.config.ollama_base_url and self.config.ollama_base_url != "http://localhost:11434":
                kwargs["base_url"] = self.config.ollama_base_url
            self._openai_client = OpenAI(**kwargs)
        return self._openai_client
    
    def complete(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None) -> str:
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
                options={
                    "temperature": self.config.llm_temperature,
                    "num_ctx": 8192
                }
            )
            return response['message']['content']
            
        elif provider == "gemini":
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import google.generativeai as genai
            if not getattr(self, '_gemini_configured', False):
                genai.configure(api_key=self.config.gemini_api_key)
                self._gemini_configured = True
            model_kwargs = {"model_name": self.config.llm_model}
            is_gemma = "gemma" in self.config.llm_model.lower()
            if system_prompt and not is_gemma:
                model_kwargs["system_instruction"] = system_prompt
            model = genai.GenerativeModel(**model_kwargs)
            
            contents = []
            for msg in history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})
            
            user_text = prompt
            if system_prompt and is_gemma:
                user_text = f"{system_prompt}\n\n{prompt}"
            contents.append({"role": "user", "parts": [user_text]})
                
            response = model.generate_content(
                contents,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.llm_max_tokens
                )
            )
            return response.text
            
        elif provider == "ollama_cloud":
            import httpx
            headers = {
                "Authorization": f"Bearer {self.config.ollama_cloud_key}",
                "Content-Type": "application/json"
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
                "options": {"temperature": self.config.llm_temperature}
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(f"{self.config.ollama_cloud_url}/generate", json=payload, headers=headers)
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
            
            response = self._get_openai_client().chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            return response.choices[0].message.content
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def stream(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None):
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
                options={
                    "temperature": self.config.llm_temperature,
                    "num_ctx": 8192
                }
            )
            for chunk in stream_resp:
                yield chunk['message']['content']
                
        elif provider == "gemini":
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import google.generativeai as genai
            if not getattr(self, '_gemini_configured', False):
                genai.configure(api_key=self.config.gemini_api_key)
                self._gemini_configured = True
            model_kwargs = {"model_name": self.config.llm_model}
            is_gemma = "gemma" in self.config.llm_model.lower()
            if system_prompt and not is_gemma:
                model_kwargs["system_instruction"] = system_prompt
            model = genai.GenerativeModel(**model_kwargs)
            
            contents = []
            for msg in history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})
            
            user_text = prompt
            if system_prompt and is_gemma:
                user_text = f"{system_prompt}\n\n{prompt}"
            contents.append({"role": "user", "parts": [user_text]})
                
            response = model.generate_content(
                contents,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.llm_max_tokens
                )
            )
            for chunk in response:
                yield chunk.text
                
        elif provider == "ollama_cloud":
            import httpx
            headers = {
                "Authorization": f"Bearer {self.config.ollama_cloud_key}",
                "Content-Type": "application/json"
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
                "options": {"temperature": self.config.llm_temperature}
            }
            import json
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", f"{self.config.ollama_cloud_url}/generate", json=payload, headers=headers) as response:
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
            
            stream = self._get_openai_client().chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content


class OpenVectorStore:
    """
    Open-source vector store interface.
    """
    
    def __init__(self, config: OpenStudioConfig):
        self.config = config
        self.provider = config.vector_store
        self.client = None
        self.collection = None
        self._init_store()
    
    def _init_store(self):
        if self.provider == "chroma":
            import chromadb
            logger.info(f"💾 Initializing ChromaDB: {self.config.vector_store_path}")
            self.client = chromadb.PersistentClient(path=self.config.vector_store_path)
            self.collection = self.client.get_or_create_collection(
                name="rag_brain",
                metadata={"hnsw:space": "cosine"}
            )
        elif self.provider == "qdrant":
            from qdrant_client import QdrantClient
            logger.info(f"💾 Initializing Qdrant: {self.config.vector_store_path}")
            self.client = QdrantClient(path=self.config.vector_store_path)
    
    def add(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        if self.provider == "chroma":
            self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        elif self.provider == "qdrant":
            from qdrant_client.models import PointStruct
            points = []
            for i, (id, embedding, text) in enumerate(zip(ids, embeddings, texts)):
                payload = {"text": text}
                if metadatas: payload.update(metadatas[i])
                points.append(PointStruct(id=id, vector=embedding, payload=payload))
            self.client.upsert(collection_name="rag_brain", points=points)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """Return all unique source files stored in the knowledge base with chunk counts.

        Returns:
            List of dicts sorted by source name, each with keys:
                - source (str): The metadata 'source' value, or 'unknown' if not set.
                - chunks (int): Number of chunks stored for that source.
                - doc_ids (List[str]): All chunk IDs belonging to this source.
        """
        if self.provider == "chroma":
            result = self.collection.get(include=["metadatas"])
            sources: Dict[str, Dict[str, Any]] = {}
            for doc_id, meta in zip(result["ids"], result["metadatas"] or [{}] * len(result["ids"])):
                source = (meta or {}).get("source", "unknown")
                if source not in sources:
                    sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
                sources[source]["chunks"] += 1
                sources[source]["doc_ids"].append(doc_id)
            return sorted(sources.values(), key=lambda x: x["source"])
        elif self.provider == "qdrant":
            from qdrant_client.models import ScrollRequest
            results, _ = self.client.scroll(collection_name="rag_brain", limit=10000, with_payload=True)
            sources: Dict[str, Dict[str, Any]] = {}
            for point in results:
                source = point.payload.get("source", "unknown")
                if source not in sources:
                    sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
                sources[source]["chunks"] += 1
                sources[source]["doc_ids"].append(str(point.id))
            return sorted(sources.values(), key=lambda x: x["source"])
        return []

    def search(self, query_embedding: List[float], top_k: int = 10, filter_dict: Dict = None) -> List[Dict]:
        if self.provider == "chroma":
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return [
                {"id": results['ids'][0][i], "text": results['documents'][0][i], "score": 1 - results['distances'][0][i], "metadata": results['metadatas'][0][i] if results['metadatas'] else {}}
                for i in range(len(results['ids'][0]))
            ]
        elif self.provider == "qdrant":
            results = self.client.search(collection_name="rag_brain", query_vector=query_embedding, limit=top_k)
            return [{"id": str(r.id), "text": r.payload.get("text", ""), "score": r.score, "metadata": {k: v for k, v in r.payload.items() if k != "text"}} for r in results]


class OpenStudioBrain:
    """
    Main interface for Local RAG Brain.
    """
    
    SYSTEM_PROMPT = """You are the 'RAG Brain', a highly capable and friendly AI assistant. 
Your primary goal is to help the user by answering questions based on the provided context from their private documents.

**Guidelines:**
1. **Prioritize Context**: If relevant information is found in the provided context, use it to answer the question accurately and cite the documents.
2. **General Knowledge Fallback**: If no relevant information is found in the context, DO NOT strictly refuse to answer. Instead, use your broad internal knowledge to provide a helpful response. 
3. **Be Transparent**: If you are using your general knowledge because no local documents matched the query, briefly mention it (e.g., 'I couldn't find specific details in your documents, but based on my general knowledge...').
4. **Agentic & Proactive**: Be helpful, concise, and encourage further discussion or ingestion of more data if needed.
"""
    
    def __init__(self, config: Optional[OpenStudioConfig] = None):
        self.config = config or OpenStudioConfig.load()
        logger.info("🧠 Initializing Local RAG Brain...")
        self.embedding = OpenEmbedding(self.config)
        self.llm = OpenLLM(self.config)
        self.vector_store = OpenVectorStore(self.config)
        self.reranker = OpenReranker(self.config)

        # Stash original (config.yaml) paths so we can restore them for "default"
        self._base_vector_store_path: str = os.path.abspath(self.config.vector_store_path)
        self._base_bm25_path: str = os.path.abspath(self.config.bm25_path)
        self._active_project: str = "default"
        
        try:
            from rag_brain.retrievers import BM25Retriever
            self.bm25 = BM25Retriever(storage_path=self.config.bm25_path)
        except ImportError:
            self.bm25 = None
            
        try:
            from rag_brain.splitters import RecursiveCharacterTextSplitter
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        except ImportError:
            self.splitter = None
        
        logger.info("✅ Local RAG Brain ready!")

    def switch_project(self, name: str) -> None:
        """Switch the active project, reinitializing vector store and BM25.

        Embedding and LLM are kept (expensive to reload). The "default" sentinel
        restores the paths from config.yaml.

        Args:
            name: Project name or "default".

        Raises:
            ValueError: If the project does not exist (use /project new first).
        """
        from rag_brain.projects import (
            project_bm25_path, project_dir, project_vector_path, set_active_project,
        )

        if name == "default":
            self.config.vector_store_path = self._base_vector_store_path
            self.config.bm25_path = self._base_bm25_path
        else:
            root = project_dir(name)
            if not root.exists():
                raise ValueError(
                    f"Project '{name}' does not exist. Create it first with /project new {name}"
                )
            self.config.vector_store_path = project_vector_path(name)
            self.config.bm25_path = project_bm25_path(name)

        # Reinitialize stores with new paths
        self.vector_store = OpenVectorStore(self.config)
        try:
            from rag_brain.retrievers import BM25Retriever
            self.bm25 = BM25Retriever(storage_path=self.config.bm25_path)
        except ImportError:
            self.bm25 = None

        self._active_project = name
        set_active_project(name)
        logger.info(f"📂 Switched to project '{name}'")
    
    def _log_query_metrics(self, query: str, vector_count: int, bm25_count: int,
                            filtered_count: int, final_count: int, top_score: float,
                            latency_ms: float, transformations: Dict = None):
        """Log structured metrics for a query."""
        logger.info({
            "event": "query_complete",
            "query_preview": query[:80],
            "latency_ms": round(latency_ms, 1),
            "results": {
                "vector": vector_count,
                "bm25": bm25_count,
                "after_filter": filtered_count,
                "final": final_count,
            },
            "top_score": round(top_score, 4) if top_score else None,
            "hybrid": self.config.hybrid_search,
            "rerank": self.config.rerank,
            "transformations": transformations or {}
        })

    def ingest(self, documents: List[Dict[str, Any]]):
        if not documents: return
        t0 = time.time()
        from tqdm import tqdm
        logger.info(f"📥 Ingesting {len(documents)} documents...")
        if self.splitter: documents = self.splitter.transform_documents(documents)
        n_chunks = len(documents)
        if self.bm25: self.bm25.add_documents(documents)

        ids = [d['id'] for d in documents]
        texts = [d['text'] for d in documents]
        metadatas = [d.get('metadata', {}) for d in documents]

        logger.info("   Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        t_embed = time.time()
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            all_embeddings.extend(self.embedding.embed(texts[i:i + batch_size]))
        embed_ms = (time.time() - t_embed) * 1000

        t_store = time.time()
        self.vector_store.add(ids, texts, all_embeddings, metadatas)
        store_ms = (time.time() - t_store) * 1000

        logger.info({
            "event": "ingest_complete",
            "chunks": n_chunks,
            "embed_ms": round(embed_ms, 1),
            "store_ms": round(store_ms, 1),
            "total_ms": round((time.time() - t0) * 1000, 1),
        })

    def _get_hyde_document(self, query: str) -> str:
        """Generate a Hypothetical Document Embedding (HyDE) passage."""
        prompt = f"Please write a hypothetical, detailed passage that directly answers the following question. Use informative and factual language.\n\nQuestion: {query}"
        return self.llm.complete(prompt, system_prompt="You are a helpful expert answering questions.")

    def _get_multi_queries(self, query: str) -> List[str]:
        """Generate alternative query phrasings for multi-query retrieval."""
        prompt = f"Generate 3 alternative phrasings of the following question to help with retrieving documents from a vector database. Output each phrasing on a new line and DO NOT output anything else.\n\nQuestion: {query}"
        response = self.llm.complete(prompt, system_prompt="You are an expert search engineer.")
        queries = [q.strip("- \t1234567890.") for q in response.split("\n") if q.strip()]
        return [query] + queries[:3]  # Always include original query

    def _execute_web_search(self, query: str, count: int = 5) -> List[Dict]:
        """Execute a web search using the Brave Search API and return results."""
        import httpx
        try:
            response = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.config.brave_api_key
                },
                params={"q": query, "count": count},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            web_results = []
            for item in data.get("web", {}).get("results", [])[:count]:
                snippet = item.get("description", "")
                title = item.get("title", "")
                url = item.get("url", "")
                web_results.append({
                    "id": url,
                    "text": f"{title}\n{snippet}",
                    "score": 1.0,
                    "metadata": {"source": url, "title": title},
                    "is_web": True
                })
            logger.info(f"🌐 Brave Search returned {len(web_results)} results for: {query[:60]}")
            return web_results
        except Exception as e:
            logger.warning(f"🌐 Web search failed: {e}")
            return []

    def _execute_retrieval(self, query: str, filters: Dict = None) -> Dict:
        """Central retrieval execution logic supporting HyDE, Multi-Query, and Web Search."""
        transforms = {"hyde_applied": False, "multi_query_applied": False, "web_search_applied": False, "queries": [query]}
        
        search_queries = [query]
        vector_query = query
        
        if self.config.multi_query:
            search_queries = self._get_multi_queries(query)
            transforms['multi_query_applied'] = True
            transforms['queries'] = search_queries
            
        if self.config.hyde:
            # We construct a single hypothetical document based on the primary query
            vector_query = self._get_hyde_document(query)
            transforms['hyde_applied'] = True

        fetch_k = self.config.top_k * 3 if (self.config.rerank or self.config.hybrid_search) else self.config.top_k
        
        all_vector_results = []
        all_bm25_results = []
        
        # Vector Search
        if self.config.hyde:
             # HyDE uses a single hypothetical document as the vector query
             all_vector_results.extend(self.vector_store.search(self.embedding.embed_query(vector_query), top_k=fetch_k, filter_dict=filters))
        else:
             # When multi_query is enabled, batch embed all queries to avoid
             # multiple sequential embedding calls (latency/cost optimization).
             if len(search_queries) == 1:
                 sq = search_queries[0]
                 all_vector_results.extend(self.vector_store.search(self.embedding.embed_query(sq), top_k=fetch_k, filter_dict=filters))
             else:
                 embeddings = self.embedding.embed(search_queries)
                 for emb in embeddings:
                     all_vector_results.extend(self.vector_store.search(emb, top_k=fetch_k, filter_dict=filters))
        
        # Dedupe vector store results based on ID
        dedup_vector = {}
        for r in all_vector_results:
            if r['id'] not in dedup_vector or r['score'] > dedup_vector[r['id']]['score']:
                dedup_vector[r['id']] = r
        vector_results = list(dedup_vector.values())
        vector_count = len(vector_results)

        # Hybrid Search
        bm25_count = 0
        if self.config.hybrid_search and self.bm25:
            for sq in search_queries:
                all_bm25_results.extend(self.bm25.search(sq, top_k=fetch_k))
                
            dedup_bm25 = {}
            for r in all_bm25_results:
                if r['id'] not in dedup_bm25 or r['score'] > dedup_bm25[r['id']]['score']:
                    dedup_bm25[r['id']] = r
            bm25_results = list(dedup_bm25.values())
            bm25_count = len(bm25_results)
            
            from rag_brain.retrievers import reciprocal_rank_fusion
            results = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            results = vector_results

        results = [r for r in results if r.get('score', 1.0) >= self.config.similarity_threshold or 'fused_only' in r]

        # Web Search (Truth Grounding)
        # Trigger when the best cosine similarity from vector search is below the threshold,
        # meaning local knowledge is genuinely insufficient (even if BM25 returned fused_only docs).
        web_count = 0
        if self.config.truth_grounding and self.config.brave_api_key:
            max_vector_score = max((r['score'] for r in vector_results), default=0.0)
            local_sufficient = max_vector_score >= self.config.similarity_threshold
            if not local_sufficient:
                logger.info(
                    f"🌐 Local knowledge insufficient (best vector score {max_vector_score:.3f} < "
                    f"threshold {self.config.similarity_threshold}) — falling back to Brave web search"
                )
                web_results = self._execute_web_search(query)
                web_count = len(web_results)
                # Replace low-relevance local results with web results
                results = web_results
                transforms['web_search_applied'] = True

        return {
            "results": results,
            "vector_count": vector_count,
            "bm25_count": bm25_count,
            "web_count": web_count,
            "filtered_count": len(results),
            "transforms": transforms
        }

    def _build_context(self, results: List[Dict]) -> tuple:
        """Build context string from results, labelling web vs local sources distinctly.

        Returns:
            Tuple of (context_string, has_web_results).
        """
        parts = []
        has_web = False
        for i, r in enumerate(results):
            if r.get("is_web"):
                has_web = True
                title = r.get("metadata", {}).get("title", r["id"])
                parts.append(f"[Web Result {i+1} — {title} ({r['id']})]\n{r['text']}")
            else:
                parts.append(f"[Document {i+1} (ID: {r['id']})]\n{r['text']}")
        return "\n\n".join(parts), has_web

    def _build_system_prompt(self, has_web: bool) -> str:
        """Return the system prompt, extended based on web search state.

        When truth_grounding is enabled but local docs answered the question,
        the LLM is told web search is available as a fallback (sets expectations).
        When web results are actually in the context, the LLM is told to use and cite them.
        """
        if not self.config.truth_grounding:
            return self.SYSTEM_PROMPT
        if has_web:
            return self.SYSTEM_PROMPT + (
                "\n\n**Web Search Used**: Local documents did not contain sufficient information, "
                "so live Brave Search results have been added to your context (marked as '[Web Result]'). "
                "Use these web results to answer the question and always cite the source URL."
            )
        return self.SYSTEM_PROMPT + (
            "\n\n**Web Search Available**: If the local documents above are insufficient, "
            "you have access to live Brave Search as a fallback tool. It was not needed for this query."
        )

    def query(self, query: str, filters: Dict = None, chat_history: List[Dict[str, str]] = None) -> str:
        t0 = time.time()
        
        retrieval = self._execute_retrieval(query, filters)
        results = retrieval['results']
        
        if not results:
            self._log_query_metrics(query, retrieval['vector_count'], retrieval['bm25_count'], 
                                    retrieval['filtered_count'], 0, 0.0, (time.time() - t0) * 1000, retrieval['transforms'])
            
            if self.config.discussion_fallback:
                # Send plain query as user message so multi-turn history stays consistent
                return self.llm.complete(query, self._build_system_prompt(False), chat_history=chat_history)
            
            return "I don't have any relevant information to answer that question."
            
        if self.config.rerank: results = self.reranker.rerank(query, results)

        results = results[:self.config.top_k]
        final_count = len(results)
        top_score = results[0].get('score', 0) if results else 0
        context, has_web = self._build_context(results)
        # Inject RAG context into system prompt so the user message stays as the plain
        # question — this keeps multi-turn chat_history consistent across turns.
        system_prompt = self._build_system_prompt(has_web) + f"\n\n**Relevant context from documents:**\n{context}"
        response = self.llm.complete(query, system_prompt, chat_history=chat_history)
        self._log_query_metrics(query, retrieval['vector_count'], retrieval['bm25_count'], 
                                retrieval['filtered_count'], final_count, top_score, (time.time() - t0) * 1000, retrieval['transforms'])
        return response

    def query_stream(self, query: str, filters: Dict = None, chat_history: List[Dict[str, str]] = None):
        retrieval = self._execute_retrieval(query, filters)
        results = retrieval['results']
        
        if not results:
            if self.config.discussion_fallback:
                # Send plain query as user message so multi-turn history stays consistent
                yield from self.llm.stream(query, self._build_system_prompt(False), chat_history=chat_history)
                return
            yield "I don't have any relevant information to answer that question."
            return
            
        if self.config.rerank: results = self.reranker.rerank(query, results)
        
        results = results[:self.config.top_k]
        context, has_web = self._build_context(results)
        # Inject RAG context into system prompt so the user message stays as the plain
        # question — this keeps multi-turn chat_history consistent across turns.
        system_prompt = self._build_system_prompt(has_web) + f"\n\n**Relevant context from documents:**\n{context}"
        
        # Yield a marker object so UI can optionally reconstruct sources
        yield {"type": "sources", "sources": results}

        yield from self.llm.stream(query, system_prompt, chat_history=chat_history)

    async def load_directory(self, directory: str):
        from rag_brain.loaders import DirectoryLoader
        loader = DirectoryLoader()
        logger.info(f"📁 Scanning: {directory}")
        documents = await loader.aload(directory)
        if documents: self.ingest(documents)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return all unique source files in the knowledge base with chunk counts.

        Returns:
            List of dicts sorted by source name, each with keys:
                - source (str): File name / source identifier.
                - chunks (int): Number of stored chunks for this source.
                - doc_ids (List[str]): All chunk IDs for this source.
        """
        return self.vector_store.list_documents()


def main():
    import argparse
    # On Windows, switch the console to UTF-8 (codepage 65001) so that
    # box-drawing characters and emoji render correctly.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Local RAG Brain CLI")
    parser.add_argument('query', nargs='?', help='Question to ask')
    parser.add_argument('--ingest', help='Path to file or directory to ingest')
    parser.add_argument('--list', action='store_true', help='List all ingested sources in the knowledge base')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--stream', action='store_true', help='Stream the response')
    parser.add_argument(
        '--provider',
        choices=['ollama', 'gemini', 'ollama_cloud', 'openai'],
        help='LLM provider to use (overrides config)',
    )
    parser.add_argument('--model', help='Model name to use (overrides config), e.g. gemma:2b, gemini-1.5-flash, gpt-4o')
    parser.add_argument('--list-models', action='store_true', help='List available Ollama models and supported cloud providers')
    parser.add_argument('--pull', metavar='MODEL', help='Pull an Ollama model by name, e.g. --pull gemma:2b')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress spinners and progress (auto-enabled when stdin is not a TTY)')
    args = parser.parse_args()

    # Suppress httpx INFO noise before _InitDisplay is active (ollama.list fires early)
    if sys.stdin.isatty():
        logging.getLogger("httpx").propagate = False
        logging.getLogger("httpx").setLevel(logging.WARNING)

    config = OpenStudioConfig.load(args.config)
    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        config.llm_model = args.model

    if args.list_models:
        print("\n🤖 Supported LLM providers and example models:\n")
        print("  ollama       (local)  — gemma:2b, gemma, llama3.1, mistral, phi3")
        print("  gemini       (cloud)  — gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash")
        print("  ollama_cloud (cloud)  — any model hosted at your OLLAMA_CLOUD_URL")
        print("  openai       (cloud)  — gpt-4o, gpt-4o-mini, gpt-3.5-turbo\n")
        try:
            import ollama as _ollama
            response = _ollama.list()
            models = response.models if hasattr(response, 'models') else response.get("models", [])
            if models:
                print("  📦 Locally available Ollama models:")
                for m in models:
                    name = m.model if hasattr(m, 'model') else m.get('name', str(m))
                    size_gb = (m.size / 1e9 if hasattr(m, 'size') and m.size else 0)
                    size_str = f"  ({size_gb:.1f} GB)" if size_gb else ""
                    print(f"     • {name}{size_str}")
        except Exception:
            print("  (Ollama not reachable — cannot list local models)")
        print()
        return

    if args.pull:
        try:
            import ollama as _ollama
            print(f"⬇️  Pulling '{args.pull}'...")
            for chunk in _ollama.pull(args.pull, stream=True):
                status = chunk.get("status", "") if isinstance(chunk, dict) else getattr(chunk, 'status', '')
                total = chunk.get("total", 0) if isinstance(chunk, dict) else getattr(chunk, 'total', 0)
                completed = chunk.get("completed", 0) if isinstance(chunk, dict) else getattr(chunk, 'completed', 0)
                if total and completed:
                    pct = int(completed / total * 100)
                    print(f"\r  {status}: {pct}%  ", end="", flush=True)
                elif status:
                    print(f"\r  {status}...    ", end="", flush=True)
            print(f"\n✅ '{args.pull}' is ready.\n")
        except Exception as e:
            print(f"\n❌ Failed to pull '{args.pull}': {e}")
        return

    # Auto-pull Ollama model if not available locally
    if config.llm_provider == "ollama" and config.llm_model:
        try:
            import ollama as _ollama
            response = _ollama.list()
            models = response.models if hasattr(response, 'models') else response.get("models", [])
            local_names = set()
            for m in models:
                name = m.model if hasattr(m, 'model') else m.get('name', '')
                local_names.add(name)
                local_names.add(name.split(":")[0])  # also match without tag
            model_tag = config.llm_model if ":" in config.llm_model else f"{config.llm_model}:latest"
            if model_tag not in local_names and config.llm_model not in local_names:
                print(f"⬇️  Model '{config.llm_model}' not found locally — pulling from Ollama...")
                for chunk in _ollama.pull(config.llm_model, stream=True):
                    status = chunk.get("status", "") if isinstance(chunk, dict) else getattr(chunk, 'status', '')
                    total = chunk.get("total", 0) if isinstance(chunk, dict) else getattr(chunk, 'total', 0)
                    completed = chunk.get("completed", 0) if isinstance(chunk, dict) else getattr(chunk, 'completed', 0)
                    if total and completed:
                        pct = int(completed / total * 100)
                        print(f"\r  {status}: {pct}%", end="", flush=True)
                    elif status:
                        print(f"\r  {status}...", end="", flush=True)
                print(f"\n✅ Model '{config.llm_model}' ready.\n")
        except Exception as e:
            logger.warning(f"Could not auto-pull model '{config.llm_model}': {e}")

    # Animated init display — only when entering interactive REPL
    _entering_repl = (
        not args.query and not getattr(args, 'ingest', None)
        and not args.list and not args.list_models
        and not getattr(args, 'pull', None)
        and sys.stdin.isatty()
    )
    _init_display: Optional[_InitDisplay] = None
    _saved_propagate: dict = {}
    _INIT_LOGGER_NAMES = [
        "RAGBrain", "StudioBrainOpen.Retrievers",
        "sentence_transformers.SentenceTransformer", "sentence_transformers",
        "chromadb", "chromadb.telemetry.product.posthog", "httpx",
    ]
    if _entering_repl:
        print()
        _init_display = _InitDisplay()
        for _n in _INIT_LOGGER_NAMES:
            _lg = logging.getLogger(_n)
            _saved_propagate[_n] = _lg.propagate
            _lg.propagate = False           # suppress default stderr handler
            _lg.setLevel(logging.INFO)
            _lg.addHandler(_init_display)

    brain = OpenStudioBrain(config)

    if _init_display:
        _init_display.stop()
        for _n in _INIT_LOGGER_NAMES:
            _lg = logging.getLogger(_n)
            _lg.removeHandler(_init_display)
            _lg.propagate = _saved_propagate.get(_n, True)
    
    if args.ingest:
        if os.path.isdir(args.ingest): asyncio.run(brain.load_directory(args.ingest))
        else:
            from rag_brain.loaders import DirectoryLoader
            ext = os.path.splitext(args.ingest)[1].lower()
            loader_mgr = DirectoryLoader()
            if ext in loader_mgr.loaders: brain.ingest(loader_mgr.loaders[ext].load(args.ingest))

    if args.list:
        docs = brain.list_documents()
        if not docs:
            print("📭 Knowledge base is empty.")
        else:
            total_chunks = sum(d["chunks"] for d in docs)
            print(f"\n📚 Knowledge Base — {len(docs)} file(s), {total_chunks} chunk(s)\n")
            print(f"  {'Source':<60} {'Chunks':>6}")
            print(f"  {'-'*60} {'-'*6}")
            for d in docs:
                print(f"  {d['source']:<60} {d['chunks']:>6}")
        return
    
    if args.query:
        if args.stream:
            for chunk in brain.query_stream(args.query):
                if isinstance(chunk, dict):
                    continue
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"\n📝 Response:\n{brain.query(args.query)}")
        return

    # No query supplied — enter interactive REPL (streaming on by default)
    _quiet = args.quiet or not sys.stdin.isatty()
    try:
        _interactive_repl(brain, stream=True, init_display=_init_display, quiet=_quiet)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n👋 Bye!")
    # Manually flush readline history, then hard-exit to skip atexit handlers
    # (colorama/posthog atexit callbacks raise tracebacks on double Ctrl+C)
    try:
        import readline as _rl
        _hist = os.path.expanduser("~/.rag_brain_history")
        _rl.write_history_file(_hist)
    except Exception:
        pass
    os._exit(0)


_SLASH_COMMANDS = [
    "/help", "/list", "/ingest ", "/model ", "/embed ",
    "/pull ", "/search", "/discuss", "/rag ", "/compact",
    "/context", "/sessions", "/resume ", "/clear", "/retry",
    "/project", "/project ", "/keys", "/quit", "/exit",
]


def _make_completer(brain: 'OpenStudioBrain'):
    """Return a readline completer for slash commands, paths, and model names."""
    def completer(text: str, state: int):
        try:
            import readline
            full_line = readline.get_line_buffer()

            # Completing a slash command name
            if full_line.startswith("/") and " " not in full_line:
                matches = [c for c in _SLASH_COMMANDS if c.startswith(full_line)]
                return matches[state] if state < len(matches) else None

            # /ingest <path|glob> — complete filesystem paths
            if full_line.startswith("/ingest "):
                path_prefix = full_line[len("/ingest "):]
                import glob as _glob
                matches = _glob.glob(path_prefix + "*")
                # Append / to directories
                matches = [m + "/" if os.path.isdir(m) else m for m in matches]
                return matches[state] if state < len(matches) else None

            # /model or /pull — complete Ollama model names
            if full_line.startswith("/model ") or full_line.startswith("/pull "):
                model_prefix = full_line.split(" ", 1)[1]
                try:
                    import ollama as _ollama
                    response = _ollama.list()
                    all_models = response.models if hasattr(response, 'models') else response.get("models", [])
                    names = [m.model if hasattr(m, 'model') else m.get('name', '') for m in all_models]
                    matches = [n for n in names if n.startswith(model_prefix)]
                    return matches[state] if state < len(matches) else None
                except Exception:
                    return None

        except Exception:
            return None
        return None

    return completer


# ── Session persistence ────────────────────────────────────────────────────────
import json as _json
from datetime import datetime as _dt, timezone as _tz

_SESSIONS_DIR = os.path.join(os.path.expanduser("~"), ".rag_brain", "sessions")


def _sessions_dir() -> str:
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    return _SESSIONS_DIR


def _new_session(brain: 'OpenStudioBrain') -> dict:
    return {
        "id": _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S%f")[:-3],
        "started_at": _dt.now(_tz.utc).isoformat(),
        "provider": brain.config.llm_provider,
        "model": brain.config.llm_model,
        "history": [],
    }


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"session_{session_id}.json")


def _save_session(session: dict) -> None:
    try:
        with open(_session_path(session["id"]), "w", encoding="utf-8") as f:
            _json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _list_sessions(limit: int = 20) -> list:
    d = _sessions_dir()
    files = sorted(
        [f for f in os.listdir(d) if f.startswith("session_") and f.endswith(".json")],
        reverse=True,
    )[:limit]
    sessions = []
    for fn in files:
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                s = _json.load(f)
            sessions.append(s)
        except Exception:
            pass
    return sessions


def _load_session(session_id: str) -> dict | None:
    p = _session_path(session_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _print_sessions(sessions: list) -> None:
    if not sessions:
        print("  (no saved sessions)")
        return
    print(f"\n  {'ID':<18}  {'Model':<30}  {'Turns':<6}  Started")
    print(f"  {'─'*18}  {'─'*30}  {'─'*6}  {'─'*20}")
    for s in sessions:
        turns = len(s.get("history", [])) // 2
        ts    = s.get("started_at", "")[:16].replace("T", " ")
        model = f"{s.get('provider','?')}/{s.get('model','?')}"
        print(f"  {s['id']:<18}  {model:<30}  {turns:<6}  {ts}")
    print()


_MODEL_CTX: Dict[str, int] = {
    "gemma": 8192, "gemma:2b": 8192, "gemma:7b": 8192,
    "llama3.1": 131072, "llama3.1:8b": 131072, "llama3.1:70b": 131072,
    "mistral": 32768, "mistral:7b": 32768,
    "phi3": 131072, "phi3:mini": 131072,
    "gemini-1.5-flash": 1048576, "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1048576, "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite": 1048576,
    "gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-3.5-turbo": 16385,
}


def _infer_provider(model: str) -> str:
    """Guess LLM provider from model name.

    Returns "gemini" for gemini-* models, "openai" for gpt-*/o1-*/o3-*/o4-*
    models (without a colon, since Ollama uses name:tag format), and "ollama"
    for everything else (local models, including gpt-oss:tag Ollama models).
    """
    m = model.lower()
    if m.startswith("gemini-"):
        return "gemini"
    # OpenAI model names never contain ':'; Ollama uses name:tag format.
    if ":" not in m and m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    return "ollama"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text)."""
    return max(1, len(text) // 4)


def _token_bar(used: int, total: int, width: int = 20) -> str:
    """Return a visual fill bar: ████░░░░ 2,340 / 8,192 (28%)."""
    pct = min(used / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "🟢" if pct < 0.6 else ("🟡" if pct < 0.85 else "🔴")
    return f"{color} {bar}  {used:,} / {total:,} ({int(pct*100)}%)"


def _show_context(
    brain: 'OpenStudioBrain',
    chat_history: list,
    last_sources: list,
    last_query: str,
) -> None:
    """Display a formatted context window panel with token usage, model info, and chat history.

    Shows:
    - Model info: LLM provider/model and context window size; embedding provider/model
    - Token usage: Rough estimates (4 chars/token) with visual bar and color indicator
    - RAG settings: top_k, similarity_threshold, hybrid_search, rerank, hyde, multi_query toggles
    - Chat history: Last 10 turns (user/assistant messages)
    - Last retrieved sources: Up to 8 chunks with similarity scores and source names
    - System prompt: First 400 characters (preview)

    All content is wrapped in a box with section separators for readability.

    Args:
        brain: OpenStudioBrain instance to extract model and config info.
        chat_history: List of message dicts {"role": "user"|"assistant", "content": str}.
        last_sources: List of document dicts from last retrieval (with "vector_score", "metadata", "text").
        last_query: The user query that was used for the last retrieval.
    """
    W      = _BW        # match main header box width
    TOP    = f"  ╭{'─' * W}╮"
    BOTTOM = f"  ╰{'─' * W}╯"
    SEP    = f"  ├{'─' * W}┤"
    BLANK  = f"  │{' ' * W}│"

    def row(text: str = "", indent: int = 4) -> str:
        content = " " * indent + text
        if len(content) > W:
            content = content[:W - 1] + "…"
        return f"  │{content:<{W}}│"

    def section(title: str) -> str:
        content = f"  ▸  {title}"
        return f"  │{content:<{W}}│"

    def wrap_row(text: str, indent: int = 4, max_lines: int = 3) -> list:
        """Word-wrap text into multiple box rows."""
        avail = W - indent
        words = text.split()
        lines_out, current = [], ""
        for w in words:
            if len(current) + len(w) + (1 if current else 0) <= avail:
                current = f"{current} {w}" if current else w
            else:
                if current:
                    lines_out.append(row(current, indent))
                current = w
        if current:
            lines_out.append(row(current, indent))
        return lines_out[:max_lines]

    # ── Token estimates ────────────────────────────────────────────────────────
    system_text = brain._build_system_prompt(False)
    sys_tokens  = _estimate_tokens(system_text)
    hist_tokens = sum(_estimate_tokens(m["content"]) for m in chat_history)
    src_tokens  = sum(_estimate_tokens(s.get("text", "")) for s in last_sources)
    total_used  = sys_tokens + hist_tokens + src_tokens

    model_key = brain.config.llm_model.split(":")[0].lower()
    ctx_size  = _MODEL_CTX.get(brain.config.llm_model,
                _MODEL_CTX.get(model_key, 8192))
    remaining = max(0, ctx_size - total_used)
    pct       = min(total_used / ctx_size, 1.0) if ctx_size > 0 else 0
    bar_w     = W - 26
    filled    = int(pct * bar_w)
    bar       = "█" * filled + "░" * (bar_w - filled)
    indicator = "🟢" if pct < 0.6 else ("🟡" if pct < 0.85 else "🔴")

    lines = [TOP, BLANK]
    lines.append(row("📋  Context Window", indent=4))
    lines.append(BLANK)

    # ── Model section ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("Model"))
    lines.append(BLANK)
    lines.append(row(f"LLM    ·  {brain.config.llm_provider}/{brain.config.llm_model}"
                     f"   ({ctx_size:,} token context window)"))
    lines.append(row(f"Embed  ·  {brain.config.embedding_provider}/{brain.config.embedding_model}"))
    lines.append(BLANK)

    # ── Token usage ───────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("Token Usage  (rough estimate — ~4 chars/token)"))
    lines.append(BLANK)
    lines.append(row(f"{indicator} {bar}  {total_used:,} / {ctx_size:,}  ({int(pct*100)}%)", indent=4))
    lines.append(BLANK)
    lines.append(row(f"{'System prompt':<22}{sys_tokens:>7,} tokens"))
    lines.append(row(f"{'Chat history':<22}{hist_tokens:>7,} tokens    ({len(chat_history) // 2} turns)"))
    lines.append(row(f"{'Retrieved context':<22}{src_tokens:>7,} tokens    ({len(last_sources)} chunks)"))
    lines.append(row("─" * 40))
    lines.append(row(f"{'Total':<22}{total_used:>7,} tokens    ({remaining:,} remaining)"))
    lines.append(BLANK)

    # ── RAG settings ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("RAG Settings"))
    lines.append(BLANK)
    lines.append(row(
        f"top-k · {brain.config.top_k}    "
        f"threshold · {brain.config.similarity_threshold}    "
        f"hybrid · {'ON' if brain.config.hybrid_search else 'OFF'}    "
        f"rerank · {'ON' if brain.config.rerank else 'OFF'}    "
        f"hyde · {'ON' if brain.config.hyde else 'OFF'}    "
        f"multi-query · {'ON' if brain.config.multi_query else 'OFF'}"
    ))
    lines.append(BLANK)

    # ── Chat history ───────────────────────────────────────────────────────────
    lines.append(SEP)
    turns = len(chat_history) // 2
    lines.append(section(f"Chat History  ({turns} turns)"))
    lines.append(BLANK)
    if not chat_history:
        lines.append(row("(empty)"))
    else:
        shown = chat_history[-10:]
        for msg in shown:
            tag  = "You   " if msg["role"] == "user" else "Brain "
            snip = msg["content"].replace("\n", " ")
            avail = W - 14
            if len(snip) > avail:
                snip = snip[:avail] + "…"
            lines.append(row(f"{tag}  {snip}"))
        if len(chat_history) > 10:
            lines.append(row(f"… {len(chat_history) - 10} earlier messages not shown"))
    lines.append(BLANK)

    # ── Last retrieved sources ─────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section(f"Last Retrieved Sources  ({len(last_sources)} chunks)"))
    lines.append(BLANK)
    if last_query:
        lines.append(row(f'query · "{last_query[:W - 14]}"'))
        lines.append(BLANK)
    if not last_sources:
        lines.append(row("(no retrieval yet)"))
    else:
        for i, src in enumerate(last_sources[:8], 1):
            meta  = src.get("metadata", {})
            name  = os.path.basename(meta.get("source", src.get("id", "?")))
            score = src.get("vector_score", src.get("score", 0))
            kind  = "🌐" if src.get("is_web") else "📄"
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(row(f"{i:>2}. {kind} {score_bar} {score:.3f}   {name}"))
    lines.append(BLANK)

    # ── System prompt preview ──────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("System Prompt  (preview)"))
    lines.append(BLANK)
    lines.extend(wrap_row(system_text[:400].replace("\n", " "), indent=4, max_lines=4))
    lines.append(BLANK)
    lines.append(BOTTOM)

    for line in lines:
        print(line)
    print()


def _do_compact(brain: 'OpenStudioBrain', chat_history: list) -> None:
    """Summarize chat history via LLM and replace it with a single summary turn.

    Condenses all messages in chat_history into a 4-6 sentence summary using the configured LLM.
    The original conversation is replaced with a single message prefixed with
    "[Conversation summary]: " to preserve context while freeing up token space.

    If chat_history is empty, prints a message and returns without action.

    Args:
        brain: OpenStudioBrain instance used to call the LLM for summarization.
        chat_history: List of message dicts to summarize (modified in-place; emptied and refilled with summary).
    """
    if not chat_history:
        print("  Nothing to compact — chat history is empty.")
        return

    turns_before = len(chat_history)
    print(f"  ⠿ Compacting {turns_before} turns…", end="", flush=True)

    conversation = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history
    )
    summary_prompt = (
        "Summarize the following conversation in 4-6 concise sentences. "
        "Preserve all key facts, decisions, and topics discussed. "
        "Write in third person ('The user asked…'). "
        "Output only the summary, no preamble.\n\n"
        f"{conversation}"
    )
    try:
        summary = brain.llm.complete(summary_prompt, system_prompt=None, chat_history=[])
        chat_history.clear()
        chat_history.append({"role": "assistant", "content": f"[Conversation summary]: {summary}"})
        tokens_saved = _estimate_tokens(conversation) - _estimate_tokens(summary)
        print(f"\r  ✅ Compacted {turns_before} turns → 1 summary  (~{tokens_saved:,} tokens freed)")
    except Exception as e:
        print(f"\r  ❌ Compact failed: {e}")


# ── Banner constants ───────────────────────────────────────────────────────────
_BW = 112         # inner box width in terminal columns
_HINT = "  Type your question  ·  /help for commands  ·  Tab to autocomplete"
_SEP  = "  " + "─" * (_BW + 2)   # separator line matching box outer width
_HEADER_ROWS = 16  # box(12) + blank(1) + hint(1) + sep(1) + blank(1)


def _brow(content: str, emoji_extra: int = 0) -> str:
    """One box row: pads/truncates content to exactly _BW terminal columns."""
    vis = len(content) + emoji_extra
    if vis > _BW:
        content = content[:_BW - emoji_extra - 1] + "…"
        vis = _BW
    pad = _BW - vis
    return f"  │{content}{' ' * pad}│"


def _build_header(brain: 'OpenStudioBrain', tick_lines: list | None = None) -> list:
    """Return lines of the pinned header box (airy layout)."""
    model_s   = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    embed_s   = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
    search_s  = "ON  (Brave Search)" if brain.config.truth_grounding  else "OFF"
    discuss_s = "ON" if brain.config.discussion_fallback else "OFF"
    hybrid_s  = "ON" if brain.config.hybrid_search else "OFF"
    topk_s    = str(brain.config.top_k)
    thr_s     = str(brain.config.similarity_threshold)
    try:
        docs  = brain.list_documents()
        doc_s = f"{sum(d['chunks'] for d in docs)} chunks  ({len(docs)} files)"
    except Exception:
        doc_s = "unknown"

    # Build tick status — wrap onto a second row if too wide
    tick_items = [f"✓ {t}" for t in tick_lines] if tick_lines else ["✓ Ready"]
    ticks_s = "   ".join(tick_items)
    inner_w = _BW - 4  # 4-char left indent "    "
    if len(ticks_s) > inner_w:
        # Split into two roughly equal halves at a separator boundary
        mid = len(tick_items) // 2
        ticks_s  = "   ".join(tick_items[:mid])
        ticks_s2 = "   ".join(tick_items[mid:])
    else:
        ticks_s2 = None

    blank = f"  │{' ' * _BW}│"
    rows = [
        f"  ╭{'─' * _BW}╮",                                                      # 1
        blank,                                                                     # 2
        _brow("    🧠  Local RAG Brain", emoji_extra=1),                          # 3
        blank,                                                                     # 4
        _brow(f"    LLM    ·  {model_s}"),                                        # 5
        _brow(f"    Embed  ·  {embed_s}"),                                        # 6
        blank,                                                                     # 7
        _brow(f"    Search ·  {search_s:<26}  Discuss  ·  {discuss_s}"),         # 8
        _brow(f"    Docs   ·  {doc_s:<26}  Hybrid   ·  {hybrid_s}   top-k · {topk_s}   threshold · {thr_s}"),  # 9
        blank,                                                                     # 10
        _brow(f"    {ticks_s}"),                                                   # 11
    ]
    if ticks_s2:
        rows.append(_brow(f"    {ticks_s2}"))                                     # 11b (overflow)
    rows.append(f"  ╰{'─' * _BW}╯")                                              # 12
    return rows


def _draw_header(brain: 'OpenStudioBrain', tick_lines: list | None = None) -> None:
    """Clear screen and draw the welcome header box with LLM and embedding model info.

    Displays initialization status lines (e.g., "✓ Embedding ready [CPU]", "✓ BM25 · 42 docs").
    Clears the entire screen and redraws the header with hints for available REPL commands.
    Uses ANSI codes to clear and position the cursor — no scroll region (natural terminal scrollback).

    Args:
        brain: OpenStudioBrain instance to extract model and provider information.
        tick_lines: Optional list of status messages (e.g., ["Starting", "Embedding ready [CPU]"])
                   to display in the header box.
    """
    lines = _build_header(brain, tick_lines)
    sys.stdout.write("\033[2J\033[H")          # clear screen, cursor to top-left
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.write("\n" + _HINT + "\n" + _SEP + "\n\n")
    sys.stdout.flush()


def _print_recent_turns(history: list, n_turns: int = 2) -> None:
    """Print the last n_turns of Q&A below the header so context is visible.

    Args:
        history: chat_history list of {"role": ..., "content": ...} dicts.
        n_turns: Number of complete Q&A turns to show (each turn = 1 user + 1 assistant message).
    """
    if not history:
        return
    recent = history[-(n_turns * 2):]
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            sys.stdout.write(f"  \033[1;32mYou\033[0m: {content}\n")
        elif role == "assistant":
            # Cap very long responses so they don't flood the screen
            if len(content) > 600:
                content = content[:600] + "…"
            sys.stdout.write(f"\n  \033[1;33mBrain\033[0m:\n  {content}\n")
        sys.stdout.write("\n")
    sys.stdout.flush()


class _InitDisplay(logging.Handler):
    """Intercepts initialization log messages and renders animated status in a box.

    Displays a 7-line box with title and status line updated in-place using ANSI cursor positioning.
    Uses a braille spinner (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) that rotates every 0.08 seconds.
    Collects completed steps as checkmarks (✓) for the final banner display.

    The box is printed once at initialization, then the step line (line 5) is updated in-place
    as different initialization phases complete (Starting, Loading models, Vector store ready, etc.).
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        super().__init__()
        self._step: str = ""
        self._idx:  int = 0
        self._lock  = threading.Lock()
        self._done  = threading.Event()
        self.tick_lines: list = []    # collected for the final banner
        # Print CLOSED 7-line box immediately — step line updated in-place
        sys.stdout.write(
            f"\n  ╭{'─' * _BW}╮\n"
            f"  │{' ' * _BW}│\n"
            f"  │{'    🧠  Local RAG Brain'.ljust(_BW - 1)}│\n"
            f"  │{' ' * _BW}│\n"
            f"  │{'    ⠿  Initializing…'.ljust(_BW)}│\n"   # step line (line 5)
            f"  │{' ' * _BW}│\n"
            f"  ╰{'─' * _BW}╯\n"
        )
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    def _spin_loop(self) -> None:
        while not self._done.wait(0.08):
            with self._lock:
                if self._step:
                    frame = self._FRAMES[self._idx % len(self._FRAMES)]
                    content = f"    {frame}  {self._step}"
                    line = _brow(content)
                    # Box is 7 lines; step is line 5; cursor after ╰╯ is at line 8.
                    # Up 3 → line 5; write; newline → line 6; down 2 → line 8.
                    sys.stdout.write(f"\033[3A\r{line}\n\033[2B")
                    sys.stdout.flush()
                    self._idx += 1

    def _tick(self, label: str) -> None:
        with self._lock:
            self._step = ""
            self.tick_lines.append(label)
            line = _brow(f"    ✓  {label}")
            sys.stdout.write(f"\033[3A\r{line}\n\033[2B")
            sys.stdout.flush()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Initializing Local RAG Brain" in msg:
            with self._lock:
                self._step = "Starting…"
        elif "Loading Sentence Transformers" in msg:
            m = re.search(r":\s*(.+)$", msg)
            with self._lock:
                self._step = f"Loading {m.group(1).strip() if m else 'model'}…"
        elif "Use pytorch device_name" in msg:
            m = re.search(r":\s*(.+)$", msg)
            self._tick(f"Embedding ready  [{m.group(1).strip() if m else 'cpu'}]")
        elif "Initializing ChromaDB" in msg:
            with self._lock:
                self._step = "Vector store…"
        elif "Loaded BM25 corpus" in msg:
            m = re.search(r"(\d+) documents", msg)
            self._tick("Vector store ready")
            self._tick(f"BM25  ·  {m.group(1) if m else '?'} docs")
        elif "Local RAG Brain ready" in msg:
            self._done.set()

    def stop(self) -> None:
        self._done.set()
        with self._lock:
            self._step = ""
        self._thread.join(timeout=0.5)


def _expand_at_files(text: str) -> str:
    """Expand @filepath references in user input with file contents."""
    def _replace(m: re.Match) -> str:
        path = m.group(1)
        if os.path.isfile(path):
            try:
                content = open(path, encoding="utf-8", errors="ignore").read()
                return f"\n\n--- @{path} ---\n{content}\n--- end ---\n"
            except OSError:
                pass
        return m.group(0)  # leave unchanged if file not found/readable
    return re.sub(r'@(\S+)', _replace, text)


def _interactive_repl(brain: 'OpenStudioBrain', stream: bool = True,
                      init_display: '_InitDisplay | None' = None,
                      quiet: bool = False) -> None:
    """Interactive REPL chat session with session persistence and live tab completion.

    Features:
    - Session persistence: auto-saves to ~/.rag_brain/sessions/session_<timestamp>.json
    - Live tab completion: slash commands, filesystem paths, Ollama model names via prompt_toolkit
    - Animated spinners: braille spinner during init and LLM generation (disabled in quiet mode)
    - Slash commands: /help, /list, /ingest, /model, /embed, /pull, /search, /discuss, /rag,
      /compact, /context, /sessions, /resume, /retry, /clear, /quit, /exit
    - @file context: type @path to inline file contents into your query
    - Shell passthrough: !command runs a shell command without leaving the REPL
    - Pinned status info: token usage, model info, RAG settings visible at terminal bottom

    Args:
        brain: OpenStudioBrain instance to use for queries.
        stream: If True, streams LLM response token-by-token; if False, waits for full response.
        init_display: Optional _InitDisplay handler to stop after initialization.
        quiet: Suppress spinners and progress bars (auto-enabled for non-TTY stdin).
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="google")

    # Silence INFO logs — they clutter the interactive UI
    import logging as _logging
    for _log in ("RAGBrain", "StudioBrainOpen.Retrievers", "httpx",
                 "sentence_transformers", "chromadb", "httpcore"):
        _lg = _logging.getLogger(_log)
        _lg.setLevel(_logging.WARNING)
        _lg.propagate = False   # prevent bubbling to root logger

    # ── Input: prefer prompt_toolkit (live completions), fall back to readline ──
    _pt_session = None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.styles import Style
        from prompt_toolkit.formatted_text import HTML as _PThtml
        from prompt_toolkit.history import FileHistory as _FileHistory
        import glob as _pglob

        _HIST_DIR = os.path.expanduser("~/.rag_brain")
        os.makedirs(_HIST_DIR, exist_ok=True)
        _HIST_FILE = os.path.join(_HIST_DIR, "repl_history")

        _PT_STYLE = Style.from_dict({
            "": "",
            "completion-menu.completion.current": "bg:#444466 #ffffff",
            "bottom-toolbar":        "bg:#1a1a2e #c8c8e8",
            "bottom-toolbar.key":    "bg:#1a1a2e #7070cc bold",
            "bottom-toolbar.on":     "bg:#1a1a2e #66cc66",
            "bottom-toolbar.off":    "bg:#1a1a2e #666688",
            "bottom-toolbar.sep":    "bg:#1a1a2e #444466",
        })

        class _PTCompleter(Completer):
            def __init__(self, brain_ref):
                self._brain = brain_ref

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                # ── slash command name ─────────────────────────────────────
                if text.startswith("/") and " " not in text:
                    for cmd in _SLASH_COMMANDS:
                        c = cmd.rstrip()
                        if c.startswith(text):
                            yield Completion(c[len(text):], display=c,
                                             display_meta="command")
                # ── /ingest path / glob ───────────────────────────────────
                elif text.startswith("/ingest "):
                    prefix = text[len("/ingest "):]
                    for p in _pglob.glob(prefix + "*"):
                        disp = p + ("/" if os.path.isdir(p) else "")
                        yield Completion(p[len(prefix):], display=disp)
                # ── /model <provider/model> ───────────────────────────────
                elif text.startswith("/model ") or text.startswith("/embed "):
                    cmd_len = len("/model ") if text.startswith("/model ") else len("/embed ")
                    prefix  = text[cmd_len:]
                    try:
                        import ollama as _ol
                        resp = _ol.list()
                        mods = resp.models if hasattr(resp, "models") else resp.get("models", [])
                        for m in mods:
                            name = m.model if hasattr(m, "model") else m.get("name", "")
                            if name.startswith(prefix):
                                yield Completion(name[len(prefix):], display=name)
                    except Exception:
                        pass
                # ── /resume <session-id> ──────────────────────────────────
                elif text.startswith("/resume "):
                    prefix = text[len("/resume "):]
                    for s in _list_sessions():
                        sid = s["id"]
                        if sid.startswith(prefix):
                            turns = len(s.get("history", [])) // 2
                            yield Completion(sid[len(prefix):], display=sid,
                                             display_meta=f"{turns} turns")
                # ── /rag <option> ─────────────────────────────────────────
                elif text.startswith("/rag "):
                    opts = ["topk ", "threshold ", "hybrid", "rerank", "hyde", "multi"]
                    prefix = text[len("/rag "):]
                    for o in opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix):], display=o)
                # ── @file context attachment ──────────────────────────────
                elif "@" in text:
                    at_pos = text.rfind("@")
                    prefix = text[at_pos + 1:]
                    for p in _pglob.glob(prefix + "*"):
                        disp = p + ("/" if os.path.isdir(p) else "")
                        yield Completion(p[len(prefix):], display=disp,
                                         display_meta="file context")
                # ── /project <subcommand> ──────────────────────────────────
                elif text.startswith("/project "):
                    sub_opts = ["list", "new ", "switch ", "delete ", "folder"]
                    prefix = text[len("/project "):]
                    for o in sub_opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix):], display=o)
                    # also complete project names for switch/delete
                    if text.startswith(("/project switch ", "/project delete ")):
                        cmd_len = len("/project switch ") if text.startswith("/project switch ") else len("/project delete ")
                        pfx = text[cmd_len:]
                        try:
                            from rag_brain.projects import list_projects
                            for p in list_projects():
                                n = p["name"]
                                if n.startswith(pfx):
                                    yield Completion(n[len(pfx):], display=n)
                        except Exception:
                            pass

        def _toolbar():
            m   = f"{brain.config.llm_provider}/{brain.config.llm_model}"
            emb = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
            try:
                docs = brain.vector_store.collection.count()
                doc_s = f"{docs} chunks"
            except Exception:
                doc_s = "?"
            proj = getattr(brain, "_active_project", "default")
            proj_s = f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  <bottom-toolbar.key>📂</bottom-toolbar.key> {proj}" if proj != "default" else ""
            s_color = "bottom-toolbar.on"  if brain.config.truth_grounding   else "bottom-toolbar.off"
            d_color = "bottom-toolbar.on"  if brain.config.discussion_fallback else "bottom-toolbar.off"
            h_color = "bottom-toolbar.on"  if brain.config.hybrid_search      else "bottom-toolbar.off"
            s_val   = "search:ON" if brain.config.truth_grounding   else "search:off"
            d_val   = "discuss:on" if brain.config.discussion_fallback else "discuss:off"
            h_val   = "hybrid:on" if brain.config.hybrid_search      else "hybrid:off"
            sep = "  <bottom-toolbar.sep>│</bottom-toolbar.sep>  "
            return HTML(
                f"  <bottom-toolbar.key>🧠 LLM</bottom-toolbar.key> {m}"
                f"{sep}<bottom-toolbar.key>Embed</bottom-toolbar.key> {emb}"
                f"{sep}<bottom-toolbar.key>Docs</bottom-toolbar.key> {doc_s}"
                f"{sep}<{s_color}>{s_val}</{s_color}>"
                f"{sep}<{d_color}>{d_val}</{d_color}>"
                f"{sep}<{h_color}>{h_val}</{h_color}>"
                f"{sep}top-k:{brain.config.top_k}  thr:{brain.config.similarity_threshold}"
                f"{proj_s}  "
            )

        _pt_session = PromptSession(
            completer=_PTCompleter(brain),
            auto_suggest=AutoSuggestFromHistory(),
            style=_PT_STYLE,
            complete_while_typing=True,
            bottom_toolbar=_toolbar,
            history=_FileHistory(_HIST_FILE),
        )
    except ImportError:
        # Fall back to readline with history persistence
        try:
            import readline
            import atexit
            _hist_file = os.path.expanduser("~/.rag_brain/repl_history")
            os.makedirs(os.path.dirname(_hist_file), exist_ok=True)
            try:
                readline.read_history_file(_hist_file)
            except FileNotFoundError:
                pass
            readline.set_history_length(2000)
            atexit.register(readline.write_history_file, _hist_file)
            readline.set_completer(_make_completer(brain))
            readline.set_completer_delims("")
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\C-l": clear-screen')
        except ImportError:
            pass

    def _read_input(prompt: str = "") -> str:
        if _pt_session:
            _p = _PThtml('<ansigreen><b>You</b></ansigreen>: ') if not prompt else prompt
            return _pt_session.prompt(_p)
        return input(prompt if prompt else '\033[1;32mYou\033[0m: ')

    # REPL is conversational — always let the LLM answer even with no RAG hits
    brain.config.discussion_fallback = True

    _tick_lines = init_display.tick_lines if init_display else []
    _draw_header(brain, _tick_lines)

    # ── Session init ───────────────────────────────────────────────────────────
    session: dict = _new_session(brain)
    chat_history: list = session["history"]

    # Offer to resume the most recent session (if within last 24 h and has turns)
    recent = _list_sessions(limit=1)
    if recent:
        last = recent[0]
        turns = len(last.get("history", [])) // 2
        if turns > 0:
            ts = last.get("started_at", "")[:16].replace("T", " ")
            try:
                ans = _read_input(
                    f"  ↩️  Resume last session? ({turns} turns, {ts} UTC)  [y/N]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans == "y":
                session = last
                chat_history = session["history"]
                print(f"  ✅ Resumed session {session['id']}  ({turns} turns loaded)\n")

    _last_sources: list = []
    _last_query: str = ""

    while True:
        try:
            user_input = _read_input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        # --- Shell passthrough: !command ---
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                import subprocess
                subprocess.run(shell_cmd, shell=True)
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Allow "/<cmd> help" as alias for "/help <cmd>"
            if arg.strip().lower() == "help" and cmd not in ("/help",):
                arg = cmd.lstrip("/")
                cmd = "/help"
            if cmd in ("/quit", "/exit", "/q"):
                print("👋 Bye!")
                break

            elif cmd == "/help":
                if arg:
                    # Per-command detail
                    _detail = {
                        "model":    "  /model <model>              keep current provider\n"
                                    "  /model <provider>/<model>   switch provider + model\n"
                                    "  providers: ollama, gemini, openai, ollama_cloud\n"
                                    "  e.g.  /model gemini/gemini-2.0-flash\n"
                                    "        /model ollama/gemma:2b\n"
                                    "        /model openai/gpt-4o",
                        "embed":    "  /embed <model>              keep current provider\n"
                                    "  /embed <provider>/<model>   switch provider + model\n"
                                    "  /embed /path/to/local        local HuggingFace folder\n"
                                    "  providers: sentence_transformers, ollama, fastembed, openai\n"
                                    "  ⚠️  Re-ingest after changing embedding model.",
                        "ingest":   "  /ingest <path>              ingest a directory\n"
                                    "  /ingest ./src/*.py           glob pattern\n"
                                    "  /ingest ./notes/**/*.md      recursive glob",
                        "rag":      "  /rag                         show all RAG settings\n"
                                    "  /rag topk <n>                results to retrieve (1–20)\n"
                                    "  /rag threshold <0.0–1.0>     min similarity score\n"
                                    "  /rag hybrid                  toggle hybrid BM25+vector\n"
                                    "  /rag rerank                  toggle cross-encoder reranker\n"
                                    "  /rag hyde                    toggle HyDE query expansion\n"
                                    "  /rag multi                   toggle multi-query expansion",
                        "sessions": "  /sessions                    list recent saved sessions\n"
                                    "  /resume <id>                 load a session by ID\n"
                                    "  Sessions auto-save after each turn.",
                        "keys":     "  /keys                        show API key status for all providers\n"
                                    "  /keys set <provider>         interactively set an API key\n"
                                    "  providers: gemini, openai, brave, ollama_cloud\n"
                                    "  Keys are saved to ~/.rag_brain/.env and loaded at startup.",
                        "project":  "  /project                     show active project + list all\n"
                                    "  /project list                 list all projects\n"
                                    "  /project new <name>           create a new project and switch to it\n"
                                    "  /project new <name> <desc>    create with description\n"
                                    "  /project switch <name>        switch to an existing project\n"
                                    "  /project switch default       return to the global knowledge base\n"
                                    "  /project delete <name>        delete a project and all its data\n"
                                    "  /project folder               open the active project folder\n"
                                    "\n"
                                    "  Projects are stored in ~/.rag_brain/projects/<name>/\n"
                                    "  Each project has its own vector store and BM25 index.\n"
                                    "  Use /ingest after switching to add documents to a project.",
                    }
                    key = arg.lstrip("/")
                    if key in _detail:
                        print(f"\n{_detail[key]}\n")
                    else:
                        print(f"  No detail for '{arg}'. Available: {', '.join(_detail)}")
                else:
                    print(
                        "\n"
                        "  Knowledge:  /list   /ingest <path>   /clear\n"
                        "  Model:      /model [<prov/>model]   /embed [<prov/>model]   /pull <name>\n"
                        "  Mode:       /search   /discuss   /rag [option value]\n"
                        "  Session:    /sessions   /resume <id>   /compact   /context   /retry\n"
                        "  Projects:   /project [list|new|switch|delete|folder]\n"
                        "  API Keys:   /keys   /keys set <provider>\n"
                        "  Other:      /help [cmd]   /quit\n"
                        "  Shell:      !<cmd>  run a shell command  ·  @<path>  attach file context\n"
                        "\n"
                        "  /help <cmd>  for details (model, embed, ingest, rag, sessions, keys, project)\n"
                        "  /<cmd> help  also works  ·  e.g. /project help   /rag help\n"
                        "  Tab  autocomplete  ·  ↑↓  history  ·  Ctrl+C  cancel  ·  Ctrl+D  exit\n"
                    )

            elif cmd == "/list":
                docs = brain.list_documents()
                if not docs:
                    print("📭 Knowledge base is empty.")
                else:
                    total = sum(d["chunks"] for d in docs)
                    print(f"\n📚 {len(docs)} file(s), {total} chunk(s)\n")
                    for d in docs:
                        print(f"  {d['source']:<60} {d['chunks']:>6}")
                    print()

            elif cmd == "/ingest":
                if not arg:
                    print("  Usage: /ingest <path|glob>  e.g. /ingest ./docs  /ingest ./src/*.py")
                else:
                    import glob as _glob
                    from rag_brain.loaders import DirectoryLoader
                    # Expand glob pattern; fallback to literal path
                    matched = sorted(_glob.glob(arg, recursive=True))
                    if not matched:
                        # No glob match — try as plain directory
                        if os.path.isdir(arg):
                            matched = [arg]
                        else:
                            print(f"  ❌ No files matched: {arg}")
                    if matched:
                        loader_mgr = DirectoryLoader()
                        ingested, skipped = 0, 0
                        for path in matched:
                            if os.path.isdir(path):
                                print(f"  📁 {path} …", end="", flush=True)
                                asyncio.run(brain.load_directory(path))
                                print("  ✅")
                                ingested += 1
                            elif os.path.isfile(path):
                                ext = os.path.splitext(path)[1].lower()
                                if ext in loader_mgr.loaders:
                                    brain.ingest(loader_mgr.loaders[ext].load(path))
                                    print(f"  ✅ {path}")
                                    ingested += 1
                                else:
                                    print(f"  ⚠️  Skipped (unsupported type): {path}")
                                    skipped += 1
                        print(f"  Done — {ingested} ingested, {skipped} skipped.")

            elif cmd == "/model":
                _PROVIDERS = ('ollama', 'gemini', 'openai', 'ollama_cloud')
                if not arg:
                    print(f"  LLM:       {brain.config.llm_provider}/{brain.config.llm_model}")
                    print(f"  Embedding: {brain.config.embedding_provider}/{brain.config.embedding_model}")
                    print(f"  Usage:   /model <model>              (auto-detect provider)")
                    print(f"           /model <provider>/<model>   (switch provider too)")
                    print(f"  Providers: {', '.join(_PROVIDERS)}")
                elif "/" in arg:
                    provider, model = arg.split("/", 1)
                    if provider not in _PROVIDERS:
                        print(f"  ❌ Unknown provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}")
                    else:
                        brain.config.llm_provider = provider
                        brain.config.llm_model = model
                        brain.llm = OpenLLM(brain.config)
                        print(f"  ✅ Switched LLM to {provider}/{model}")
                        if provider != "ollama":
                            print(f"  ℹ️  Make sure the required API key env var is set.")
                else:
                    inferred = _infer_provider(arg)
                    brain.config.llm_provider = inferred
                    brain.config.llm_model = arg
                    brain.llm = OpenLLM(brain.config)
                    print(f"  ✅ Switched LLM to {inferred}/{arg}")
                    if inferred != "ollama":
                        print(f"  ℹ️  Make sure the required API key env var is set.")

            elif cmd == "/embed":
                _EMBED_PROVIDERS = ('sentence_transformers', 'ollama', 'fastembed', 'openai')
                if not arg:
                    print(f"  Current:   {brain.config.embedding_provider}/{brain.config.embedding_model}")
                    print(f"  Usage:   /embed <model>              (keep current provider)")
                    print(f"           /embed <provider>/<model>   (switch provider too)")
                    print(f"  Providers: {', '.join(_EMBED_PROVIDERS)}")
                    print(f"  Examples:")
                    print(f"    /embed all-MiniLM-L6-v2                    (sentence_transformers)")
                    print(f"    /embed /path/to/local/model                (local folder)")
                    print(f"    /embed ollama/nomic-embed-text")
                    print(f"    /embed fastembed/BAAI/bge-small-en")
                    print(f"  ⚠️  Changing embedding model invalidates existing indexed documents.")
                else:
                    if "/" in arg:
                        provider, model = arg.split("/", 1)
                        if provider not in _EMBED_PROVIDERS:
                            # Could be a path like /home/user/model — treat as local st path
                            provider = brain.config.embedding_provider
                            model = arg
                        else:
                            brain.config.embedding_provider = provider
                            brain.config.embedding_model = model
                    else:
                        provider = brain.config.embedding_provider
                        model = arg
                        brain.config.embedding_model = model
                    try:
                        print(f"  ⠿ Loading embedding model…", end="", flush=True)
                        brain.embedding = OpenEmbedding(brain.config)
                        print(f"\r  ✅ Embedding switched to {brain.config.embedding_provider}/{brain.config.embedding_model}")
                        print(f"  ⚠️  Re-ingest your documents so they use the new embedding model.")
                    except Exception as e:
                        print(f"\r  ❌ Failed to load embedding: {e}")

            elif cmd == "/pull":
                if not arg:
                    print("  Usage: /pull <model-name>")
                else:
                    try:
                        import ollama as _ollama
                        print(f"  ⬇️  Pulling '{arg}' …")
                        last_status = ""
                        for chunk in _ollama.pull(arg, stream=True):
                            status = chunk.get("status", "") if isinstance(chunk, dict) else getattr(chunk, 'status', '')
                            total = chunk.get("total", 0) if isinstance(chunk, dict) else getattr(chunk, 'total', 0)
                            completed = chunk.get("completed", 0) if isinstance(chunk, dict) else getattr(chunk, 'completed', 0)
                            if total and completed:
                                line = f"  {status}: {int(completed/total*100)}%"
                            elif status:
                                line = f"  {status}"
                            else:
                                continue
                            # Pad to clear previous longer line
                            print(f"\r{line:<60}", end="", flush=True)
                            last_status = line
                        print(f"\r  ✅ '{arg}' ready.{' ' * 50}")
                    except Exception as e:
                        print(f"  ❌ Pull failed: {e}")

            elif cmd == "/clear":
                chat_history.clear()
                print("  🗑️  Chat history cleared.")

            elif cmd == "/search":
                if brain.config.truth_grounding:
                    brain.config.truth_grounding = False
                    print("  🔍 Web search OFF — answers from local knowledge only.")
                else:
                    if not brain.config.brave_api_key:
                        print("  ❌ BRAVE_API_KEY is not set. Export it and restart, or set it with:")
                        print("     export BRAVE_API_KEY=your_key")
                    else:
                        brain.config.truth_grounding = True
                        print("  🔍 Web search ON — Brave Search will be used as fallback when local knowledge is insufficient.")

            elif cmd == "/discuss":
                brain.config.discussion_fallback = not brain.config.discussion_fallback
                state = "ON" if brain.config.discussion_fallback else "OFF"
                print(f"  💬 Discussion mode {state}.")

            elif cmd == "/rag":
                _RAG_TOGGLES = {"hybrid", "rerank", "hyde", "multi"}
                if not arg:
                    print(
                        f"\n  top-k        · {brain.config.top_k}\n"
                        f"  threshold    · {brain.config.similarity_threshold}\n"
                        f"  hybrid       · {'ON' if brain.config.hybrid_search else 'OFF'}\n"
                        f"  rerank       · {'ON' if brain.config.rerank else 'OFF'}\n"
                        f"  hyde         · {'ON' if brain.config.hyde else 'OFF'}\n"
                        f"  multi-query  · {'ON' if brain.config.multi_query else 'OFF'}\n"
                        f"\n  /help rag   for usage details\n"
                    )
                else:
                    rag_parts = arg.split(maxsplit=1)
                    rag_opt = rag_parts[0].lower()
                    rag_val = rag_parts[1] if len(rag_parts) > 1 else ""
                    if rag_opt == "topk":
                        try:
                            n = int(rag_val)
                            assert 1 <= n <= 50
                            brain.config.top_k = n
                            print(f"  ✅ top-k set to {n}")
                        except Exception:
                            print("  Usage: /rag topk <integer 1–50>")
                    elif rag_opt == "threshold":
                        try:
                            v = float(rag_val)
                            assert 0.0 <= v <= 1.0
                            brain.config.similarity_threshold = v
                            print(f"  ✅ threshold set to {v}")
                        except Exception:
                            print("  Usage: /rag threshold <float 0.0–1.0>")
                    elif rag_opt == "hybrid":
                        brain.config.hybrid_search = not brain.config.hybrid_search
                        print(f"  ✅ Hybrid search {'ON' if brain.config.hybrid_search else 'OFF'}")
                    elif rag_opt == "rerank":
                        brain.config.rerank = not brain.config.rerank
                        print(f"  ✅ Reranker {'ON' if brain.config.rerank else 'OFF'}")
                    elif rag_opt == "hyde":
                        brain.config.hyde = not brain.config.hyde
                        print(f"  ✅ HyDE {'ON' if brain.config.hyde else 'OFF'}")
                    elif rag_opt == "multi":
                        brain.config.multi_query = not brain.config.multi_query
                        print(f"  ✅ Multi-query {'ON' if brain.config.multi_query else 'OFF'}")
                    else:
                        print(f"  Unknown option '{rag_opt}'. Try: topk, threshold, hybrid, rerank, hyde, multi")

            elif cmd == "/compact":
                _do_compact(brain, chat_history)
                _save_session(session)

            elif cmd == "/project":
                from rag_brain.projects import (
                    delete_project, ensure_project,
                    list_projects, project_dir,
                )
                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

                if not sub or sub == "list":
                    projects = list_projects()
                    active = brain._active_project
                    if not projects:
                        print("  No projects yet. Use /project new <name> to create one.")
                    else:
                        print()
                        for p in projects:
                            marker = "●" if p["name"] == active else " "
                            ts = p["created_at"][:10] if p["created_at"] else ""
                            desc = f"  {p['description']}" if p["description"] else ""
                            print(f"  {marker} {p['name']:<24} {ts}{desc}")
                    print(f"\n  Active: {active}")
                    print("  /project new <name>    create + switch")
                    print("  /project switch <name> switch to existing")
                    print("  /project folder        open active project folder\n")

                elif sub == "new":
                    if not sub_arg:
                        print("  Usage: /project new <name>  [description]")
                    else:
                        name_parts = sub_arg.split(maxsplit=1)
                        proj_name = name_parts[0].lower()
                        proj_desc = name_parts[1] if len(name_parts) > 1 else ""
                        try:
                            ensure_project(proj_name, proj_desc)
                            brain.switch_project(proj_name)
                            print(f"  ✅ Created and switched to project '{proj_name}'")
                            print(f"  📁 {project_dir(proj_name)}")
                            print(f"  Use /ingest to add documents to this project.\n")
                        except ValueError as e:
                            print(f"  ❌ {e}")

                elif sub == "switch":
                    if not sub_arg:
                        print("  Usage: /project switch <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        if proj_name == "default" or project_dir(proj_name).exists():
                            try:
                                brain.switch_project(proj_name)
                                count = brain.vector_store.collection.count() if brain.vector_store.provider == "chroma" else "?"
                                print(f"  ✅ Switched to project '{proj_name}'  ({count} chunks)\n")
                            except Exception as e:
                                print(f"  ❌ {e}")
                        else:
                            print(f"  ❌ Project '{proj_name}' not found. Use /project list or /project new {proj_name}")

                elif sub == "delete":
                    if not sub_arg:
                        print("  Usage: /project delete <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        try:
                            confirm = _read_input(
                                f"  ⚠️  Delete project '{proj_name}' and ALL its data? [y/N]: "
                            ).strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            confirm = "n"
                        if confirm == "y":
                            try:
                                if brain._active_project == proj_name:
                                    brain.switch_project("default")
                                    print(f"  ↩️  Switched back to default project.")
                                delete_project(proj_name)
                                print(f"  ✅ Deleted project '{proj_name}'.\n")
                            except ValueError as e:
                                print(f"  ❌ {e}")
                        else:
                            print("  Cancelled.")

                elif sub == "folder":
                    active = brain._active_project
                    if active == "default":
                        print(f"  Default project uses config paths:")
                        print(f"    Vector store: {brain.config.vector_store_path}")
                        print(f"    BM25 index:   {brain.config.bm25_path}\n")
                    else:
                        folder = str(project_dir(active))
                        print(f"  📁 {folder}")
                        import subprocess
                        try:
                            if os.name == "nt":
                                subprocess.Popen(["explorer", folder])
                            elif sys.platform == "darwin":
                                subprocess.Popen(["open", folder])
                            else:
                                subprocess.Popen(["xdg-open", folder])
                        except Exception:
                            pass

                else:
                    print(f"  Unknown sub-command '{sub}'. Try: list, new, switch, delete, folder")

            elif cmd == "/retry":
                if not _last_query:
                    print("  Nothing to retry — no previous query.")
                else:
                    user_input = _last_query
                    print(f"  ↩️  Retrying: {user_input}")

            elif cmd == "/context":
                _show_context(brain, chat_history, _last_sources, _last_query)

            elif cmd == "/sessions":
                _print_sessions(_list_sessions())

            elif cmd == "/resume":
                if not arg:
                    print("  Usage: /resume <session-id>")
                else:
                    loaded = _load_session(arg)
                    if loaded is None:
                        print(f"  ❌ Session '{arg}' not found. Use /sessions to list.")
                    else:
                        session = loaded
                        chat_history.clear()
                        chat_history.extend(session["history"])
                        turns = len(chat_history) // 2
                        print(f"  ✅ Loaded session {session['id']}  ({turns} turns)\n")

            elif cmd == "/keys":
                _env_file = Path.home() / ".rag_brain" / ".env"
                _provider_keys = {
                    "gemini":       ("GEMINI_API_KEY",   "https://aistudio.google.com/app/apikey"),
                    "openai":       ("OPENAI_API_KEY",   "https://platform.openai.com/api-keys"),
                    "brave":        ("BRAVE_API_KEY",    "https://api.search.brave.com/app/keys"),
                    "ollama_cloud": ("OLLAMA_CLOUD_KEY", "https://ollama.com/settings"),
                }
                if arg.lower().startswith("set"):
                    set_parts = arg.split(maxsplit=1)
                    prov = set_parts[1].lower().strip() if len(set_parts) > 1 else ""
                    if not prov or prov not in _provider_keys:
                        print(f"  Usage: /keys set <provider>")
                        print(f"  Providers: {', '.join(_provider_keys)}")
                    else:
                        env_name, url = _provider_keys[prov]
                        print(f"  Get your key at: {url}")
                        try:
                            import getpass
                            new_key = getpass.getpass(f"  Enter {env_name} (hidden): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n  Cancelled.")
                        else:
                            if new_key:
                                _env_file.parent.mkdir(parents=True, exist_ok=True)
                                existing = _env_file.read_text(encoding="utf-8") if _env_file.exists() else ""
                                lines = [l for l in existing.splitlines() if not l.startswith(f"{env_name}=")]
                                lines.append(f"{env_name}={new_key}")
                                _env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                                os.environ[env_name] = new_key
                                if prov == "brave":
                                    brain.config.brave_api_key = new_key
                                elif prov == "gemini":
                                    brain.config.gemini_api_key = new_key
                                elif prov == "openai":
                                    brain.config.openai_api_key = new_key
                                elif prov == "ollama_cloud":
                                    brain.config.ollama_cloud_key = new_key
                                print(f"  ✅ {env_name} saved to {_env_file} and applied.")
                                print(f"  Switch provider: /model {prov}/<model-name>")
                            else:
                                print("  No key entered — nothing saved.")
                else:
                    print("\n  API Key Status\n  " + "─" * 50)
                    for prov, (env_name, url) in _provider_keys.items():
                        val = os.environ.get(env_name, "")
                        if val:
                            masked = val[:4] + "****" + val[-2:] if len(val) > 6 else "****"
                            status = f"✅ {masked}"
                        else:
                            status = "❌ not set"
                        print(f"  {prov:<14} {env_name:<22} {status}")
                    if _env_file.exists():
                        print(f"\n  Keys file: {_env_file}")
                    else:
                        print(f"\n  No keys file yet. Use /keys set <provider> to add keys.")
                    print(f"  /keys set <provider>  to set a key interactively")
                    print(f"  /help keys            for provider URLs and usage\n")

            else:
                print(f"  Unknown command: {cmd}. Type /help for options.")

            if cmd != "/retry":
                continue

        # --- @file expansion: replace @path references with file contents ---
        query_text = _expand_at_files(user_input)
        if query_text != user_input:
            at_files = re.findall(r'@(\S+)', user_input)
            print(f"  📎 Attached: {', '.join(at_files)}")

        # --- Regular query — use Rich Live for spinner + streaming response ---
        response_parts: list = []
        _cancelled = False
        try:
            from rich.console import Console as _RC
            from rich.live import Live as _RL
            from rich.markdown import Markdown as _RM
            from rich.text import Text as _RT
            _console = _RC()

            # ── Spinner phase (transient=True removes it cleanly when stopped) ──
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop  = threading.Event()
            _spin_idx   = [0]

            def _spin_update(live: '_RL') -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    live.update(_RT.from_markup(f"[bold yellow]Brain:[/bold yellow] {f} thinking…"))
                    _spin_idx[0] += 1

            if stream:
                token_gen = brain.query_stream(query_text, chat_history=chat_history)
                first_tokens: list = []

                # Show spinner until the first real token arrives
                if not quiet:
                    print()
                    with _RL(_RT.from_markup("[bold yellow]Brain:[/bold yellow] ⠋ thinking…"), console=_console,
                             transient=True, refresh_per_second=10) as _spin_live:
                        _st = threading.Thread(target=_spin_update, args=(_spin_live,), daemon=True)
                        _st.start()
                        for chunk in token_gen:
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            first_tokens.append(chunk)
                            break   # first token received — exit spinner Live context
                        _spin_stop.set()
                        _st.join(timeout=0.3)
                    # spinner is now gone from terminal (transient)

                # Stream remaining tokens with a live Markdown view
                try:
                    print()   # blank line between You: and Brain:
                    _console.print("[bold yellow]Brain:[/bold yellow]")
                    with _RL(console=_console, refresh_per_second=12) as _resp_live:
                        for chunk in first_tokens:          # replay buffered token
                            response_parts.append(chunk)
                            _resp_live.update(_RM("".join(response_parts)))
                        for chunk in token_gen:             # continue generator
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            response_parts.append(chunk)
                            _resp_live.update(_RM("".join(response_parts)))
                    print()   # blank line after Brain response, before next You:
                except KeyboardInterrupt:
                    _cancelled = True
                    print("  ⚠️  Cancelled.\n")
            else:
                # Non-streaming: spinner while brain.query() blocks
                _spin_stop2 = threading.Event()
                _result: list = []
                _err: list = []

                def _run_query() -> None:
                    try:
                        _result.append(brain.query(query_text, chat_history=chat_history))
                    except Exception as exc:
                        _err.append(exc)
                    finally:
                        _spin_stop2.set()

                _qt = threading.Thread(target=_run_query, daemon=True)
                _qt.start()

                if not quiet:
                    print()
                    with _RL(_RT.from_markup("[bold yellow]Brain:[/bold yellow] ⠋ thinking…"), console=_console,
                             transient=True, refresh_per_second=10) as _spin_live2:
                        _st2 = threading.Thread(target=_spin_update, args=(_spin_live2,), daemon=True)
                        _st2.start()
                        _spin_stop2.wait()
                        _spin_stop.set()
                        _st2.join(timeout=0.3)
                else:
                    _qt.join()

                if _err:
                    raise _err[0]
                response = _result[0] if _result else ""
                print()   # blank line between You: and Brain:
                _console.print("[bold yellow]Brain:[/bold yellow]")
                _console.print(_RM(response))
                print()   # blank line after Brain response, before next You:
                response_parts = [response]

        except ImportError:
            # rich not available — plain fallback
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop = threading.Event()
            _spin_idx  = [0]

            def _spin_plain() -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    sys.stdout.write(f"\r  Brain: {f} thinking…")
                    sys.stdout.flush()
                    _spin_idx[0] += 1

            if not quiet:
                print()
                _spt = threading.Thread(target=_spin_plain, daemon=True)
                _spt.start()
            response = brain.query(query_text, chat_history=chat_history)
            if not quiet:
                _spin_stop.set()
            print(f"\n\033[1;33mBrain:\033[0m {response}\n")
            response_parts = [response]

        response = "".join(response_parts)
        if not _cancelled:
            # Append both turns so future queries have full context
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            _last_query = user_input
            _save_session(session)   # persist after every turn


if __name__ == "__main__":
    main()
