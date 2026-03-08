"""
Core engine for Local RAG Brain - Open Source RAG Interface.
"""

import os
import time
import yaml
import logging
import asyncio
from typing import Literal, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    discussion_fallback: bool = False

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
                prompt_fallback = f"The user asked: '{query}'. I found no relevant documents in the local knowledge base. Please provide a helpful response based on your general knowledge, while noting the lack of specific local context."
                return self.llm.complete(prompt_fallback, self._build_system_prompt(False), chat_history=chat_history)
            
            return "I don't have any relevant information to answer that question."
            
        if self.config.rerank: results = self.reranker.rerank(query, results)

        results = results[:self.config.top_k]
        final_count = len(results)
        top_score = results[0].get('score', 0) if results else 0
        context, has_web = self._build_context(results)
        system_prompt = self._build_system_prompt(has_web)
        prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}\n\nProvide a comprehensive but concise answer."
        response = self.llm.complete(prompt, system_prompt, chat_history=chat_history)
        self._log_query_metrics(query, retrieval['vector_count'], retrieval['bm25_count'], 
                                retrieval['filtered_count'], final_count, top_score, (time.time() - t0) * 1000, retrieval['transforms'])
        return response

    def query_stream(self, query: str, filters: Dict = None, chat_history: List[Dict[str, str]] = None):
        retrieval = self._execute_retrieval(query, filters)
        results = retrieval['results']
        
        if not results:
            if self.config.discussion_fallback:
                prompt_fallback = f"The user asked: '{query}'. I found no relevant documents in the local knowledge base. Please provide a helpful response based on your general knowledge, while noting the lack of specific local context."
                yield from self.llm.stream(prompt_fallback, self._build_system_prompt(False), chat_history=chat_history)
                return
            yield "I don't have any relevant information to answer that question."
            return
            
        if self.config.rerank: results = self.reranker.rerank(query, results)
        
        results = results[:self.config.top_k]
        context, has_web = self._build_context(results)
        system_prompt = self._build_system_prompt(has_web)
        
        # Yield a marker object so UI can optionally reconstruct sources
        yield {"type": "sources", "sources": results}

        prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}\n\nProvide a comprehensive but concise answer."
        
        yield from self.llm.stream(prompt, system_prompt, chat_history=chat_history)

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
    args = parser.parse_args()

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

    brain = OpenStudioBrain(config)
    
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
    _interactive_repl(brain, stream=True)


_SLASH_COMMANDS = ["/help", "/list", "/ingest ", "/model ", "/pull ", "/search", "/clear", "/quit", "/exit"]


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

            # /ingest <path> — complete filesystem paths
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


def _interactive_repl(brain: 'OpenStudioBrain', stream: bool = True) -> None:
    """Interactive chat REPL — initializes brain once, maintains history across turns."""
    # Silence INFO logs — they clutter the interactive UI
    import logging as _logging
    for _log in ("RAGBrain", "StudioBrainOpen.Retrievers", "httpx",
                 "sentence_transformers", "chromadb", "httpcore"):
        _logging.getLogger(_log).setLevel(_logging.WARNING)

    try:
        import readline
        readline.set_history_length(200)
        readline.set_completer(_make_completer(brain))
        readline.set_completer_delims("")   # treat full line as one token
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass

    model_info = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    search_status = "🔍 web search ON" if brain.config.truth_grounding else "🔍 web search OFF"
    print(f"\n🧠 Local RAG Brain  [{model_info}]  [{search_status}]")
    print("   Type your question, or use a slash command.")
    print("   /help  /list  /ingest <path>  /model <provider>/<model>  /search  /pull <name>  /clear  /quit\n")

    chat_history: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Bye!")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("👋 Bye!")
                break

            elif cmd == "/help":
                print(
                    "\n  /list                        — list ingested documents\n"
                    "  /ingest <path>               — ingest a file or directory\n"
                    "  /model <model>               — switch model (keep provider)\n"
                    "  /model <provider>/<model>    — switch provider and model\n"
                    "    providers: ollama, gemini, openai, ollama_cloud\n"
                    "    e.g. /model gemini/gemini-1.5-flash\n"
                    "         /model openai/gpt-4o\n"
                    "         /model ollama/gemma:2b\n"
                    "  /search                      — toggle web search (Brave API) on/off\n"
                    "  /pull <name>                 — pull an Ollama model\n"
                    "  /clear                       — clear chat history\n"
                    "  /quit                        — exit\n"
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
                    print("  Usage: /ingest <path>")
                elif os.path.isdir(arg):
                    print(f"  Ingesting {arg} …")
                    asyncio.run(brain.load_directory(arg))
                    print("  ✅ Done.")
                elif os.path.isfile(arg):
                    from rag_brain.loaders import DirectoryLoader
                    ext = os.path.splitext(arg)[1].lower()
                    mgr = DirectoryLoader()
                    if ext in mgr.loaders:
                        brain.ingest(mgr.loaders[ext].load(arg))
                        print("  ✅ Done.")
                    else:
                        print(f"  ❌ Unsupported file type: {ext}")
                else:
                    print(f"  ❌ Path not found: {arg}")

            elif cmd == "/model":
                _PROVIDERS = ('ollama', 'gemini', 'openai', 'ollama_cloud')
                if not arg:
                    print(f"  Current: {brain.config.llm_provider}/{brain.config.llm_model}")
                    print(f"  Usage:   /model <model>              (keep current provider)")
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
                        print(f"  ✅ Switched to {provider}/{model}")
                        if provider != "ollama":
                            print(f"  ℹ️  Make sure the required API key env var is set.")
                else:
                    brain.config.llm_model = arg
                    brain.llm = OpenLLM(brain.config)
                    print(f"  ✅ Switched to {brain.config.llm_provider}/{arg}")

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

            else:
                print(f"  Unknown command: {cmd}. Type /help for options.")

            continue

        # --- Regular query ---
        print("\nBrain: ⏳ thinking…", end="", flush=True)
        try:
            first_chunk = True
            if stream:
                for chunk in brain.query_stream(user_input, chat_history=chat_history):
                    if isinstance(chunk, dict):
                        continue
                    if first_chunk:
                        # Clear "thinking…" and start response on same line
                        print("\rBrain: " + " " * 15 + "\rBrain: ", end="", flush=True)
                        first_chunk = False
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = brain.query(user_input, chat_history=chat_history)
                print(f"\rBrain: {response}\n")

            chat_history.append({"role": "user", "content": user_input})
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")

if __name__ == "__main__":
    main()
