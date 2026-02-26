"""
Core engine for Local RAG Brain - Open Source RAG Interface.
"""

import os
import yaml
import logging
import asyncio
from typing import Literal, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

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
    embedding_provider: Literal["sentence_transformers", "ollama", "fastembed"] = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    
    # LLM
    llm_provider: Literal["ollama", "vllm", "llama_cpp"] = "ollama"
    llm_model: str = "llama3.1"
    llm_temperature: float = 0.7
    
    # Vector Store
    vector_store: Literal["chroma", "qdrant", "lancedb"] = "chroma"
    vector_store_path: str = "./chroma_data"
    
    # BM25 Settings
    bm25_path: str = "./bm25_index"
    
    # RAG Settings
    top_k: int = 10
    similarity_threshold: float = 0.5
    hybrid_search: bool = True
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Re-ranking
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
            if 'model' in data['rerank']:
                config_dict['reranker_model'] = data['rerank']['model']
        
        # Map some specific names if they don't match exactly
        if 'ollama_base_url' not in config_dict and 'llm_base_url' in config_dict:
            config_dict['ollama_base_url'] = config_dict['llm_base_url']
            
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
        if self.config.rerank:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"📊 Loading Reranker: {self.config.reranker_model}")
                self.model = CrossEncoder(self.config.reranker_model)
            except ImportError:
                logger.error("sentence-transformers not installed. Reranking disabled.")
                self.config.rerank = False
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on a query.
        """
        if not self.config.rerank or not self.model or not documents:
            return documents
        
        logger.info(f"🔄 Reranking {len(documents)} documents...")
        
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
            self.dimension = 384
            
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "sentence_transformers":
            embeddings = self.model.encode(texts, convert_to_list=True)
            return embeddings
            
        elif self.provider == "ollama":
            import ollama
            embeddings = []
            for text in texts:
                response = ollama.embeddings(model=self.config.embedding_model, prompt=text)
                embeddings.append(response['embedding'])
            return embeddings
            
        elif self.provider == "fastembed":
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]


class OpenLLM:
    """
    Open-source LLM provider.
    """
    
    def __init__(self, config: OpenStudioConfig):
        self.config = config
        self.provider = config.llm_provider
    
    def complete(self, prompt: str, system_prompt: str = None) -> str:
        if self.provider == "ollama":
            import ollama
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.config.llm_model,
                messages=messages,
                options={"temperature": self.config.llm_temperature}
            )
            return response['message']['content']
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    def stream(self, prompt: str, system_prompt: str = None):
        if self.provider == "ollama":
            import ollama
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            stream = ollama.chat(
                model=self.config.llm_model,
                messages=messages,
                stream=True,
                options={"temperature": self.config.llm_temperature}
            )
            for chunk in stream:
                yield chunk['message']['content']


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
    
    SYSTEM_PROMPT = """You are a helpful AI assistant.
You help users by retrieving and synthesizing information from their local knowledge base.
Answer based on the provided context. If the answer is not in the context, say you don't know.
Be concise, professional, and helpful."""
    
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
    
    def ingest(self, documents: List[Dict[str, Any]]):
        if not documents: return
        from tqdm import tqdm
        logger.info(f"📥 Ingesting {len(documents)} documents...")
        if self.splitter: documents = self.splitter.transform_documents(documents)
        if self.bm25: self.bm25.add_documents(documents)
            
        ids, texts, metadatas = [d['id'] for d in documents], [d['text'] for d in documents], [d.get('metadata', {}) for d in documents]
        
        logger.info("   Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            all_embeddings.extend(self.embedding.embed(texts[i:i + batch_size]))
        
        self.vector_store.add(ids, texts, all_embeddings, metadatas)
        logger.info(f"✅ Ingested {len(documents)} chunks")

    def query(self, query: str, filters: Dict = None) -> str:
        query_embedding = self.embedding.embed_query(query)
        fetch_k = self.config.top_k * 3 if (self.config.rerank or self.config.hybrid_search) else self.config.top_k
        vector_results = self.vector_store.search(query_embedding, top_k=fetch_k, filter_dict=filters)
        
        if self.config.hybrid_search and self.bm25:
            bm25_results = self.bm25.search(query, top_k=fetch_k)
            from rag_brain.retrievers import reciprocal_rank_fusion
            results = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            results = vector_results
        
        results = [r for r in results if r.get('score', 1.0) >= self.config.similarity_threshold or 'fused_only' in r]
        if not results: return "I don't have any relevant information to answer that question."
        if self.config.rerank: results = self.reranker.rerank(query, results)
        
        results = results[:self.config.top_k]
        context = "\n\n".join([f"[Document {i+1} (ID: {r['id']})]\n{r['text']}" for i, r in enumerate(results)])
        prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}\n\nProvide a comprehensive but concise answer."
        return self.llm.complete(prompt, self.SYSTEM_PROMPT)

    def query_stream(self, query: str, filters: Dict = None):
        query_embedding = self.embedding.embed_query(query)
        fetch_k = self.config.top_k * 3 if (self.config.rerank or self.config.hybrid_search) else self.config.top_k
        vector_results = self.vector_store.search(query_embedding, top_k=fetch_k, filter_dict=filters)
        
        if self.config.hybrid_search and self.bm25:
            bm25_results = self.bm25.search(query, top_k=fetch_k)
            from rag_brain.retrievers import reciprocal_rank_fusion
            results = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            results = vector_results
            
        results = [r for r in results if r.get('score', 1.0) >= self.config.similarity_threshold or 'fused_only' in r]
        if not results:
            yield "I don't have any relevant information to answer that question."
            return
        if self.config.rerank: results = self.reranker.rerank(query, results)
        
        results = results[:self.config.top_k]
        context = "\n\n".join([f"[Document {i+1} (ID: {r['id']})]\n{r['text']}" for i, r in enumerate(results)])
        prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}\n\nProvide a comprehensive but concise answer."
        yield from self.llm.stream(prompt, self.SYSTEM_PROMPT)

    async def load_directory(self, directory: str):
        from rag_brain.loaders import DirectoryLoader
        loader = DirectoryLoader()
        logger.info(f"📁 Scanning: {directory}")
        documents = await loader.aload(directory)
        if documents: self.ingest(documents)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local RAG Brain CLI")
    parser.add_argument('query', nargs='?', help='Question to ask')
    parser.add_argument('--ingest', help='Path to file or directory to ingest')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--stream', action='store_true', help='Stream the response')
    args = parser.parse_args()
    
    config = OpenStudioConfig.load(args.config)
    brain = OpenStudioBrain(config)
    
    if args.ingest:
        if os.path.isdir(args.ingest): asyncio.run(brain.load_directory(args.ingest))
        else:
            from rag_brain.loaders import DirectoryLoader
            ext = os.path.splitext(args.ingest)[1].lower()
            loader_mgr = DirectoryLoader()
            if ext in loader_mgr.loaders: brain.ingest(loader_mgr.loaders[ext].load(args.ingest))
    
    if args.query:
        if args.stream:
            for chunk in brain.query_stream(args.query): print(chunk, end="", flush=True)
            print()
        else: print(f"\n📝 Response:\n{brain.query(args.query)}")

if __name__ == "__main__":
    main()
