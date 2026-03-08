import os
import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger("StudioBrainOpen.Retrievers")

class BM25Retriever:
    """
    Keyword-based retriever using BM25 algorithm.
    Complements vector search for specific term matching.
    """
    
    def __init__(self, storage_path: str = "./bm25_index"):
        self.storage_path = storage_path
        self.corpus_file = os.path.join(storage_path, "bm25_corpus.json")
        # Legacy alias — used only during pickle→JSON migration
        self.index_file = os.path.join(storage_path, "bm25_index.pkl")
        self.bm25 = None
        self.corpus = []  # List of dicts: {'id': id, 'text': text, 'metadata': meta}
        
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
            
        self.load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        return text.lower().split()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the BM25 index. No-op if documents is empty."""
        if not documents:
            return
        self.corpus.extend(documents)
        tokenized_corpus = [self._tokenize(doc['text']) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    # Explicit alias for callers that batch their writes; semantics are identical.
    batch_add_documents = add_documents

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the BM25 index."""
        if self.bm25 is None or not self.corpus:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:
                doc = self.corpus[i].copy()
                doc['score'] = float(scores[i])
                results.append(doc)
                
        return results

    def delete_documents(self, doc_ids: List[str]) -> None:
        """Remove documents by ID and rebuild index."""
        original_count = len(self.corpus)
        self.corpus = [doc for doc in self.corpus if doc['id'] not in doc_ids]
        if len(self.corpus) < original_count:
            if self.corpus:
                tokenized_corpus = [self._tokenize(doc['text']) for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
            self.save()

    def save(self):
        """Save the corpus to disk as JSON."""
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, ensure_ascii=False)
        logger.info(f"💾 BM25 corpus saved to {self.corpus_file}")

    def load(self):
        """Load corpus from JSON (or migrate from legacy pickle)."""
        # Migrate legacy pickle if it exists
        legacy_pkl = os.path.join(self.storage_path, "bm25_index.pkl")
        if os.path.exists(legacy_pkl) and not os.path.exists(self.corpus_file):
            try:
                import pickle
                with open(legacy_pkl, 'rb') as f:
                    corpus, _ = pickle.load(f)
                self.corpus = corpus
                self.save()  # save as JSON
                os.remove(legacy_pkl)
                logger.info(f"✅ Migrated BM25 index from pickle to JSON")
            except Exception as e:
                logger.error(f"Failed to migrate BM25 pickle: {e}")
                return
        elif os.path.exists(self.corpus_file):
            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
                if self.corpus:
                    tokenized_corpus = [self._tokenize(doc['text']) for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"📂 Loaded BM25 corpus with {len(self.corpus)} documents")
            except Exception as e:
                logger.error(f"Failed to load BM25 corpus: {e}")
                self.corpus = []
                self.bm25 = None


def reciprocal_rank_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """Merge results from multiple retrievers using Reciprocal Rank Fusion.

    The original cosine similarity score from vector search is preserved in the
    ``vector_score`` field so the UI can display a meaningful relevance value.
    The RRF-fused score (used only for ranking) is stored in ``score``.
    """
    fused_scores = {}

    # Preserve original cosine scores keyed by doc_id
    vector_scores = {doc['id']: doc.get('score', 0.0) for doc in vector_results}

    for rank, doc in enumerate(vector_results):
        doc_id = doc['id']
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)

    for rank, doc in enumerate(bm25_results):
        doc_id = doc['id']
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 1 / (rank + k)
            doc['fused_only'] = True
        else:
            fused_scores[doc_id] += 1 / (rank + k)

    all_docs = {doc['id']: doc for doc in vector_results}
    for doc in bm25_results:
        if doc['id'] not in all_docs:
            all_docs[doc['id']] = doc

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    final_results = []
    for doc_id in sorted_ids:
        doc = all_docs[doc_id]
        doc['score'] = fused_scores[doc_id]
        # Expose the original cosine similarity for display purposes
        if doc_id in vector_scores:
            doc['vector_score'] = vector_scores[doc_id]
        final_results.append(doc)

    return final_results
