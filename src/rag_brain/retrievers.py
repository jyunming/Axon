import os
import pickle
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
        self.index_file = os.path.join(storage_path, "bm25_index.pkl")
        self.bm25 = None
        self.corpus = []  # List of dicts: {'id': id, 'text': text, 'metadata': meta}
        
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
            
        self.load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        return text.lower().split()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the BM25 index."""
        self.corpus.extend(documents)
        tokenized_corpus = [self._tokenize(doc['text']) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

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

    def save(self):
        """Save the index to disk."""
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.corpus, self.bm25), f)
        logger.info(f"💾 BM25 index saved to {self.index_file}")

    def load(self):
        """Load the index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    self.corpus, self.bm25 = pickle.load(f)
                logger.info(f"📂 Loaded BM25 index with {len(self.corpus)} documents")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.corpus = []
                self.bm25 = None


def reciprocal_rank_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """
    Merge results from multiple retrievers using Reciprocal Rank Fusion.
    """
    fused_scores = {}
    
    # Process vector results
    for rank, doc in enumerate(vector_results):
        doc_id = doc['id']
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)
        
    # Process BM25 results
    for rank, doc in enumerate(bm25_results):
        doc_id = doc['id']
        if doc_id not in fused_scores:
            # We need to keep the full doc data if it wasn't in vector results
            fused_scores[doc_id] = 1 / (rank + k)
            # Store the doc structure for later
            doc['fused_only'] = True 
        else:
            fused_scores[doc_id] += 1 / (rank + k)

    # Combine all docs
    all_docs = {doc['id']: doc for doc in vector_results}
    for doc in bm25_results:
        if doc['id'] not in all_docs:
            all_docs[doc['id']] = doc

    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    final_results = []
    for doc_id in sorted_ids:
        doc = all_docs[doc_id]
        doc['score'] = fused_scores[doc_id]
        final_results.append(doc)
        
    return final_results
