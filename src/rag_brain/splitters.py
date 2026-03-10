import re
from typing import List, Dict, Any

class RecursiveCharacterTextSplitter:
    """
    Split text into chunks based on character length and overlap,
    recursively trying different separators to maintain context.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []
            
        chunks = []
        current_text = text
        
        while len(current_text) > self.chunk_size:
            # Find the best separator within the chunk_size
            split_at = -1
            for sep in self.separators:
                # Find the last occurrence of separator within chunk_size
                if sep == "":
                    split_at = self.chunk_size
                    break
                
                last_idx = current_text[:self.chunk_size].rfind(sep)
                if last_idx != -1:
                    split_at = last_idx + len(sep)
                    break
            
            if split_at == -1:
                split_at = self.chunk_size
                
            chunks.append(current_text[:split_at].strip())
            current_text = current_text[split_at - self.chunk_overlap:]
            
        if current_text:
            chunks.append(current_text.strip())
            
        return chunks

    def transform_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split a list of documents into chunks."""
        all_chunks = []
        for doc in documents:
            text_chunks = self.split(doc['text'])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get('metadata', {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                all_chunks.append({
                    "id": f"{doc['id']}_chunk_{i}",
                    "text": chunk,
                    "metadata": metadata
                })
        return all_chunks
