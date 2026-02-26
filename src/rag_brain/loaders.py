import os
import json
import pandas as pd
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image
import io
import base64

logger = logging.getLogger("StudioBrainOpen.Loaders")

class BaseLoader:
    """Base class for document loaders."""
    def load(self, path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    async def aload(self, path: str) -> List[Dict[str, Any]]:
        """Async version of load."""
        return await asyncio.to_thread(self.load, path)

class TextLoader(BaseLoader):
    """Loader for plain text files."""
    def load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{
            "id": os.path.basename(path),
            "text": content,
            "metadata": {"source": path, "type": "text"}
        }]

class TSVLoader(BaseLoader):
    """Loader for tab-delimited files."""
    def load(self, path: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(path, sep='	')
        documents = []
        for i, row in df.iterrows():
            # Assume first column or 'content' column is the text
            text_col = 'content' if 'content' in df.columns else df.columns[0]
            text = str(row[text_col])
            metadata = row.drop(text_col).to_dict()
            metadata.update({"source": path, "type": "tsv", "row": i})
            documents.append({
                "id": f"{os.path.basename(path)}_row_{i}",
                "text": text,
                "metadata": metadata
            })
        return documents

class JSONLoader(BaseLoader):
    """Loader for JSON files."""
    def load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                text = item.get('text', item.get('content', json.dumps(item)))
                metadata = {k: v for k, v in item.items() if k not in ['text', 'content']}
                metadata.update({"source": path, "type": "json", "index": i})
                documents.append({
                    "id": f"{os.path.basename(path)}_{i}",
                    "text": text,
                    "metadata": metadata
                })
            return documents
        else:
            text = data.get('text', data.get('content', json.dumps(data)))
            metadata = {k: v for k, v in data.items() if k not in ['text', 'content']}
            metadata.update({"source": path, "type": "json"})
            return [{
                "id": os.path.basename(path),
                "text": text,
                "metadata": metadata
            }]

class BMPLoader(BaseLoader):
    """
    Loader for BMP files using Ollama VLM for captioning.
    """
    def __init__(self, ollama_model: str = "llava"):
        self.ollama_model = ollama_model
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            self.ollama = None
            logger.error("ollama package not installed. BMP loading will fail.")

    def load(self, path: str) -> List[Dict[str, Any]]:
        if self.ollama is None:
            return []
            
        logger.info(f"🖼️ Processing image: {path} with {self.ollama_model}...")
        
        try:
            with open(path, 'rb') as f:
                image_data = f.read()
            
            # Call Ollama VLM
            response = self.ollama.generate(
                model=self.ollama_model,
                prompt="Describe this image in detail. Mention any text, objects, people, or patterns visible.",
                images=[image_data]
            )
            
            description = response['response']
            
            return [{
                "id": os.path.basename(path),
                "text": f"Image Description: {description}",
                "metadata": {
                    "source": path,
                    "type": "image",
                    "format": "bmp",
                    "model": self.ollama_model
                }
            }]
        except Exception as e:
            logger.error(f"Error processing BMP {path}: {e}")
            return []

class DirectoryLoader:
    """Loader that crawls a directory and uses appropriate loaders for each file."""
    def __init__(self, vlm_model: str = "llava"):
        self.loaders = {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".tsv": TSVLoader(),
            ".json": JSONLoader(),
            ".bmp": BMPLoader(ollama_model=vlm_model)
        }
    
    def load(self, directory: str) -> List[Dict[str, Any]]:
        all_documents = []
        path = Path(directory)
        
        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in self.loaders:
                loader = self.loaders[file_path.suffix.lower()]
                try:
                    docs = loader.load(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return all_documents

    async def aload(self, directory: str) -> List[Dict[str, Any]]:
        """Async version of load_directory."""
        path = Path(directory)
        tasks = []
        
        for file_path in path.rglob("*"):
            suffix = file_path.suffix.lower()
            if suffix in self.loaders:
                loader = self.loaders[suffix]
                tasks.append(loader.aload(str(file_path)))
        
        if not tasks:
            return []
            
        results = await asyncio.gather(*tasks)
        all_documents = []
        for docs in results:
            all_documents.extend(docs)
            
        return all_documents
