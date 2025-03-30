from pydantic import BaseModel
from typing import Optional, List
import numpy as np

class TextInput(BaseModel):
    text: str
    metadata: Optional[dict] = None

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    similarity_score: float
    metadata: Optional[dict] = None

class VectorEntry:
    def __init__(self, text: str, embedding: np.ndarray, metadata: Optional[dict] = None):
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
