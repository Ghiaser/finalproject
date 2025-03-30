import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import Union, List
import io

class CLIPEmbeddingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text input."""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Normalize and convert to numpy
        embedding = text_features.cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding[0]  # Return the first (and only) embedding
    
    def get_image_embedding(self, image: Union[Image.Image, bytes]) -> np.ndarray:
        """Generate embedding for image input."""
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize and convert to numpy
        embedding = image_features.cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding[0]  # Return the first (and only) embedding

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2))
