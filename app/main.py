import os
import base64
import hashlib
import tenseal as ts
import numpy as np
from PIL import Image
from typing import List

import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from cryptography.fernet import Fernet
import pickle

def generate_fernet_key(password: str) -> bytes:
    hashed = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hashed[:32])  # Ensure 32 bytes for Fernet

def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    return context

# יצירת ההקשר ההומומורפי
ckks_context = create_ckks_context()

def encrypt_vector_homomorphic(vec: np.ndarray, context: ts.Context) -> ts.CKKSVector:
    return ts.ckks_vector(context, vec.flatten())

def decrypt_vector_homomorphic(enc_vec: ts.CKKSVector) -> np.ndarray:
    return np.array(enc_vec.decrypt()).reshape(1, -1)

def encrypt_file(file_path: str, fernet_key: bytes):
    fernet = Fernet(fernet_key)
    with open(file_path, 'rb') as f:
        data = f.read()
    encrypted = fernet.encrypt(data)
    with open(file_path + '.enc', 'wb') as f:
        f.write(encrypted)
    os.remove(file_path)

def decrypt_file(enc_file_path: str, fernet_key: bytes) -> bytes:
    fernet = Fernet(fernet_key)
    with open(enc_file_path, 'rb') as f:
        encrypted = f.read()
    return fernet.decrypt(encrypted)

def save_index(index_data, path: str):
    with open(path, 'wb') as f:
        pickle.dump(index_data, f)

def load_index(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

class CLIPSecureEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.context = create_ckks_context()

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,  # ✅ חשוב: לחתוך טקסטים ארוכים
            max_length=77  # ✅ מגבלה של CLIP
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs.cpu().numpy().flatten()

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()

def encrypt_vector_ckks(vec: np.ndarray, context: ts.Context) -> ts.CKKSVector:
    return ts.ckks_vector(context, vec.flatten())

def decrypt_vector_ckks(enc_vec: ts.CKKSVector) -> np.ndarray:
    return np.array(enc_vec.decrypt()).reshape(-1)
