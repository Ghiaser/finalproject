import os
import base64
import hashlib
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
    return base64.urlsafe_b64encode(hashed)


def encrypt_vector(vec: np.ndarray, fernet: Fernet) -> np.ndarray:
    flat_bytes = vec.tobytes()
    encrypted = fernet.encrypt(flat_bytes)
    return np.frombuffer(fernet.decrypt(encrypted), dtype=np.float32).reshape(vec.shape)


class CLIPSecureEncryptor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.index = None
        self.data_refs = []
        self.fernet = None

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vec = self.model.get_image_features(**inputs).cpu().numpy().astype("float32")
            print(f"✅ Encoded image: {image_path} | Shape: {vec.shape}")
            return vec
        except Exception as e:
            print(f"❌ Failed to encode image: {image_path} | Error: {e}")
            return None

    def encode_text_file(self, text_path: str) -> np.ndarray:
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            inputs = self.processor(
                text=content,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            with torch.no_grad():
                vec = self.model.get_text_features(**inputs).cpu().numpy().astype("float32")
            print(f"✅ Encoded text: {text_path} | Shape: {vec.shape}")
            return vec
        except Exception as e:
            print(f"❌ Failed to encode text: {text_path} | Error: {e}")
            return None

    def build_index_from_files(self, file_paths: List[str], password: str):
        self.model.eval()
        self.fernet = Fernet(generate_fernet_key(password))
        encrypted_vectors = []
        self.data_refs = []

        for path in file_paths:
            vec = None
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                vec = self.encode_image(path)
            elif path.lower().endswith('.txt'):
                vec = self.encode_text_file(path)

            if vec is not None:
                try:
                    enc_vec = encrypt_vector(vec, self.fernet)
                    encrypted_vectors.append(enc_vec)
                    self.data_refs.append(self.fernet.encrypt(path.encode()).decode())
                except Exception as e:
                    print(f"⚠️ Encryption or indexing error: {path} | Error: {e}")
            else:
                print(f"⚠️ Skipped: {path}")

        if encrypted_vectors:
            embedding_matrix = np.vstack(encrypted_vectors).astype("float32")
            print(f"🔧 Final index shape: {embedding_matrix.shape}")
            self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
            self.index.add(embedding_matrix)
            print("✅ FAISS index built.")
        else:
            print("⚠️ No vectors to index.")

    def _decrypt_refs(self):
        try:
            return [self.fernet.decrypt(p.encode()).decode() for p in self.data_refs]
        except Exception:
            raise ValueError("❌ ERROR: Unable to decrypt file paths — wrong password?")

    def query_text(self, query: str, password: str, k=10):
        if self.index is None:
            raise ValueError("Index is not loaded or built yet.")
        self.fernet = Fernet(generate_fernet_key(password))
        vec = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**vec).cpu().numpy().astype("float32")
        enc_vec = encrypt_vector(emb, self.fernet)
        distances, indices = self.index.search(enc_vec, k)
        refs = self._decrypt_refs()

        scored = []
        for i, dist in zip(indices[0], distances[0]):
            ref = refs[i]
            weight = 0.8 if ref.lower().endswith(('.jpg', '.jpeg', '.png')) else 1.0
            scored.append((ref, dist * weight))

        sorted_refs = sorted(scored, key=lambda x: x[1])
        return [r[0] for r in sorted_refs[:3]]

    def query_image(self, image_path: str, password: str, k=3):
        if self.index is None:
            raise ValueError("Index is not loaded or built yet.")
        self.fernet = Fernet(generate_fernet_key(password))
        vec = self.encode_image(image_path)
        enc_vec = encrypt_vector(vec, self.fernet)
        distances, indices = self.index.search(enc_vec, k)
        refs = self._decrypt_refs()
        return [refs[i] for i in indices[0]]

    def save_index(self, path="encrypted_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "data_refs": self.data_refs
            }, f)

    def load_index(self, path="encrypted_index.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.index = obj["index"]
            self.data_refs = obj["data_refs"]