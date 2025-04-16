
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


def encrypt_vector(vec: np.ndarray, fernet: Fernet) -> bytes:
    flat_bytes = vec.tobytes()
    return fernet.encrypt(flat_bytes)

def decrypt_vector(encrypted_bytes: bytes, fernet: Fernet, shape=(1, 512)) -> np.ndarray:
    flat_bytes = fernet.decrypt(encrypted_bytes)
    return np.frombuffer(flat_bytes, dtype=np.float32).reshape(shape)


class CLIPSecureEncryptor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.index = None
        self.data_refs = []
        self.fernet = None  # for decrypting file paths later

    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vec = self.model.get_image_features(**inputs).cpu().numpy().astype("float32")
        return vec

    def encode_text_file(self, text_path: str) -> np.ndarray:
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
        return vec

    def build_index_from_files(self, folder_path: str, password: str):
        self.encrypted_vectors = []
        self.data_refs = []
        self.fernet = Fernet(generate_fernet_key(password))

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, "rb") as f:
                content = f.read()
            self.data_refs.append(self.fernet.encrypt(content).decode())

            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(BytesIO(content)).convert("RGB")
                processed = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vec = self.model.get_image_features(**processed).cpu().numpy().astype("float32")
            else:
                processed = self.processor(text=filename, return_tensors="pt", padding=True, truncation=True).to(
                    self.device)
                with torch.no_grad():
                    vec = self.model.get_text_features(**processed).cpu().numpy().astype("float32")

            encrypted_vec = encrypt_vector(vec, self.fernet)
            self.encrypted_vectors.append(encrypted_vec)


    def _decrypt_refs(self):
        try:
            return [self.fernet.decrypt(p.encode()).decode() for p in self.data_refs]
        except Exception:
            raise ValueError("❌ ERROR: Unable to decrypt file paths — wrong password?")

        return [self.fernet.decrypt(p.encode()).decode() for p in self.data_refs]

    def query_text(self, query: str, password: str, k=3):
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

        scores = []
        for i, enc_vec in enumerate(self.encrypted_vectors):
            try:
                decrypted_vec = decrypt_vector(enc_vec, self.fernet, shape=emb.shape)
                sim = np.dot(emb, decrypted_vec.T)[0][0]
                scores.append((self.data_refs[i], sim))
            except Exception as e:
                print(f"❌ Failed to decrypt or compare vector {i}: {e}")

        sorted_refs = sorted(scores, key=lambda x: -x[1])[:k]
        return [self.fernet.decrypt(ref.encode()).decode() for ref, _ in sorted_refs]

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
                "data_refs": self.data_refs,
                "encrypted_vectors": self.encrypted_vectors
            }, f)

    def load_index(self, path="encrypted_index.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.data_refs = obj["data_refs"]
            self.encrypted_vectors = obj["encrypted_vectors"]

