import os
import base64
import hashlib
import hmac
import pickle
import numpy as np
from PIL import Image
from typing import List, Tuple

import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from cryptography.fernet import Fernet


def generate_fernet_key(password: str) -> bytes:
    hashed = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hashed)


def encrypt_vector(vec: np.ndarray, fernet: Fernet) -> bytes:
    return fernet.encrypt(vec.tobytes())


def decrypt_vector(enc_bytes: bytes, shape, fernet: Fernet) -> np.ndarray:
    return np.frombuffer(fernet.decrypt(enc_bytes), dtype=np.float32).reshape(shape)


def sign_file(path: str, key: str) -> str:
    with open(path, 'rb') as f:
        data = f.read()
    return hmac.new(key.encode(), data, hashlib.sha256).hexdigest()


def verify_signature(path: str, key: str, signature: str) -> bool:
    return sign_file(path, key) == signature


class CLIPSecureEncryptor:
    def __init__(self, device='cpu'):  # Force CPU to avoid CUDA/OpenMP conflicts
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.index = None
        self.data_refs = []
        self.vector_shapes = []
        self.fernet = None

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vec = self.model.get_image_features(**inputs).cpu().numpy().astype("float32")
            return vec / np.linalg.norm(vec)
        except Exception as e:
            print(f"Failed to encode image: {image_path} | Error: {e}")
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
            return vec / np.linalg.norm(vec)
        except Exception as e:
            print(f"Failed to encode text: {text_path} | Error: {e}")
            return None

    def build_index_from_files(self, file_paths: List[str], password: str):
        self.model.eval()
        self.fernet = Fernet(generate_fernet_key(password))
        encrypted_vectors = []
        self.data_refs = []
        self.vector_shapes = []

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
                    self.vector_shapes.append(vec.shape)
                    self.data_refs.append(self.fernet.encrypt(path.encode()).decode())
                except Exception as e:
                    print(f"Encryption or indexing error: {path} | Error: {e}")

        if encrypted_vectors:
            decrypted_matrix = [decrypt_vector(enc_vec, shape, self.fernet)
                                for enc_vec, shape in zip(encrypted_vectors, self.vector_shapes)]
            matrix = np.vstack(decrypted_matrix).astype("float32")
            self.index = faiss.IndexFlatL2(matrix.shape[1])
            self.index.add(matrix)
        else:
            raise ValueError("No valid vectors to index")

    def _decrypt_refs(self):
        return [self.fernet.decrypt(p.encode()).decode() for p in self.data_refs]

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
        emb = emb / np.linalg.norm(emb)
        enc_vec = encrypt_vector(emb, self.fernet)
        query_vec = decrypt_vector(enc_vec, emb.shape, self.fernet)
        distances, indices = self.index.search(query_vec, k)
        refs = self._decrypt_refs()
        return [(refs[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def query_image(self, image_path: str, password: str, k=3):
        if self.index is None:
            raise ValueError("Index is not loaded or built yet.")
        self.fernet = Fernet(generate_fernet_key(password))
        vec = self.encode_image(image_path)
        enc_vec = encrypt_vector(vec, self.fernet)
        query_vec = decrypt_vector(enc_vec, vec.shape, self.fernet)
        distances, indices = self.index.search(query_vec, k)
        refs = self._decrypt_refs()
        return [(refs[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def save_index(self, path="encrypted_index.pkl", password: str = ""):
        with open(path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "data_refs": self.data_refs,
                "vector_shapes": self.vector_shapes
            }, f)
        if password:
            signature = sign_file(path, password)
            with open(path + ".sig", "w") as sig_file:
                sig_file.write(signature)

    def load_index(self, path="encrypted_index.pkl", password: str = ""):
        sig_path = path + ".sig"
        if not os.path.exists(sig_path):
            raise ValueError("Missing index signature file")
        with open(sig_path, "r") as sig_file:
            signature = sig_file.read().strip()
        if not verify_signature(path, password, signature):
            raise ValueError("Index signature mismatch. Possible tampering detected.")

        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.index = obj["index"]
            self.data_refs = obj["data_refs"]
            self.vector_shapes = obj["vector_shapes"]
