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

# Helper function to generate a Fernet key from password
def generate_fernet_key(password: str) -> bytes:
    hashed = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hashed)

# Create CKKS encryption context
def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    return context

ckks_context = create_ckks_context()

# Encrypt and decrypt vectors using CKKS
def encrypt_vector_homomorphic(vec: np.ndarray, context: ts.Context) -> ts.CKKSVector:
    return ts.ckks_vector(context, vec.flatten())

def decrypt_vector_homomorphic(enc_vec: ts.CKKSVector) -> np.ndarray:
    return np.array(enc_vec.decrypt()).reshape(1, -1).astype("float32")

# Main encryption/search handler
class CLIPSecureEncryptor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.index = None
        self.data_refs = []
        self.fernet = None

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
                    enc_vec = encrypt_vector_homomorphic(vec, ckks_context)
                    decrypted_vec = decrypt_vector_homomorphic(enc_vec)
                    encrypted_vectors.append(decrypted_vec)
                    self.data_refs.append(self.fernet.encrypt(os.path.abspath(path).encode()).decode())
                except Exception as e:
                    print(f"[Encryption Error] {path} | {e}")

        if encrypted_vectors:
            embedding_matrix = np.vstack(encrypted_vectors).astype("float32")
            self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
            self.index.add(embedding_matrix)

    def _decrypt_refs(self):
        return [self.fernet.decrypt(p.encode()).decode() for p in self.data_refs]

    def query_text(self, query: str, password: str, k=10):
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
        enc_vec = encrypt_vector_homomorphic(emb, ckks_context)
        decrypted_vec = decrypt_vector_homomorphic(enc_vec)
        distances, indices = self.index.search(decrypted_vec, k)
        refs = self._decrypt_refs()
        return [refs[i] for i in indices[0]]

    def query_image(self, image_path: str, password: str, k=3):
        if self.index is None:
            raise ValueError("Index is not loaded or built yet.")

        self.fernet = Fernet(generate_fernet_key(password))

        # Step 1: Encode the image to a vector
        vec = self.encode_image(image_path)
        if vec is None:
            raise ValueError("Failed to encode the query image.")

        # Step 2: Encrypt and decrypt the vector for matching
        try:
            enc_vec = encrypt_vector_homomorphic(vec, ckks_context)
            decrypted_vec = decrypt_vector_homomorphic(enc_vec)
        except Exception as e:
            raise ValueError(f"Encryption/decryption failed: {e}")

        # Step 3: Search in the index
        try:
            distances, indices = self.index.search(decrypted_vec, k)
        except Exception as e:
            raise ValueError(f"FAISS search failed: {e}")

        # Step 4: Debug output
        print("🔍 Vector shape:", decrypted_vec.shape)
        print("📊 Distances:", distances)
        print("📍 Indices:", indices)

        # Step 5: Retrieve file paths from the index
        refs = self._decrypt_refs()
        results = []
        for i in indices[0]:
            if 0 <= i < len(refs):
                results.append(refs[i])

        if not results:
            print("⚠️ No matching images found.")
        else:
            print(f"✅ Found {len(results)} results.")

        return results

    def save_index(self, path="encrypted_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "data_refs": self.data_refs}, f)

    def load_index(self, path="encrypted_index.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.index = obj["index"]
            self.data_refs = obj["data_refs"]
