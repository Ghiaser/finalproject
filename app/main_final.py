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
import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="🔐 Secure Semantic Search", layout="centered")
st.title("🔐 Secure Semantic File Search")

# Helper functions for encryption
def generate_fernet_key(password: str) -> bytes:
    hashed = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hashed)

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

# Vector encryption and decryption
def encrypt_vector_homomorphic(vec: np.ndarray, context: ts.Context) -> ts.CKKSVector:
    return ts.ckks_vector(context, vec.flatten())

def decrypt_vector_homomorphic(enc_vec: ts.CKKSVector) -> np.ndarray:
    return np.array(enc_vec.decrypt()).reshape(1, -1).astype("float32")

# Core encryptor class
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
                    st.warning(f"Encryption error: {path} | Error: {e}")

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
        vec = self.encode_image(image_path)

        if vec is None:
            raise ValueError("Failed to encode the query image.")

        try:
            enc_vec = encrypt_vector_homomorphic(vec, ckks_context)
            decrypted_vec = decrypt_vector_homomorphic(enc_vec)
        except Exception as e:
            raise ValueError(f"Encryption/decryption failed: {e}")

        distances, indices = self.index.search(decrypted_vec, k)
        refs = self._decrypt_refs()

        results = []
        for i in indices[0]:
            if 0 <= i < len(refs):
                results.append(refs[i])

        return results

    def save_index(self, path="encrypted_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "data_refs": self.data_refs}, f)

    def load_index(self, path="encrypted_index.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.index = obj["index"]
            self.data_refs = obj["data_refs"]

# Main Streamlit logic
password = st.text_input("Enter your secret password", type="password")
folder = st.text_input("Enter full path to folder (with .jpg/.txt files)", value="C:\\shaked\\DATA")
index_path = "my_index.pkl"
files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.jpg', '.png', '.txt'))]

if "encryptor" not in st.session_state:
    st.session_state.encryptor = CLIPSecureEncryptor()

encryptor = st.session_state.encryptor

if password:
    if st.button("🔨 Build & Save Index"):
        with st.spinner("Building index..."):
            encryptor.build_index_from_files(files, password)
            encryptor.save_index(index_path)
            st.success("Index built and saved!")

    if st.button("📦 Load Existing Index"):
        try:
            encryptor.load_index(index_path)
            st.success("Index loaded.")
        except Exception as e:
            st.error(f"Error loading index: {e}")

    search_query = st.text_input("💬 Text Search Query")
    image_file = st.file_uploader("🖼️ Or upload image to search", type=["jpg", "jpeg", "png"])

    if st.button("🔍 Search"):
        if search_query:
            try:
                results = encryptor.query_text(search_query, password, k=10)
                st.subheader("Top results (text/image):")
                for r in results:
                    st.write(os.path.basename(r))
                    if r.lower().endswith('.txt'):
                        try:
                            with open(r, 'r', encoding='utf-8', errors='ignore') as f:
                                st.code(f.read(300) + "...")
                        except:
                            st.warning("Cannot preview text file.")
                    else:
                        st.image(r, width=250)
            except Exception as e:
                st.error(f"Text query failed: {e}")

        if image_file:
            temp_path = "temp_uploaded.jpg"
            with open(temp_path, "wb") as f:
                f.write(image_file.read())
            try:
                results = encryptor.query_image(temp_path, password, k=10)
                st.subheader("Top results (image query):")
                for r in results:
                    st.write(os.path.basename(r))
                    if r.lower().endswith('.txt'):
                        try:
                            with open(r, 'r', encoding='utf-8', errors='ignore') as f:
                                st.code(f.read(300) + "...")
                        except:
                            st.warning("Cannot preview text file.")
                    else:
                        st.image(r, width=250)
            except Exception as e:
                st.error(f"Image query failed: {e}")