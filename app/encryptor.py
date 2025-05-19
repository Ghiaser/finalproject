import os
import numpy as np
import faiss
import pickle
from PIL import Image
from cryptography.fernet import Fernet
from app.main import (
    CLIPSecureEmbedder,
    encrypt_vector_ckks,
    decrypt_vector_ckks,
    generate_fernet_key
)

class CLIPSecureEncryptor:
    def __init__(self, vec_dim=768):
        self.embedder = CLIPSecureEmbedder()
        self.index = faiss.IndexFlatIP(vec_dim)
        self.file_map = []

    def normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    def encrypt_file(self, file_path, password):
        key = generate_fernet_key(password)
        fernet = Fernet(key)
        with open(file_path, 'rb') as f:
            data = f.read()
        encrypted = fernet.encrypt(data)
        with open(file_path + '.enc', 'wb') as f:
            f.write(encrypted)
        os.remove(file_path)

    def decrypt_file(self, enc_file_path, password):
        key = generate_fernet_key(password)
        fernet = Fernet(key)
        with open(enc_file_path, 'rb') as f:
            encrypted = f.read()
        return fernet.decrypt(encrypted)

    def build_index_from_files(self, files, password):
        vectors = []
        file_map = []
        for file_path in files:
            try:
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    vec = self.embedder.embed_image(file_path)
                elif file_path.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    vec = self.embedder.embed_text(text)
                else:
                    continue

                self.encrypt_file(file_path, password)
                enc_vec = encrypt_vector_ckks(vec, self.embedder.context)
                dec_vec = decrypt_vector_ckks(enc_vec)
                norm_vec = self.normalize(dec_vec)
                vectors.append(norm_vec.astype('float32'))
                file_map.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        if vectors:
            self.index.add(np.array(vectors))
            self.file_map = file_map

    def save_index(self, path, password):
        data = {
            'index': self.index,
            'file_map': self.file_map
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path, password):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.file_map = data['file_map']

    def query_text(self, query, password, k=5):
        vec = self.embedder.embed_text(query)
        norm_vec = self.normalize(vec)
        D, I = self.index.search(np.array([norm_vec]).astype('float32'), k)
        results = [(self.file_map[i], D[0][idx]) for idx, i in enumerate(I[0])]
        return results

    def query_image(self, image_path, password, k=5):
        vec = self.embedder.embed_image(image_path)
        norm_vec = self.normalize(vec)
        D, I = self.index.search(np.array([norm_vec]).astype('float32'), k)
        results = [(self.file_map[i], D[0][idx]) for idx, i in enumerate(I[0])]
        return results
