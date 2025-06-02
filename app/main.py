import os
import base64
import hashlib
import tenseal as ts
import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from cryptography.fernet import Fernet
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


def generate_fernet_key(password: str) -> bytes:
    """
    Returns a standard Fernet key derived from the SHA256 of the password (returns 32 valid bytes).
    """
    hashed = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hashed[:32])


def create_ckks_context():
    """
    Creates a basic context for CKKS (TenSEAL) with reasonable parameters (8192, coeff_mod_bit_sizes…).
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    return context


def encrypt_vector_ckks(vec: np.ndarray, context: ts.Context) -> ts.CKKSVector:
    """
    Wraps a call to ts.ckks_vector to fully encrypt a CKKS vector.
    """
    return ts.ckks_vector(context, vec.flatten())


def decrypt_vector_ckks(enc_vec: ts.CKKSVector) -> np.ndarray:
    """
    Decrypts a CKKSVector and returns a flattened numpy.ndarray.
    """
    return np.array(enc_vec.decrypt()).reshape(-1)


def encrypt_file(file_path: str, fernet_key: bytes):
    """
    Takes a file (binary), encrypts it deterministically using Fernet, saves it to `file_path + '.enc'`,
    and deletes the original.
    """
    fernet = Fernet(fernet_key)
    with open(file_path, 'rb') as f:
        data = f.read()
    encrypted = fernet.encrypt(data)
    with open(file_path + '.enc', 'wb') as f:
        f.write(encrypted)
    try:
        os.remove(file_path)
    except OSError:
        pass


def decrypt_file(enc_file_path: str, fernet_key: bytes) -> bytes:
    """
    Decrypts a Fernet-encrypted file and returns its bytes.
    """
    fernet = Fernet(fernet_key)
    with open(enc_file_path, 'rb') as f:
        encrypted = f.read()
    return fernet.decrypt(encrypted)


def save_index(index_data, path: str):
    """
    Saves any pickle-able object to disk.
    """
    with open(path, 'wb') as f:
        pickle.dump(index_data, f)


def load_index(path: str):
    """
    Loads a pickle object from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


class CLIPSecureEmbedder:
    def __init__(self, device: str = None):
        # If no device is given, choose GPU if available:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load CLIP processor + model:
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        # (Optionally: store your CKKS context if you need it here)
        self.context = create_ckks_context()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Given a single string, return a normalized 512-dim numpy embedding.
        """
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        vec = outputs.cpu().numpy().flatten()  # (512,)
        vec = vec / (np.linalg.norm(vec) + 1e-10)  # normalize to unit-length
        return vec

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Given a path to an image, return a normalized 512-dim numpy embedding.
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        vec = outputs.cpu().numpy().flatten()  # (512,)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec
