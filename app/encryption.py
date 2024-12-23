from cryptography.fernet import Fernet
import numpy as np

key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_vector(vector):
    vector_bytes = vector.numpy().tobytes()
    return cipher.encrypt(vector_bytes)

def decrypt_vector(encrypted_vector):
    decrypted_bytes = cipher.decrypt(encrypted_vector)
    return np.frombuffer(decrypted_bytes)
