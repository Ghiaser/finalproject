import streamlit as st
import os
import faiss
import numpy as np
from main import (
    CLIPSecureEmbedder,
    create_ckks_context,
    decrypt_vector_ckks,
    encrypt_vector_ckks,
    encrypt_file,
    decrypt_file,
    save_index,
    load_index
)
from PIL import Image
import io

# ========== CONFIG ==========
VEC_DIM = 768  # CLIP ViT-L/14
DATA_DIR = "user_data/testuser/data"
INDEX_PATH = "user_data/testuser/indexes/index.pkl"

# ========== UTILS ==========
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# ========== INIT ==========
st.set_page_config(page_title="🔐 Secure Semantic Search", layout="wide")
embedder = CLIPSecureEmbedder()
context = create_ckks_context()

if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatIP(VEC_DIM)
    st.session_state.file_map = []

# ========== SIDEBAR ==========
st.sidebar.title("🔐 Secure Encrypted Search")
option = st.sidebar.radio("Choose an action:", ["🔍 Search", "📁 Build Index", "🔑 Decrypt File"])

# ========== Build Index ==========
if option == "📁 Build Index":
    st.header("📁 Upload files to build the encrypted index")
    files = st.file_uploader("Upload images or text files", type=["jpg", "png", "jpeg", "txt"], accept_multiple_files=True)
    password = st.text_input("Encryption password", type="password")

    if st.button("🔐 Encrypt and Add to Index"):
        for file in files:
            file_path = os.path.join(DATA_DIR, file.name)
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.read())

            # Embed vector
            if file.type.startswith("image"):
                vec = embedder.embed_image(file_path)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                vec = embedder.embed_text(text)

            encrypt_file(file_path, password)
            enc_vec = encrypt_vector_ckks(vec, context)
            dec_vec = decrypt_vector_ckks(enc_vec)
            norm_vec = normalize(dec_vec)

            st.session_state.index.add(np.array([norm_vec]).astype("float32"))
            st.session_state.file_map.append(file.name)

        save_index((st.session_state.index, st.session_state.file_map), INDEX_PATH)
        st.success("✅ Files encrypted and added to index")

# ========== Search ==========
elif option == "🔍 Search":
    st.header("🔍 Semantic Search")
    query = st.text_input("Enter your search text")
    password = st.text_input("Decryption password", type="password")

    if st.button("Search") and query:
        if not os.path.exists(INDEX_PATH):
            st.error("No index found. Please build it first.")
        else:
            index, file_map = load_index(INDEX_PATH)
            vec = embedder.embed_text(query)
            enc_vec = encrypt_vector_ckks(vec, context)
            dec_vec = decrypt_vector_ckks(enc_vec)
            norm_vec = normalize(dec_vec)
            D, I = index.search(np.array([norm_vec]).astype("float32"), k=5)

            st.subheader("🔎 Top Results:")
            for idx, (i, dist) in enumerate(zip(I[0], D[0])):
                filename = file_map[i]
                encrypted_path = os.path.join(DATA_DIR, filename + ".enc")
                try:
                    decrypted = decrypt_file(encrypted_path, password)
                    st.markdown(f"**Similarity Score:** {dist:.4f}")
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image = Image.open(io.BytesIO(decrypted))
                        st.image(image, caption=filename, width=200)
                    elif filename.lower().endswith(".txt"):
                        text = decrypted.decode("utf-8", errors="ignore")
                        st.text_area("📄 Text Preview", text[:300], height=100, key=f"preview_{idx}")
                    else:
                        st.warning(f"Unsupported file: {filename}")
                except Exception as e:
                    st.error(f"❌ Failed to decrypt {filename}: {str(e)}")

# ========== Decrypt ==========
elif option == "🔑 Decrypt File":
    st.header("🔑 Decrypt a file")
    enc_file = st.file_uploader("Upload .enc file", type=["enc"])
    password = st.text_input("Password", type="password")

    if st.button("🔓 Decrypt") and enc_file:
        encrypted_path = os.path.join(DATA_DIR, enc_file.name)
        with open(encrypted_path, "wb") as f:
            f.write(enc_file.read())

        try:
            decrypted = decrypt_file(encrypted_path, password)
            st.success("✅ Decryption successful")
            if enc_file.name.endswith((".jpg.enc", ".png.enc", ".jpeg.enc")):
                image = Image.open(io.BytesIO(decrypted))
                st.image(image, caption="Decrypted Image")
            else:
                st.text_area("📄 Decrypted Content", decrypted.decode("utf-8"))
        except Exception as e:
            st.error(f"❌ Decryption failed: {str(e)}")