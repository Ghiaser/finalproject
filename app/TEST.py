import os
import streamlit as st
import torch
import warnings
from main import CLIPSecureEncryptor

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variable to fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Force PyTorch to use CPU
if torch.cuda.is_available():
    torch.cuda.is_available = lambda: False

st.set_page_config(page_title="🔐 Secure Clip Test", layout="centered")
st.title("🧪 Test Secure Semantic Search")

password = st.text_input("Enter your password", type="password")
index_path = "encrypted_index.pkl"
data_folder = "./DATA"

if password:
    if "encryptor" not in st.session_state:
        st.session_state.encryptor = CLIPSecureEncryptor()

    encryptor = st.session_state.encryptor
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.lower().endswith((".txt", ".jpg", ".jpeg", ".png"))]

    if st.button("🔨 Build Index"):
        try:
            encryptor.build_index_from_files(files, password)
            encryptor.save_index(index_path, password)
            st.success("✅ Index built and saved with signature.")
        except Exception as e:
            st.error(f"❌ Failed to build index: {e}")

    if st.button("📥 Load Index"):
        try:
            encryptor.load_index(index_path, password)
            st.success("✅ Index loaded and signature verified.")
        except Exception as e:
            st.error(f"❌ Failed to load index: {e}")

    st.subheader("💬 Search Text")
    query = st.text_input("Enter your search query")
    if st.button("🔍 Search") and query:
        try:
            results = encryptor.query_text(query, password, k=5)
            for ref, _ in results:
                st.write(f"📄 {os.path.basename(ref)}")
        except Exception as e:
            st.error(f"Search failed: {e}")

    st.subheader("🖼️ Upload Image to Search")
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if st.button("🔍 Search Image") and image:
        temp_path = "temp_uploaded.jpg"
        with open(temp_path, "wb") as f:
            f.write(image.read())
        try:
            results = encryptor.query_image(temp_path, password, k=5)
            for ref, _ in results:
                st.write(f"🖼️ {os.path.basename(ref)}")
        except Exception as e:
            st.error(f"Image search failed: {e}")
        finally:
            os.remove(temp_path)
