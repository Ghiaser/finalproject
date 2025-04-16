import os
import streamlit as st
from main_final import CLIPSecureEncryptor
from PIL import Image

st.set_page_config(page_title="🔐 Secure Semantic Search", layout="centered")
st.title("🔐 Secure Semantic File Search")

# Password input
password = st.text_input("🔑 Enter your secret password", type="password")

# Folder path input
folder = st.text_input("📁 Enter path to folder with files (txt / jpg / png)", value="C:\\shaked\\DATA")

# Path to save/load the index
index_path = os.path.join(folder, "secure_index.pkl")

# Initialize encryptor and state only once
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "encryptor" not in st.session_state:
    st.session_state.encryptor = CLIPSecureEncryptor()

encryptor = st.session_state.encryptor

if password and folder:
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.txt'))]

    # Load existing index if it exists
    if not st.session_state.index_ready and os.path.exists(index_path):
        try:
            encryptor.load_index(index_path)
            st.session_state.index_ready = True
            st.success("✅ Index auto-loaded.")
        except Exception as e:
            st.error(f"❌ Failed to auto-load index: {e}")

    # Build new index and save it
    if st.button("🔒 Encrypt & Build Index"):
        with st.spinner("Encrypting files and building index..."):
            try:
                encryptor.build_index_from_files(files, password)
                encryptor.save_index(index_path)
                st.session_state.index_ready = True
                st.success("✅ Index built and saved.")
            except Exception as e:
                st.error(f"❌ Failed to build index: {e}")

    # Manual loading of index
    if st.button("📦 Load Index"):
        try:
            encryptor.load_index(index_path)
            st.session_state.index_ready = True
            st.success("✅ Index loaded.")
        except Exception as e:
            st.error(f"❌ Error loading index: {e}")

    # Search section
    if st.session_state.index_ready:
        search_query = st.text_input("💬 Search by text")
        image_file = st.file_uploader("🖼️ Or upload an image to search", type=["jpg", "jpeg", "png"])

        if st.button("🔍 Search"):
            if search_query:
                try:
                    results = encryptor.query_text(search_query, password=password, k=10)

                    text_results = [r for r in results if r.lower().endswith(".txt")][:3]
                    image_results = [r for r in results if r.lower().endswith((".jpg", ".jpeg", ".png"))][:3]

                    st.subheader("📄 Top text matches:")
                    for r in text_results:
                        st.markdown(f"**{os.path.basename(r)}**")
                        try:
                            with open(r, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read(300)
                            st.code(content.strip() + "...")
                        except:
                            st.warning("Could not preview text file.")

                    st.subheader("🖼️ Top image matches:")
                    for r in image_results:
                        st.markdown(f"**{os.path.basename(r)}**")
                        st.image(r, width=300)

                except Exception as e:
                    st.error(f"Search failed: {e}")

            elif image_file:
                with open("temp_img.jpg", "wb") as f:
                    f.write(image_file.read())
                try:
                    results = encryptor.query_image("temp_img.jpg", password=password, k=10)

                    text_results = [r for r in results if r.lower().endswith(".txt")][:3]
                    image_results = [r for r in results if r.lower().endswith((".jpg", ".jpeg", ".png"))][:3]

                    st.subheader("📄 Top text matches (image query):")
                    for r in text_results:
                        st.markdown(f"**{os.path.basename(r)}**")
                        try:
                            with open(r, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read(300)
                            st.code(content.strip() + "...")
                        except:
                            st.warning("Could not preview text file.")

                    st.subheader("🖼️ Top image matches:")
                    for r in image_results:
                        st.markdown(f"**{os.path.basename(r)}**")
                        st.image(r, width=300)

                except Exception as e:
                    st.error(f"Image search failed: {e}")
