
import os
import streamlit as st
from main_final import CLIPSecureEncryptor
from PIL import Image

st.set_page_config(page_title="🔐 Secure Semantic Search", layout="centered")
st.title("🔐 Secure Semantic File Search")

password = st.text_input("Enter your secret password", type="password")

folder = "/home/danielbes/Desktop/BETA/DATA"
index_path = "my_index.pkl"
files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.jpg', '.png', '.txt'))]

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "encryptor" not in st.session_state:
    st.session_state.encryptor = CLIPSecureEncryptor()

encryptor = st.session_state.encryptor

if password:
    if not st.session_state.index_ready and os.path.exists(index_path):
        try:
            encryptor.load_index(index_path)
            st.session_state.index_ready = True
            st.success("✅ Index auto-loaded.")
        except Exception as e:
            st.error(f"❌ Failed to auto-load index: {e}")

    if st.button("🔨 Build Index"):
        with st.spinner("Building index, please wait..."):
            encryptor.build_index_from_files(files, password)
            encryptor.save_index(index_path)
            st.session_state.index_ready = True
            st.success("✅ Index built and saved!")

    if st.button("📦 Load Index"):
        try:
            encryptor.load_index(index_path)
            st.session_state.index_ready = True
            st.success("✅ Index loaded.")
        except Exception as e:
            st.error(f"❌ Error loading index: {e}")

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
                    if text_results:
                        for r in text_results:
                            st.markdown(f"**`{os.path.basename(r)}`**")
                            try:
                                with open(r, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read(300)
                                st.code(content.strip() + "...")
                            except:
                                st.warning("Could not preview text file.")
                    else:
                        st.info("No matching text files found.")

                    st.subheader("🖼️ Top image matches:")
                    if image_results:
                        for r in image_results:
                            st.markdown(f"**`{os.path.basename(r)}`**")
                            st.image(r, width=300)
                    else:
                        st.info("No matching images found.")

                except Exception as e:
                    st.error(f"Search failed: {e}")

            if image_file:
                with open("temp_img.jpg", "wb") as f:
                    f.write(image_file.read())
                try:
                    results = encryptor.query_image("temp_img.jpg", password=password, k=10)

                    text_results = [r for r in results if r.lower().endswith(".txt")][:3]
                    image_results = [r for r in results if r.lower().endswith((".jpg", ".jpeg", ".png"))][:3]

                    st.subheader("📄 Top text matches (image query):")
                    if text_results:
                        for r in text_results:
                            st.markdown(f"**`{os.path.basename(r)}`**")
                            try:
                                with open(r, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read(300)
                                st.code(content.strip() + "...")
                            except:
                                st.warning("Could not preview text file.")
                    else:
                        st.info("No matching text files found.")

                    st.subheader("🖼️ Top image matches:")
                    if image_results:
                        for r in image_results:
                            st.markdown(f"**`{os.path.basename(r)}`**")
                            st.image(r, width=300)
                    else:
                        st.write("_No matching images found._")


                except Exception as e:
                    st.error(f"Image search failed: {e}")