import streamlit as st
import os
import json
import hashlib
from main import (
    CLIPSecureEmbedder,
    create_ckks_context,
    decrypt_vector_ckks,
    encrypt_vector_ckks,
    encrypt_file,
    decrypt_file,
    save_index,
    load_index,
    generate_fernet_key
)
import faiss
import numpy as np

# ========== UTILS ==========
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# ========== USER MANAGEMENT ==========
USER_FILE = "app/users.json"
VEC_DIM = 768

os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ========== FLEXIBLE TEXT DECODING ==========
def read_text_flexible(path):
    encodings_to_try = ["utf-8", "windows-1255", "iso-8859-8", "cp1252", "latin1"]
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Could not decode file with any known encoding.")

# ========== INIT ==========
st.set_page_config(page_title="🔐 Multi-User Secure Search", layout="wide")
embedder = CLIPSecureEmbedder()
context = create_ckks_context()

# ========== AUTH ==========
st.sidebar.title("👤 User Login")
users = load_users()
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
authenticated = False

if st.sidebar.button("🔐 Login"):
    if username in users and users[username]["password_hash"] == hash_password(password):
        st.session_state.user = username
        authenticated = True
        st.sidebar.success(f"Welcome, {username}!")
    else:
        st.sidebar.error("Invalid credentials")

if st.sidebar.button("📝 Register"):
    if username not in users:
        users[username] = {"password_hash": hash_password(password)}
        save_users(users)
        st.sidebar.success("User registered successfully!")
    else:
        st.sidebar.warning("User already exists")

# ========== MAIN APP ==========
if "user" in st.session_state:
    username = st.session_state.user
    data_folder = os.path.join("user_data", username, "data")
    index_path = os.path.join("user_data", username, "indexes", "index.pkl")
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if "index" not in st.session_state:
        st.session_state.index = faiss.IndexFlatIP(VEC_DIM)
        st.session_state.file_map = []

    option = st.sidebar.radio("Choose action:", ["🔍 Search", "📁 Upload Files", "🔓 Decrypt File", "🔬 Compare Vectors"])

    if option == "📁 Upload Files":
        st.header(f"📁 Upload Files for User: {username}")
        files = st.file_uploader("Upload images or text files", type=["jpg", "png", "jpeg", "txt"], accept_multiple_files=True)

        if st.button("🔐 Encrypt and Add to Index"):
            for file in files:
                file_path = os.path.join(data_folder, file.name)

                with open(file_path, "wb") as f:
                    f.write(file.read())

                if file.type.startswith("image"):
                    vec = embedder.embed_image(file_path)
                else:
                    text = read_text_flexible(file_path)
                    vec = embedder.embed_text(text)

                fernet_key = generate_fernet_key(password)
                encrypt_file(file_path, fernet_key)

                enc_vec = encrypt_vector_ckks(vec, context)
                dec_vec = decrypt_vector_ckks(enc_vec)
                norm_vec = normalize(dec_vec)
                st.session_state.index.add(np.array([norm_vec]).astype("float32"))
                st.session_state.file_map.append(file.name)

            save_index((st.session_state.index, st.session_state.file_map), index_path)
            st.success("✅ Files encrypted and indexed")

    elif option == "🔍 Search":
        st.header("🔍 Semantic Search")
        query_type = st.radio("Choose query type:", ["Text", "Image"])

        vec = None
        query_label = ""
        if query_type == "Text":
            query = st.text_input("Enter search text")
            if st.button("Search") and query:
                query_label = query
                vec = embedder.embed_text(query)

        elif query_type == "Image":
            uploaded_image = st.file_uploader("Upload query image", type=["jpg", "png", "jpeg"], key="query_img")
            if st.button("Search") and uploaded_image:
                temp_path = os.path.join("app/temp_query_image.jpg")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_image.read())
                query_label = uploaded_image.name
                vec = embedder.embed_image(temp_path)

        if vec is not None:
            if not os.path.exists(index_path):
                st.error("No index found")
            else:
                index, file_map = load_index(index_path)
                enc_vec = encrypt_vector_ckks(vec, context)
                dec_vec = decrypt_vector_ckks(enc_vec)
                norm_vec = normalize(dec_vec)
                D, I = index.search(np.array([norm_vec]).astype("float32"), k=5)

                st.subheader(f"🔎 Top Results for '{query_label}' (higher = more similar):")
                for idx, (i, dist) in enumerate(zip(I[0], D[0])):
                    filename = file_map[i]
                    encrypted_path = os.path.join(data_folder, filename + ".enc")
                    try:
                        fernet_key = generate_fernet_key(password)
                        decrypted = decrypt_file(encrypted_path, fernet_key)
                        with st.container():
                            st.markdown(f"**Similarity Score:** {dist:.4f}")
                            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                                from PIL import Image
                                import io
                                image = Image.open(io.BytesIO(decrypted))
                                st.image(image, caption=f"🖼️ {filename}", width=200, use_container_width=False)
                                st.download_button(
                                    label="⬇️ Download Image",
                                    data=decrypted,
                                    file_name=filename,
                                    mime="image/jpeg",
                                    key=f"download_img_{idx}"
                                )
                            elif filename.lower().endswith(".txt"):
                                text_full = decrypted.decode("utf-8", errors="ignore")
                                text_preview = text_full[:300]
                                st.markdown(f"📄 **{filename}**")
                                st.text_area("Preview:", text_preview, height=80, key=f"preview_{idx}")
                                with st.expander("📖 Show full text", expanded=False):
                                    st.text_area("Full Text:", text_full, height=160, key=f"full_text_{idx}")
                                st.download_button(
                                    label="⬇️ Download Text File",
                                    data=text_full,
                                    file_name=filename,
                                    mime="text/plain",
                                    key=f"download_txt_{idx}"
                                )
                            else:
                                st.markdown(f"📁 {filename} (Unsupported preview type)")
                    except Exception as e:
                        st.error(f"❌ Failed to decrypt {filename}: {str(e)}")

    elif option == "🔬 Compare Vectors":
        st.header("🔬 Compare text and image embedding vectors")
        col1, col2 = st.columns(2)

        with col1:
            text_query = st.text_input("Enter text (e.g. 'ring')", key="cmp_text")
            if text_query:
                text_vec = embedder.embed_text(text_query)

        with col2:
            cmp_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="cmp_img")
            if cmp_image:
                temp_path = os.path.join("app/temp_cmp_image.jpg")
                with open(temp_path, "wb") as f:
                    f.write(cmp_image.read())
                img_vec = embedder.embed_image(temp_path)

        if text_query and cmp_image:
            norm_text = normalize(text_vec)
            norm_img = normalize(img_vec)
            dist = np.dot(norm_text, norm_img)
            st.success(f"Cosine similarity between '{text_query}' and image: {dist:.4f} (higher = more similar)")

    elif option == "🔓 Decrypt File":
        st.header("🔓 Decrypt File")
        enc_file = st.file_uploader("Upload .enc file", type=["enc"])

        if st.button("🔓 Decrypt") and enc_file:
            encrypted_path = os.path.join(data_folder, enc_file.name)
            with open(encrypted_path, "wb") as f:
                f.write(enc_file.read())

            try:
                fernet_key = generate_fernet_key(password)
                decrypted = decrypt_file(encrypted_path, fernet_key)
                st.success("✅ Decryption successful")
                if enc_file.name.endswith(".jpg.enc"):
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(decrypted))
                    st.image(image, caption="Decrypted Image")
                else:
                    st.text_area("Decrypted Content", decrypted.decode("utf-8"))
            except Exception as e:
                st.error(f"❌ Decryption failed: {str(e)}")
else:
    st.warning("🔐 Please log in to use the application")