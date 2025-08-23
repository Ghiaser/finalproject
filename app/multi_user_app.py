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

    # ================== UPLOAD FILES ==================
    if option == "📁 Upload Files":
        st.header(f"📁 Upload Files for User: {username}")
        files = st.file_uploader(
            "Upload images or text files",
            type=["jpg", "png", "jpeg", "txt"],
            accept_multiple_files=True
        )

        if st.button("🔐 Encrypt and Add to Index"):
            if not files:
                st.error("Please select at least one file.")
            else:
                for file in files:
                    file_path = os.path.join(data_folder, file.name)

                    # 1. Save the file temporarily as plain
                    with open(file_path, "wb") as f:
                        f.write(file.read())

                    # 2. Compute appropriate embedding vector (image or text)
                    if file.name.lower().endswith(("jpg", "jpeg", "png")):
                        vec = embedder.embed_image(file_path)
                    else:  # assume .txt
                        text = read_text_flexible(file_path)
                        vec = embedder.embed_text(text)

                    # 3. Generate Fernet key from password and encrypt the file
                    fernet_key = generate_fernet_key(password)
                    encrypt_file(file_path, fernet_key)

                    # 4. CKKS encrypt the vector and decrypt immediately to ensure integrity
                    enc_vec = encrypt_vector_ckks(vec, context)
                    dec_vec = decrypt_vector_ckks(enc_vec)
                    norm_vec = normalize(dec_vec)

                    # 5. Add to local FAISS index
                    st.session_state.index.add(np.array([norm_vec]).astype("float32"))
                    st.session_state.file_map.append(file.name)

                # 6. Save index and file_map to disk
                save_index((st.session_state.index, st.session_state.file_map), index_path)
                st.success("✅ Files encrypted and indexed")

    # ================== SEARCH ==================
    elif option == "🔍 Search":
        st.header("🔍 Semantic Search")

        # 1. Add multimodal search option alongside Text-only and Image-only
        query_mode = st.radio("Choose query type:", ["Text-only", "Image-only", "Multimodal (Text + Image)"])

        vec = None
        query_label = ""
        combined_vec = None

        if query_mode == "Text-only":
            query_text = st.text_input("Enter search text:")
            if st.button("Search Text"):
                if not query_text:
                    st.error("Please enter some text to search.")
                else:
                    query_label = query_text
                    vec = embedder.embed_text(query_text)

        elif query_mode == "Image-only":
            uploaded_image = st.file_uploader("Upload query image", type=["jpg", "png", "jpeg"], key="search_img")
            if st.button("Search Image"):
                if not uploaded_image:
                    st.error("Please upload an image to search.")
                else:
                    temp_path = os.path.join("app", "temp_query_image.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_image.read())
                    query_label = uploaded_image.name
                    vec = embedder.embed_image(temp_path)

        else:  # Multimodal (Text + Image)
            col1, col2 = st.columns(2)
            with col1:
                query_text_m = st.text_input("Enter search text (optional)", key="mm_text")
            with col2:
                uploaded_image_m = st.file_uploader("Upload query image (optional)", type=["jpg", "png", "jpeg"], key="mm_img")

            if st.button("Search Multimodal"):
                if not query_text_m and not uploaded_image_m:
                    st.error("Please provide text and/or an image to search.")
                else:
                    vecs = []
                    if query_text_m:
                        query_label = query_text_m
                        txt_vec = embedder.embed_text(query_text_m)
                        vecs.append(txt_vec)
                    if uploaded_image_m:
                        temp_path_m = os.path.join("app", "temp_query_image_m.jpg")
                        with open(temp_path_m, "wb") as f:
                            f.write(uploaded_image_m.read())
                        query_label = query_text_m + " + " + uploaded_image_m.name if query_text_m else uploaded_image_m.name
                        img_vec = embedder.embed_image(temp_path_m)
                        vecs.append(img_vec)

                    # Average the normalized vectors to get a combined vector
                    stacked = np.stack([normalize(v) for v in vecs], axis=0)
                    combined = np.mean(stacked, axis=0)
                    vec = combined  # combined 512-dim vector

        # 2. If there is a query vector, perform FAISS search
        if vec is not None:
            if not os.path.exists(index_path):
                st.error("No index found for this user.")
            else:
                # 2.1. Load index and file_map
                index, file_map = load_index(index_path)

                # 2.2. CKKS encrypt/decrypt (not necessary here, but keeps consistency)
                enc_vec = encrypt_vector_ckks(vec, context)
                dec_vec = decrypt_vector_ckks(enc_vec)
                norm_query = normalize(dec_vec)

                # 2.3. FAISS search: returns distances (D) and indices (I)
                D, I = index.search(np.array([norm_query]).astype("float32"), k=5)

                st.subheader(f"🔎 Top Results for '{query_label}' (higher = more similar):")
                for idx, dist in zip(I[0], D[0]):
                    if idx == -1:
                        continue
                    filename = file_map[idx]
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

    # ========== COMPARE VECTORS ==========
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
                temp_path = os.path.join("app", "temp_cmp_image.jpg")
                with open(temp_path, "wb") as f:
                    f.write(cmp_image.read())
                img_vec = embedder.embed_image(temp_path)

        if 'text_vec' in locals() and 'img_vec' in locals():
            norm_text = normalize(text_vec)
            norm_img = normalize(img_vec)
            dist = np.dot(norm_text, norm_img)
            st.success(f"Cosine similarity between '{text_query}' and image: {dist:.4f} (higher = more similar)")

    # ========== DECRYPT FILE ==========
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
                    st.text_area("Decrypted Content", decrypted.decode("utf-8", errors="ignore"))
            except Exception as e:
                st.error(f"❌ Decryption failed: {str(e)}")
else:
    st.warning("🔐 Please log in to use the application")
