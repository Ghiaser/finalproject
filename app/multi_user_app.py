import os
import streamlit as st
import torch
import warnings
import shutil
from main import CLIPSecureEncryptor
from user_manager import UserManager

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variable to fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Force PyTorch to use CPU
if torch.cuda.is_available():
    torch.cuda.is_available = lambda: False

st.set_page_config(page_title="🔐 Multi-User Secure CLIP", layout="centered")

# Initialize user manager
user_manager = UserManager()

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "encryptor" not in st.session_state:
    st.session_state.encryptor = CLIPSecureEncryptor()
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

# Authentication UI
def show_auth():
    st.title("🔐 Secure CLIP Search - Multi-User")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit and username and password:
                success, message = user_manager.authenticate(username, password)
                if success:
                    st.session_state.user = username
                    st.session_state.password = password  # Store for index operations
                    st.session_state.index_loaded = False
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit and new_username and new_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = user_manager.create_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

# Main app UI
def show_app():
    st.title(f"🧪 Secure Semantic Search - {st.session_state.user}")
    
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.password = None
        st.session_state.index_loaded = False
        st.rerun()
    
    username = st.session_state.user
    password = st.session_state.password
    user_folder = user_manager.get_user_folder(username)
    data_folder = f"{user_folder}/data"
    indexes_folder = f"{user_folder}/indexes"
    
    # Sidebar for data management
    with st.sidebar:
        st.header("Data Management")
        
        # File upload
        uploaded_files = st.file_uploader("Upload files", type=["txt", "jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                file_path = os.path.join(data_folder, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"Uploaded {len(uploaded_files)} files to your data folder")
            
        # Show existing files
        st.subheader("Your Files")
        if os.path.exists(data_folder):
            files = [f for f in os.listdir(data_folder) 
                    if f.lower().endswith((".txt", ".jpg", ".jpeg", ".png"))]
            if files:
                for file in files:
                    st.write(f"- {file}")
            else:
                st.write("No files found. Upload some files to get started.")
    
    # Main content for search
    st.header("Index Management")
    
    tab1, tab2 = st.tabs(["Create Index", "Load Index"])
    
    with tab1:
        index_name = st.text_input("Index Name")
        if st.button("🔨 Build Index") and index_name:
            try:
                files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
                         if f.lower().endswith((".txt", ".jpg", ".jpeg", ".png"))]
                
                if not files:
                    st.error("No files found in your data folder. Upload some files first.")
                else:
                    encryptor = st.session_state.encryptor
                    encryptor.build_index_from_files(files, password)
                    
                    index_path = os.path.join(indexes_folder, f"{index_name}.pkl")
                    encryptor.save_index(index_path, password)
                    
                    user_manager.add_user_index(username, index_name)
                    st.session_state.index_loaded = True
                    st.success("✅ Index built and saved with signature.")
            except Exception as e:
                st.error(f"❌ Failed to build index: {e}")
    
    with tab2:
        user_indexes = user_manager.get_user_indexes(username)
        if not user_indexes:
            st.info("You don't have any indexes yet. Create one in the 'Create Index' tab.")
        else:
            selected_index = st.selectbox("Select an index", user_indexes)
            if st.button("📥 Load Index") and selected_index:
                try:
                    index_path = os.path.join(indexes_folder, f"{selected_index}.pkl")
                    encryptor = st.session_state.encryptor
                    encryptor.load_index(index_path, password)
                    st.session_state.index_loaded = True
                    st.success("✅ Index loaded and signature verified.")
                except Exception as e:
                    st.error(f"❌ Failed to load index: {e}")
    
    # Search functionality
    st.header("Search")
    
    if not st.session_state.index_loaded:
        st.info("Please build or load an index first to enable search functionality.")
    else:
        st.subheader("💬 Text Search")
        query = st.text_input("Enter your search query")
        if st.button("🔍 Search Text") and query:
            try:
                results = st.session_state.encryptor.query_text(query, password, k=5)
                st.subheader("Results:")
                for i, (ref, score) in enumerate(results):
                    st.write(f"{i+1}. 📄 {os.path.basename(ref)} (Score: {score:.2f})")
            except Exception as e:
                st.error(f"Search failed: {e}")
        
        st.subheader("🖼️ Image Search")
        image = st.file_uploader("Upload an image to search", type=["jpg", "jpeg", "png"])
        if st.button("🔍 Search Image") and image:
            temp_path = "temp_uploaded.jpg"
            with open(temp_path, "wb") as f:
                f.write(image.read())
            try:
                results = st.session_state.encryptor.query_image(temp_path, password, k=5)
                st.subheader("Results:")
                for i, (ref, score) in enumerate(results):
                    st.write(f"{i+1}. 🖼️ {os.path.basename(ref)} (Score: {score:.2f})")
            except Exception as e:
                st.error(f"Image search failed: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# Main app flow
if st.session_state.user is None:
    show_auth()
else:
    show_app()