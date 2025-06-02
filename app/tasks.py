import os
import pickle
import numpy as np
from PIL import Image
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

from app.celery_app import celery
from app.user_manager import UserManager
from app.main import CLIPSecureEmbedder  # <-- Here we import the wrapper we wrote above

# ===============================
# 3.1. Define the CLIP model (using the wrapper)
# ===============================
# Instead of writing the processor/model again here, we simply use it from the wrapper
embedder = CLIPSecureEmbedder()  # Create a single instance for the lifetime of the module


# ===============================
# 3.2. Create / Load FAISS Index and mapping
# ===============================
def get_faiss_index_and_map(username: str):
    """
    Loads (if exists) or creates a new FAISS index and an ID->doc_id map.
    The store is saved in app/user_data/USERNAME/mm_index/:
      - faiss_index.bin
      - id_map.pkl
    The dimension of the CLIP vector is 512 (L2 / Cosine)
    """
    um = UserManager()
    user_folder = um.get_user_folder(username)
    mm_folder = os.path.join(user_folder, "mm_index")
    os.makedirs(mm_folder, exist_ok=True)

    index_path = os.path.join(mm_folder, "faiss_index.bin")
    map_path = os.path.join(mm_folder, "id_map.pkl")
    DIM = 512

    if os.path.exists(index_path) and os.path.exists(map_path):
        # Load existing FAISS index
        index = faiss.read_index(index_path)
        with open(map_path, 'rb') as f:
            id_map = pickle.load(f)
    else:
        # Create new FAISS Index
        index = faiss.IndexFlatIP(DIM)
        id_map = {}  # dict: faiss_int_id (int) -> doc_id (str)

    return index, id_map, index_path, map_path


def save_faiss_index_and_map(index, id_map, index_path, map_path):
    # Save index + id_map to disk
    faiss.write_index(index, index_path)
    with open(map_path, 'wb') as f:
        pickle.dump(id_map, f)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the CLIP‐ViT‐Base‐Patch32 model (512‐dim embeddings)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


def embed_text(text: str) -> np.ndarray:
    """
    Standalone function (no ‘self’) that returns a normalized 512‐dim CLIP text embedding.
    """
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    vec = outputs.cpu().numpy().flatten()   # shape: (512,)
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec  # → np.ndarray of shape (512,)


def embed_image(image_path: str) -> np.ndarray:
    """
    Standalone function (no ‘self’) that returns a normalized 512‐dim CLIP image embedding.
    We forcibly resize any tiny test image up to 224×224 RGB so that
    CLIP’s internal normalize() always sees 3 channels of reasonable size.
    """
    # 1. Load the image as RGB
    img = Image.open(image_path).convert("RGB")

    # 2. If it is smaller than 224×224 (e.g. a 1×1 test image), resize it:
    img = img.resize((224, 224))

    # 3. Pass to CLIPProcessor exactly one PIL Image:
    inputs = processor(images=img, return_tensors="pt").to(device)

    # 4. Extract CLIP image features
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    vec = outputs.cpu().numpy().flatten()  # shape: (512,)
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec  # → np.ndarray of shape (512,)


# ============================================================
# 3.4. Asynchronous task: Multimodal indexing (text + image)
# ============================================================
@celery.task(name='secure_clip.index_multimodal')
def index_multimodal(username: str, doc_id: str, text: str = None, image_path: str = None):
    """
    Asynchronous multimodal indexing:
    For each doc_id, if there is text – load text embedding,
                     if there is an image – load image embedding.
    Combine them (e.g., average) and add to the FAISS index.
    """
    # 1. Obtain existing FAISS index (or create new one)
    index, id_map, index_path, map_path = get_faiss_index_and_map(username)

    # 2. Extract embedded vectors for text and image
    vecs = []
    if text:
        txt_vec = embed_text(text)  # Here the CLIPSecureEmbedder.embed_text is used
        vecs.append(txt_vec)
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")
        img_vec = embed_image(image_path)  # Here the CLIPSecureEmbedder.embed_image is used
        vecs.append(img_vec)

    if not vecs:
        raise ValueError("No text or image provided for indexing")

    # 3. Combine vectors (e.g., simple average)
    combined_vec = np.mean(np.stack(vecs, axis=0), axis=0).astype('float32').reshape(1, -1)

    # 4. Add to FAISS index
    faiss_int_id = index.ntotal
    index.add(combined_vec)  # Adds it as a new vector
    id_map[faiss_int_id] = doc_id

    # 5. Save index + id_map back to disk
    save_faiss_index_and_map(index, id_map, index_path, map_path)

    return {"status": "indexed_multimodal", "doc_id": doc_id}


# ========================================================
# 3.5. Asynchronous task: Multimodal search
# ========================================================
@celery.task(name='secure_clip.search_multimodal')
def search_multimodal(username: str,
                      query_text: str = None,
                      query_image_path: str = None,
                      top_k: int = 5):
    """
    Multimodal search query:
    - Produce embedding vectors from text, image, or both
    - Perform FAISS search (inner product / cosine)
    - Return a list of tuples (doc_id, score)

    If index is empty – simply return {"results": []}
    """
    if not query_text and not query_image_path:
        raise ValueError("At least one of query_text or query_image_path must be provided")

    # 1. Generate the query vector
    vecs = []
    if query_text:
        q_txt_vec = embed_text(query_text)
        vecs.append(q_txt_vec)
    if query_image_path:
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Query image path not found: {query_image_path}")
        q_img_vec = embed_image(query_image_path)
        vecs.append(q_img_vec)

    combined_vec = np.mean(np.stack(vecs, axis=0), axis=0).astype('float32').reshape(1, -1)

    # 2. Load FAISS index + id_map
    index, id_map, index_path, map_path = get_faiss_index_and_map(username)
    if index.ntotal == 0:
        return {"results": []}

    # 3. Search in FAISS
    D, I = index.search(combined_vec, top_k)
    sims = D[0].tolist()
    ids = I[0].tolist()

    results = []
    for idx, sim in zip(ids, sims):
        if idx == -1:
            continue
        doc_id = id_map.get(idx, None)
        if doc_id is not None:
            results.append((doc_id, float(sim)))

    return {"results": results}
