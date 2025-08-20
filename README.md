# Multi-User Secure Multimodal Search

This repository implements a **secure, multi-user** system for indexing and searching **multimodal (text + image)** data. The system combines:
- **End-to-end encryption** of user files (images & text) using Fernet (via `cryptography`),  
- **Homomorphic‐style encrypted embeddings** via CKKS (TenSEAL),  
- **Multimodal embedding** with CLIP (Hugging Face Transformers),  
- **FAISS** for fast similarity search on combined embeddings,  
- **Celery** (in *eager* mode) to demonstrate asynchronous tasks,  
- A **FastAPI**–based backend for API endpoints (optional),  
- A **Streamlit**–based front-end for user interaction (uploading, indexing, searching, downloading),  
- **User management** (username/password, salted+hashed) with isolated “data” & “indexes” folders per user,  
- **PyTest** test suite covering embedding functions and multimodal indexing/search.

---


- **`app/`**   
  Contains all business logic, including:
  - **`user_manager.py`** – handles user registration/authentication, per-user folders, salts & hashed passwords.  
  - **`encryptor.py`** – functions to encrypt/decrypt files (Fernet) and to create CKKS contexts & encrypt/decrypt vectors.  
  - **`main.py`** – a high-level wrapper around CLIP: `CLIPSecureEmbedder` (text/image embedding with L2-norm), CKKS contexts, helper functions for file encryption & index saving/loading.  
  - **`tasks.py`** – Celery tasks for:
    - `index_multimodal(username, doc_id, text, image_path)` → eager multimodal indexing (compute CLIP embeddings, combine by average, add to FAISS index on disk).
    - `search_multimodal(username, query_text, query_image_path, top_k)` → eager multimodal search (compute combined query embedding, run FAISS search, return `(doc_id, score)` results).
  - **`api.py`** – (in case you want to expose FastAPI endpoints; these can be called by a frontend or used for programmatic access).  
  - **`celery_app.py`** – configures Celery in *eager* (synchronous) mode, with a Redis broker/backend (for demo; no external workers needed).  
  - **`user_data/`** is the base folder where, for each `username`, you will find:
    - `user_data/<username>/data/` – encrypted files (e.g. `foo.jpg.enc`, `bar.txt.enc`).  
    - `user_data/<username>/mm_index/` – FAISS index (binary) + `id_map.pkl` storing `faiss_int_id → doc_id`.

- **`multi_user_app.py`**   
  The Streamlit application. It supports:
  1. **Login/Register** (username/password).  
  2. **Upload & Encrypt** image/text files → generate CLIP embeddings, encrypt files with user’s Fernet key, index vectors in FAISS.  
  3. **Search** (text / image / multimodal) → send query to `search_multimodal`, decrypt & display results.  
  4. **Decrypt** a previously encrypted file upon request.  
  5. **Compare Vectors** (text vs. image cosine similarity demo).

- **`requirements.txt`**   
  Lists all Python dependencies for the entire project. (See below.)

- **`tests/`**   
  Contains PyTest fixtures & tests for:
  - Verifying that **`embed_text(...)`** & **`embed_image(...)`** produce normalized (∥v∥≈1) 512-dim vectors.  
  - Verifying that **`index_multimodal(...)`** creates a FAISS index + id_map, and that **`search_multimodal(...)`** correctly returns the indexed `doc_id` when querying by text alone, image alone, and combined.

---

## 🚀 Quick Start

### 1. Prerequisites

1. **Python 3.10+** (tested on 3.10.12).  
2. **Redis** running locally (on `localhost:6379`) for Celery’s broker & backend.  
   ```bash
   # Ubuntu / Debian:
   sudo apt update
   sudo apt install redis
   sudo service redis start

# 1. (Optional) If you haven’t already, start Redis on localhost:6379
#    (Skip this step if you're using Celery in “eager” mode—Redis is not strictly required.)
redis-server &

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install all project dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) If you want real asynchronous Celery tasks, disable eager mode in app/celery_app.py 
#    by commenting out or removing:
#      celery.conf.task_always_eager = True
#      celery.conf.task_eager_propagates = True
#
#    Then start a Celery worker. Otherwise, you can skip this—tasks will run synchronously.
celery -A app.celery_app.celery worker --loglevel=info &

# 5. (Optional) If you want to expose a FastAPI backend, launch Uvicorn:
uvicorn app.api:app --reload &

# 6. Launch the Streamlit front-end. It will automatically use the same Celery configuration.
streamlit run multi_user_app.py
# 7. Architecture Diagram
flowchart LR
  subgraph UI["User Interface"]
    S[Streamlit UI]
    A[FastAPI (optional)]
  end

  subgraph APP["Application Logic"]
    C[Celery Tasks<br/>index_multimodal / search_multimodal]
    UM[User Manager<br/>auth + salts + folders]
    ENC[Encryptor<br/>Fernet files + CKKS demo]
    EMB[CLIP Embedder<br/>Text/Image 512-dim, L2]
    IDX[FAISS Index<br/>IndexFlatIP (cosine via L2)]
  end

  subgraph STORAGE["Per-User Storage"]
    D1["user_data/<user>/data<br/>(encrypted files *.enc)"]
    D2["user_data/<user>/(indexes | mm_index)<br/>(faiss + maps)"]
  end

  S --> C
  A --> C
  C --> UM
  C --> ENC
  C --> EMB
  EMB --> IDX
  C --> IDX
  ENC --> D1
  IDX --> D2
  D2 --> C
  D1 --> S


