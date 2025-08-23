# app/api.py

import os
import tempfile
import base64

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Body
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Celery
from celery.result import AsyncResult
from celery_app import celery

# Existing modules
from user_manager import UserManager
from encryptor import CLIPSecureEmbedder
from tasks import index_multimodal, search_multimodal

# ===============================
# 4.1. FastAPI initialization, CORS settings, BasicAuth
# ===============================
app = FastAPI(
    title="Multi-Modal Secure Search API",
    description="מערכת אינדוקס + חיפוש רב־ממדי (טקסט + תמונה) עם Celery + Flower",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()
user_manager = UserManager(users_file="app/user_data/users.json")
encryptor = CLIPSecureEmbedder()


# ===============================
# 4.2. Pydantic Models
# ===============================
class UserCreate(BaseModel):
    username: str
    password: str


class IndexDocRequest(BaseModel):
    doc_id: str
    text: str = None     # optional
    # image will come as UploadFile in multipart, not as Base64 here


class SearchTextRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchImageBase64Request(BaseModel):
    image_base64: str
    top_k: int = 5


# ===============================
# 4.3. Authenticator
# ===============================
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    success, _ = user_manager.authenticate(username, password)
    if not success:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"}
        )
    return {"username": username, "password": password}


# ===============================
# 4.4. Endpoint: Create User
# ===============================
@app.post("/users", status_code=201, tags=["Users"])
async def create_user(user: UserCreate):
    success, message = user_manager.create_user(user.username, user.password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@app.get("/users/me", tags=["Users"])
async def get_current_user(user: dict = Depends(authenticate_user)):
    username = user["username"]
    indexes = user_manager.get_user_indexes(username)
    return {"username": username, "indexes": indexes}


# ===============================
# 4.5. Endpoint: Upload files + simultaneous multimodal indexing
# ===============================
@app.post("/index/document", status_code=202, tags=["Index"])
async def index_document(
    doc_id: str = Form(...),
    text_file: UploadFile = File(None),        # optional
    image_file: UploadFile = File(None),       # optional
    user: dict = Depends(authenticate_user)
):
    """
    Asynchronous multimodal document indexing:
    1. Receives doc_id (unique ID), text or image (or both).
    2. Temporarily saves them, decrypts, then sends to tasks.index_multimodal.delay(...)
    """
    username = user["username"]
    password = user["password"]

    # Must send at least text or image
    if (not text_file or not text_file.filename) and (not image_file or not image_file.filename):
        raise HTTPException(status_code=400, detail="At least one of text_file or image_file must be provided")

    user_folder = user_manager.get_user_folder(username)
    data_folder = os.path.join(user_folder, "data")
    os.makedirs(data_folder, exist_ok=True)

    # ###### 1. Handling text file (if exists) ######
    text_contents = None
    if text_file and text_file.filename.lower().endswith(".txt"):
        # 1.1. Save temporary file
        temp_txt_path = os.path.join(data_folder, f"{doc_id}.txt")
        contents = await text_file.read()
        with open(temp_txt_path, "wb") as f:
            f.write(contents)
        # 1.2. Read as plain text (UTF-8 expected)
        try:
            with open(temp_txt_path, "r", encoding="utf-8") as f:
                text_contents = f.read()
        except Exception:
            # If reading with UTF-8 fails, try Latin-1 as fallback
            with open(temp_txt_path, "r", encoding="latin-1", errors="ignore") as f:
                text_contents = f.read()

        # 1.3. Encrypt and write back (Fernet)
        encryptor.encrypt_file(temp_txt_path, password)
        # Now the persisted file in data_folder is temp_txt_path + ".enc"

    # ###### 2. Handling image file (if exists) ######
    temp_img_path = None
    if image_file and image_file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # 2.1. Save temporary file
        fname = f"{doc_id}{os.path.splitext(image_file.filename)[1]}"
        temp_img_path = os.path.join(data_folder, fname)
        contents = await image_file.read()
        with open(temp_img_path, "wb") as f:
            f.write(contents)
        # 2.2. Encrypt and write (.enc)
        encryptor.encrypt_file(temp_img_path, password)
        # Now the file in data_folder is temp_img_path + ".enc"

    # ###### 3. Temporary decryption for the task ######
    temp_dec_folder = os.path.join(user_folder, "temp_decrypted")
    os.makedirs(temp_dec_folder, exist_ok=True)

    # 3.1. Decrypt text (if exists)
    plain_txt_path = None
    if text_contents is not None:
        enc_txt_path = os.path.join(data_folder, f"{doc_id}.txt.enc")
        plain_txt_path = os.path.join(temp_dec_folder, f"{doc_id}.txt")
        decrypted_bytes = encryptor.decrypt_file(enc_txt_path, password)
        with open(plain_txt_path, "wb") as f:
            f.write(decrypted_bytes)

    # 3.2. Decrypt image (if exists)
    plain_img_path = None
    if temp_img_path is not None:
        enc_img_path = temp_img_path + ".enc"
        ext = os.path.splitext(temp_img_path)[1]
        plain_img_path = os.path.join(temp_dec_folder, f"{doc_id}{ext}")
        decrypted_bytes = encryptor.decrypt_file(enc_img_path, password)
        with open(plain_img_path, "wb") as f:
            f.write(decrypted_bytes)

    # ###### 4. Call the Celery task ######
    task = index_multimodal.delay(username, doc_id, text_contents, plain_img_path)

    return {"task_id": task.id, "status": "queued"}


# ===============================
# 4.6. Endpoint: Check task status
# ===============================
@app.get("/task_status/{task_id}", tags=["Tasks"])
async def task_status(task_id: str):
    """
    Returns the status of a Celery task by task_id:
      - PENDING / STARTED / RETRY / SUCCESS / FAILURE
      - If SUCCESS → also returns res.result
    """
    res = AsyncResult(task_id, app=celery)
    state = res.state

    if state == 'PENDING':
        return {"status": "pending"}
    elif state == 'SUCCESS':
        return {"status": "success", "result": res.result}
    elif state == 'FAILURE':
        return {"status": "failure", "error": str(res.result)}
    else:
        return {"status": state}


# ===============================
# 4.7. Endpoints: Multimodal Search
# ===============================

@app.post("/search/text", status_code=202, tags=["Search"])
async def search_text(
    search_req: SearchTextRequest,
    user: dict = Depends(authenticate_user)
):
    """
    Asynchronous text search:
    1. Generate embedding for the search text
    2. Run search_multimodal(username, query_text, None, top_k)
    """
    username = user["username"]
    query = search_req.query
    top_k = search_req.top_k

    task = search_multimodal.delay(username, query, None, top_k)
    return {"task_id": task.id, "status": "queued"}


@app.post("/search/image", status_code=202, tags=["Search"])
async def search_image(
    image: UploadFile = File(...),
    user: dict = Depends(authenticate_user),
    top_k: int = Form(5)
):
    """
    Asynchronous image search:
    1. Receive image file
    2. Temporarily save it
    3. Run search_multimodal(username, None, image_path, top_k)
    """
    username = user["username"]
    password = user["password"]

    user_folder = user_manager.get_user_folder(username)
    temp_upload_folder = os.path.join(user_folder, "temp_query_images")
    os.makedirs(temp_upload_folder, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=temp_upload_folder, delete=False, suffix=".jpg") as temp:
        temp_path = temp.name
        content = await image.read()
        temp.write(content)

    task = search_multimodal.delay(username, None, temp_path, top_k)
    return {"task_id": task.id, "status": "queued"}


@app.post("/search/multimodal", status_code=202, tags=["Search"])
async def search_multimodal_endpoint(
    text: str = Form(None),
    image: UploadFile = File(None),
    top_k: int = Form(5),
    user: dict = Depends(authenticate_user)
):
    """
    Multimodal query (text and/or image), asynchronous:
    1. If there is text → set query_text
    2. If there is image → save temporarily and set query_image_path
    3. Run search_multimodal(username, query_text, query_image_path, top_k)
    """
    username = user["username"]
    query_text = text
    query_image_path = None

    user_folder = user_manager.get_user_folder(username)
    temp_query_folder = os.path.join(user_folder, "temp_query_images")
    os.makedirs(temp_query_folder, exist_ok=True)

    if image:
        with tempfile.NamedTemporaryFile(dir=temp_query_folder, delete=False, suffix=".jpg") as temp:
            temp_path = temp.name
            content = await image.read()
            temp.write(content)
        query_image_path = temp_path

    if not query_text and not query_image_path:
        raise HTTPException(status_code=400, detail="At least one of text or image must be provided")

    task = search_multimodal.delay(username, query_text, query_image_path, top_k)
    return {"task_id": task.id, "status": "queued"}


@app.post("/search/image/base64", status_code=202, tags=["Search"])
async def search_image_base64(
    payload: SearchImageBase64Request = Body(...),
    user: dict = Depends(authenticate_user)
):
    """
    Asynchronous BASE64 image search:
    Receives JSON: {"image_base64": "...", "top_k":5}
    """
    username = user["username"]
    raw_b64 = payload.image_base64
    top_k = payload.top_k

    user_folder = user_manager.get_user_folder(username)
    temp_folder = os.path.join(user_folder, "temp_query_images")
    os.makedirs(temp_folder, exist_ok=True)

    try:
        img_data = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    with tempfile.NamedTemporaryFile(dir=temp_folder, delete=False, suffix=".jpg") as temp:
        temp_path = temp.name
        temp.write(img_data)

    task = search_multimodal.delay(username, None, temp_path, top_k)
    return {"task_id": task.id, "status": "queued"}


# ===============================
# 4.8. Running Uvicorn
# ===============================
if __name__ == "__main__":
    os.makedirs("app/user_data", exist_ok=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
