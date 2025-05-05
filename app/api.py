import os
import tempfile
import base64
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from main import CLIPSecureEncryptor
from user_manager import UserManager

# Initialize FastAPI
app = FastAPI(
    title="Secure CLIP API",
    description="API for secure semantic search using CLIP",
    version="1.0.0"
)

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize security
security = HTTPBasic()

# Initialize user manager
user_manager = UserManager()

# Initialize encryptor (will be user-specific in routes)
encryptor = CLIPSecureEncryptor()

# Models
class UserCreate(BaseModel):
    username: str
    password: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class IndexCreate(BaseModel):
    index_name: str

class SearchResult(BaseModel):
    filename: str
    score: float

# Authentication dependency
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    
    success, _ = user_manager.authenticate(username, password)
    if not success:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"username": username, "password": password}

# Routes
@app.post("/users", status_code=201, tags=["Users"])
async def create_user(user: UserCreate):
    """Create a new user"""
    success, message = user_manager.create_user(user.username, user.password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@app.get("/users/me", tags=["Users"])
async def get_current_user(user: dict = Depends(authenticate_user)):
    """Get current user info"""
    username = user["username"]
    indexes = user_manager.get_user_indexes(username)
    return {
        "username": username,
        "indexes": indexes
    }

@app.post("/upload", tags=["Data"])
async def upload_files(
    files: List[UploadFile] = File(...),
    user: dict = Depends(authenticate_user)
):
    """Upload files to user's data folder"""
    username = user["username"]
    data_folder = os.path.join(user_manager.get_user_folder(username), "data")
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        # Validate file extension
        if not file.filename.lower().endswith((".txt", ".jpg", ".jpeg", ".png")):
            continue
            
        file_path = os.path.join(data_folder, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        uploaded_files.append(file.filename)
    
    return {"uploaded_files": uploaded_files}

@app.get("/files", tags=["Data"])
async def list_files(user: dict = Depends(authenticate_user)):
    """List files in user's data folder"""
    username = user["username"]
    data_folder = os.path.join(user_manager.get_user_folder(username), "data")
    
    if not os.path.exists(data_folder):
        return {"files": []}
    
    files = [f for f in os.listdir(data_folder) 
            if f.lower().endswith((".txt", ".jpg", ".jpeg", ".png"))]
    
    return {"files": files}

@app.post("/indexes", status_code=201, tags=["Indexes"])
async def create_index(
    index_create: IndexCreate,
    user: dict = Depends(authenticate_user)
):
    """Build and save an index"""
    username = user["username"]
    password = user["password"]
    index_name = index_create.index_name
    
    user_folder = user_manager.get_user_folder(username)
    data_folder = os.path.join(user_folder, "data")
    indexes_folder = os.path.join(user_folder, "indexes")
    
    if not os.path.exists(indexes_folder):
        os.makedirs(indexes_folder, exist_ok=True)
    
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.lower().endswith((".txt", ".jpg", ".jpeg", ".png"))]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files found in data folder")
    
    try:
        encryptor.build_index_from_files(files, password)
        index_path = os.path.join(indexes_folder, f"{index_name}.pkl")
        encryptor.save_index(index_path, password)
        
        user_manager.add_user_index(username, index_name)
        return {"message": f"Index '{index_name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(e)}")

@app.get("/indexes", tags=["Indexes"])
async def list_indexes(user: dict = Depends(authenticate_user)):
    """List user's indexes"""
    username = user["username"]
    indexes = user_manager.get_user_indexes(username)
    return {"indexes": indexes}

@app.post("/search/text", tags=["Search"])
async def search_text(
    search_query: SearchQuery,
    index_name: str,
    user: dict = Depends(authenticate_user)
):
    """Search by text query"""
    username = user["username"]
    password = user["password"]
    
    indexes = user_manager.get_user_indexes(username)
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    
    indexes_folder = os.path.join(user_manager.get_user_folder(username), "indexes")
    index_path = os.path.join(indexes_folder, f"{index_name}.pkl")
    
    try:
        encryptor.load_index(index_path, password)
        results = encryptor.query_text(search_query.query, password, k=search_query.top_k)
        
        return {
            "results": [
                {"filename": os.path.basename(ref), "score": float(score)}
                for ref, score in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/image", tags=["Search"])
async def search_image(
    image: UploadFile = File(...),
    index_name: str = Form(...),
    top_k: int = Form(5),
    user: dict = Depends(authenticate_user)
):
    """Search by image upload"""
    username = user["username"]
    password = user["password"]
    
    indexes = user_manager.get_user_indexes(username)
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    
    indexes_folder = os.path.join(user_manager.get_user_folder(username), "indexes")
    index_path = os.path.join(indexes_folder, f"{index_name}.pkl")
    
    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp_path = temp.name
        content = await image.read()
        temp.write(content)
    
    try:
        encryptor.load_index(index_path, password)
        results = encryptor.query_image(temp_path, password, k=top_k)
        
        return {
            "results": [
                {"filename": os.path.basename(ref), "score": float(score)}
                for ref, score in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search/image/base64", tags=["Search"])
async def search_image_base64(
    index_name: str,
    body: dict,
    top_k: int = 5,
    user: dict = Depends(authenticate_user)
):
    """Search by image using base64-encoded image data"""
    username = user["username"]
    password = user["password"]
    
    # Get the base64 image from the request body
    image_base64 = body.get("image_base64")
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 field is required in the request body")
    
    indexes = user_manager.get_user_indexes(username)
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    
    indexes_folder = os.path.join(user_manager.get_user_folder(username), "indexes")
    index_path = os.path.join(indexes_folder, f"{index_name}.pkl")
    
    # Decode base64 and save to temp file
    try:
        image_data = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp_path = temp.name
        temp.write(image_data)
    
    try:
        encryptor.load_index(index_path, password)
        results = encryptor.query_image(temp_path, password, k=top_k)
        
        return {
            "results": [
                {"filename": os.path.basename(ref), "score": float(score)}
                for ref, score in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Run server directly when script is executed
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("user_data", exist_ok=True)
    
    # Set environment variable to fix OpenMP error
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Run server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)