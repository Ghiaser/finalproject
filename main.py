from fastapi import FastAPI, File, UploadFile
from app.vectorization import encode_text, encode_image
from app.storage import add_vector_to_index, search_vector
from app.encryption import encrypt_vector, decrypt_vector

app = FastAPI()

@app.post("/process-file/")
async def process_file(file: UploadFile):
    # קרא את התוכן של הקובץ
    content = await file.read()
    text = content.decode("utf-8")  # מניחים שזה טקסט

    # שלב 1: המרת טקסט לווקטור
    vector = encode_text(text)

    # שלב 2: הצפנת הווקטור
    encrypted_vector = encrypt_vector(vector)

    # שלב 3: אחסון הווקטור במאגר
    add_vector_to_index(encrypted_vector, vector_id=1)  # השתמש במזהה ייחודי

    return {"message": f"File {file.filename} processed successfully!"}

@app.post("/search/")
async def search_query(query: str):
    # המרת הטקסט לווקטור
    query_vector = encode_text(query)

    # שלב חיפוש
    indices, distances = search_vector(query_vector.numpy(), top_k=5)

    return {"results": {"indices": indices.tolist(), "distances": distances.tolist()}}
