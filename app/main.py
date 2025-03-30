from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.embedding_service import CLIPEmbeddingService
from services.vector_store import VectorStore
from models import TextInput, SearchQuery, SearchResult
from typing import List
import io
from PIL import Image
from models import VectorEntry

app = FastAPI(
    title="CLIP Search API",
    description="API for semantic search using CLIP embeddings",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = CLIPEmbeddingService()
vector_store = VectorStore()


@app.post("/add/text", response_model=dict)
async def add_text(text_input: TextInput):
    """Add text to the vector store."""
    try:
        # Generate embedding
        embedding = embedding_service.get_text_embedding(text_input.text)

        # Create vector entry
        entry = VectorEntry(
            text=text_input.text, embedding=embedding, metadata=text_input.metadata
        )

        # Add to vector store
        vector_store.add_entry(entry)

        return {"status": "success", "message": "Text added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add/image", response_model=dict)
async def add_image(image: UploadFile = File(...), text: str = None):
    """Add image to the vector store."""
    try:
        # Read and validate image
        contents = await image.read()
        try:
            img = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Generate embedding
        embedding = embedding_service.get_image_embedding(img)

        # Create vector entry
        entry = VectorEntry(
            text=text or image.filename,
            embedding=embedding,
            metadata={"filename": image.filename, "content_type": image.content_type},
        )

        # Add to vector store
        vector_store.add_entry(entry)

        return {"status": "success", "message": "Image added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """Search for similar items using text query."""
    try:
        # Generate query embedding
        query_embedding = embedding_service.get_text_embedding(query.query)

        # Search vector store
        results = vector_store.search(query_embedding, query.top_k)

        # Format results
        search_results = []
        for entry, score in results:
            search_results.append(
                SearchResult(
                    text=entry.text, similarity_score=score, metadata=entry.metadata
                )
            )

        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store."""
    return {"total_items": len(vector_store), "dimension": vector_store.dimension}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
