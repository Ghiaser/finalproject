# CLIP Search API

A FastAPI application that provides semantic search capabilities using OpenAI's CLIP model for both text and images.

## Features

- Text-to-text semantic search
- Image-to-text semantic search
- Vector similarity search using FAISS
- Support for metadata storage
- RESTful API interface

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8002`

## API Endpoints

### Add Text

```bash
POST /add/text
{
    "text": "your text here",
    "metadata": {"optional": "metadata"}
}
```

### Add Image

```bash
POST /add/image
Form data:
- image: file
- text: optional description
```

### Search

```bash
POST /search
{
    "query": "your search query",
    "top_k": 5
}
```

### Stats

```bash
GET /stats
```

## Technology Stack

### Core Technologies

- **FastAPI**: Modern, fast web framework for building APIs with Python. Used for creating the RESTful API endpoints and handling HTTP requests.
- **CLIP (Contrastive Language-Image Pre-training)**: OpenAI's neural network model that can understand both images and text. Used for generating embeddings from both text and images for semantic search.
- **FAISS (Facebook AI Similarity Search)**: Efficient similarity search library developed by Facebook. Used for storing and searching high-dimensional vectors quickly.

### Supporting Technologies

- **PyTorch**: Deep learning framework used by CLIP for model operations and GPU acceleration.
- **Transformers (Hugging Face)**: Library providing the CLIP model implementation and preprocessing utilities.
- **Pillow (PIL)**: Python Imaging Library for handling image processing tasks.
- **NumPy**: Library for numerical computing, used for vector operations and normalization.
- **Uvicorn**: ASGI server implementation, used to run the FastAPI application.

### Development Tools

- **Python 3.10+**: Main programming language used for the application.
- **pip**: Package installer for Python, used for managing project dependencies.
- **venv**: Python's virtual environment system for isolating project dependencies.

### Storage

- **File System**: Used for persisting vector store data and FAISS indices.
- **Pickle**: Python's native serialization format, used for storing vector entries and metadata.

## API Documentation

Once the server is running, you can access:

- Interactive API documentation at `http://localhost:8000/docs`
- Alternative API documentation at `http://localhost:8000/redoc`
