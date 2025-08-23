# 🔐 Secure CLIP Multi-User Guide

This guide explains how to run and use the multi-user version of the Secure CLIP semantic search application.

## Overview

The multi-user version provides:
- User registration and authentication
- Per-user data storage and indexes
- Web UI interface
- RESTful API access
- Secure password-based encryption

## Components

The application consists of three main components:

1. **Multi-User Web UI** (`multi_user_app.py`)
2. **RESTful API** (`api.py`)
3. **API Tests** (`api_tests.py`)

## Setup

### 1. Requirements

Install the required dependencies:

```bash
pip install streamlit torch transformers faiss-cpu cryptography pillow fastapi uvicorn python-multipart requests
```

### 2. Directory Structure

Ensure the application can create these directories:
- `user_data/` - For storing per-user data and indexes
- `DATA/` - For sample files when testing

## Running the Multi-User Web UI

To run the Streamlit-based web interface:

```bash
streamlit run multi_user_app.py
```

This will start the UI at `http://localhost:8501`

### Usage Instructions

1. **Registration**:
   - Create a new account with username and password
   - Each user gets their own private data space

2. **Data Management**:
   - Upload files through the sidebar
   - View your uploaded files

3. **Index Management**:
   - Create indexes from your uploaded files
   - Load previously created indexes

4. **Search**:
   - Perform text searches against your indexes
   - Upload images to find similar content

## Running the API Server

To run the RESTful API:

```bash
python api.py
```

This will start the API server at `http://localhost:8000`

### API Documentation

- **Interactive Docs**: Visit `http://localhost:8000/docs` for Swagger UI
- **Detailed Documentation**: See `API_DOCS.md` for endpoint information

### Key Endpoints

- `/users` - Create users and get user info
- `/upload` - Upload files
- `/files` - List files 
- `/indexes` - Create and list indexes
- `/search/text` - Text-based semantic search
- `/search/image` - Image upload search
- `/search/image/base64` - Base64 image search

## Running API Tests

To test the API functionality:

```bash
# Make sure the API server is running first
python api.py

# In another terminal, run the tests
python api_tests.py
```

### Test Requirements

For best results, place these files in your `DATA` directory:
- `1.jpg` - For testing image upload search
- `2.jpg` - For testing base64 image search

The tests will still run without these files but will use fallback methods.

### Test Configuration

You can customize the test settings with command-line arguments:

```bash
python api_tests.py --url http://localhost:8000
```

## Security Features

- User passwords are hashed (not stored in plaintext)
- Vectors and file references are encrypted with Fernet
- Each user's data is isolated from others
- Index files are signed to prevent tampering
- Client-side password handling (never sent to server in plaintext)

## Known Issues and Limitations

- The vector encryption adds some performance overhead
- The API currently uses basic authentication (consider adding JWT for production)
- Large files may require tuning the timeout settings

