# Secure CLIP API Documentation

This document provides information on how to use the REST API for the Secure CLIP semantic search application.

## Base URL

```
http://localhost:8000
```

## Authentication

All endpoints (except user creation) require Basic Authentication.

### Example:

```bash
curl -X GET "http://localhost:8000/users/me" \
     -H "accept: application/json" \
     -u "username:password"
```

## Endpoints

### User Management

#### Create User

```
POST /users
```

Create a new user.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "message": "User created successfully"
}
```

#### Get Current User

```
GET /users/me
```

Get information about the currently authenticated user.

**Response:**
```json
{
  "username": "string",
  "indexes": ["index1", "index2"]
}
```

### Data Management

#### Upload Files

```
POST /upload
```

Upload files to user's data folder.

**Request:**
- Form data with files

**Response:**
```json
{
  "uploaded_files": ["file1.jpg", "file2.txt"]
}
```

#### List Files

```
GET /files
```

List files in user's data folder.

**Response:**
```json
{
  "files": ["file1.jpg", "file2.txt"]
}
```

### Index Management

#### Create Index

```
POST /indexes
```

Build and save an index.

**Request Body:**
```json
{
  "index_name": "string"
}
```

**Response:**
```json
{
  "message": "Index 'myindex' created successfully"
}
```

#### List Indexes

```
GET /indexes
```

List user's indexes.

**Response:**
```json
{
  "indexes": ["index1", "index2"]
}
```

### Search

#### Text Search

```
POST /search/text
```

Search by text query.

**Request:**
- Query parameter:
  - `index_name`: Name of the index to search
- JSON body:
  ```json
  {
    "query": "string",
    "top_k": 5
  }
  ```

**Response:**
```json
{
  "results": [
    {
      "filename": "string",
      "score": 0.0
    }
  ]
}
```

#### Image Search (Upload)

```
POST /search/image
```

Search by image upload.

**Request:**
- Form data:
  - `image`: Image file
  - `index_name`: Name of the index to search
  - `top_k`: Number of results (default: 5)

**Response:**
```json
{
  "results": [
    {
      "filename": "string",
      "score": 0.0
    }
  ]
}
```

#### Image Search (Base64)

```
POST /search/image/base64
```

Search by image using base64-encoded image data.

**Request:**
- Query parameters:
  - `index_name`: Name of the index to search
  - `top_k`: Number of results (default: 5)
- JSON body:
  ```json
  {
    "image_base64": "string"
  }
  ```

**Response:**
```json
{
  "results": [
    {
      "filename": "string",
      "score": 0.0
    }
  ]
}
```

## Example Usage

### Create User

```bash
curl -X POST "http://localhost:8000/users" \
     -H "Content-Type: application/json" \
     -d '{"username":"testuser","password":"testpass"}'
```

### Upload Files

```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -u "testuser:testpass" \
     -F "files=@/path/to/image.jpg" \
     -F "files=@/path/to/document.txt"
```

### Create Index

```bash
curl -X POST "http://localhost:8000/indexes" \
     -H "Content-Type: application/json" \
     -u "testuser:testpass" \
     -d '{"index_name":"myindex"}'
```

### Text Search

```bash
curl -X POST "http://localhost:8000/search/text?index_name=myindex" \
     -H "Content-Type: application/json" \
     -u "testuser:testpass" \
     -d '{"query":"mountains and nature","top_k":5}'
```

### Image Search (Upload)

```bash
curl -X POST "http://localhost:8000/search/image" \
     -H "accept: application/json" \
     -u "testuser:testpass" \
     -F "image=@/path/to/query_image.jpg" \
     -F "index_name=myindex" \
     -F "top_k=5"
```

### Image Search (Base64)

```bash
curl -X POST "http://localhost:8000/search/image/base64?index_name=myindex&top_k=5" \
     -H "Content-Type: application/json" \
     -u "testuser:testpass" \
     -d '{"image_base64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}'
```

## Interactive API Documentation

When the API is running, you can access the interactive Swagger UI documentation at:

```
http://localhost:8000/docs
```

This provides a web interface to explore and test all API endpoints.

## Running the API

To start the API server:

```bash
python api.py
```

This will start the server on port 8000.