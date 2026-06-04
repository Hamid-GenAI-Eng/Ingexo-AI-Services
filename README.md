# Ingexo AI Ingestion Service

A high-performance dedicated **Python/FastAPI** microservice handling:
1. **Multimodal OCR**: Gemini 1.5 Flash layout-preserved page reads.
2. **Translation**: Dynamic language detection and translations.
3. **Structured Entity Extraction**: Enforces strict Pydantic JSON schemas.
4. **Local CPU Vector Embeddings**: Local text chunking and `sentence-transformers` vectorization.
5. **Pinecone Indexing**: Serverless metadata upserting and RLS querying.

---

## 🛠️ Getting Started

### 1. Setup Virtual Environment
Run from the `ai-service/` root directory to create a virtual environment and isolate python packages:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
Install all required machine learning and server packages:
```bash
pip install -r requirements.txt
```

### 3. Environment variables Setup
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your **Google Studio Gemini Key** (`GEMINI_API_KEY`) and **Pinecone Database API Key** (`PINECONE_API_KEY`). Ensure the Pinecone index name matches your active dashboard.

### 4. Running the Service
Start the FastAPI server in development mode (with auto-reloads):
```bash
python -m uvicorn app.main:app --port 8000 --reload
```
Once active, you can visit the automatic interactive API Swagger documentation at: **`http://127.0.0.1:8000/docs`**.

---

## 🔌 API Documentation Reference

### 1. Ingestion Pipeline: `POST /process`
Triggered asynchronously by the Express server on upload.
- **Payload**:
  ```json
  {
    "document_id": "mongo_document_id_string",
    "file_url": "https://res.cloudinary.com/...",
    "ocr": true,
    "translate": false,
    "target_lang": "en",
    "extract": true
  }
  ```
- **Response**: returns `202 Accepted` immediately. Processing continues in a background thread, culminating in a webhook post back to the Express server at `http://localhost:5000/api/documents/webhook-callback`.

### 2. Vector Query Search: `POST /query`
Queried by the Express backend during RAG Chat.
- **Payload**:
  ```json
  {
    "query_text": "What was the total invoice amount?",
    "allowed_document_ids": ["doc_123", "doc_456"],
    "top_k": 5
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "matches": [
      {
        "score": 0.85,
        "docName": "invoice.pdf",
        "documentId": "doc_123",
        "pageNumber": 1,
        "text": "Total Due: $24,612.40"
      }
    ]
  }
  ```

### 3. Clear Vector Data: `DELETE /delete/{doc_id}`
Triggers Pinecone vector deletion on document purge.
- **URL Param**: `doc_id`
- **Response**: `{"status": "success"}` on deletion of all matching vectors.
