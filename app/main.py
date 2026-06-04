from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
import uvicorn
from app.config import PORT, HOST, WEBHOOK_URL, GEMINI_MODEL
from app.pipeline import process_document_pipeline, generate_rag_response
from app.vectorizer import chunk_text
from app.pinecone_db import upsert_document_chunks, query_pinecone_index, delete_document_vectors

app = FastAPI(
    title="Ingexo AI Dedicated Ingestion Service",
    description="Python microservice executing Gemini Multimodal OCR and CPU vector indexing.",
    version="1.0.0"
)

# 1. Define Request Schemas
class IngestRequest(BaseModel):
    document_id: str = Field(description="Database ObjectId of Document record")
    file_url: str = Field(description="Cloudinary secure URL of PDF/Image")
    ocr: bool = Field(default=True, description="Enable layout OCR")
    translate: bool = Field(default=False, description="Translate to English")
    target_lang: str = Field(default="en", description="Target ISO language code")
    extract: bool = Field(default=True, description="Enable structured extraction")

class QueryRequest(BaseModel):
    query_text: str = Field(description="Semantic chat question")
    allowed_document_ids: List[str] = Field(default=[], description="Workspace RLS document boundaries")
    top_k: int = Field(default=5, description="Number of segments to retrieve")

class ChatMessage(BaseModel):
    role: str = Field(description="Sender role: user or assistant")
    text: str = Field(description="Content of the message")

class ChatRequest(BaseModel):
    query_text: str = Field(description="Semantic user prompt")
    allowed_document_ids: List[str] = Field(default=[], description="Workspace RLS document boundaries")
    history: List[ChatMessage] = Field(default=[], description="Previous conversation history")
    top_k: int = Field(default=5, description="Number of segments to retrieve for RAG context")


# 2. Async Background Worker Routine
def execute_background_ingest(task: IngestRequest):
    """
    Asynchronous background job downloading files, extracting text
    and structured maps, computing embeddings, indexing Pinecone, and web-hooking.
    """
    print(f"\n🚀 Starting background processing for Document: {task.document_id}")
    try:
        # Step 1: Run OCR, translation, and structured data extraction
        extraction = process_document_pipeline(
            file_url=task.file_url,
            ocr=task.ocr,
            translate=task.translate,
            target_lang=task.target_lang,
            extract=task.extract
        )

        # Step 2: Slice layout text into paragraph segments
        layout_text = extraction["translated_text"] or extraction["original_text"]
        text_chunks = chunk_text(layout_text)

        # Step 3: Compute embeddings locally and index Pinecone
        if text_chunks:
            # We use the document_id as the metadata reference matching our database
            doc_name = task.file_url.split("/")[-1].split("?")[0] # extract clean filename
            upsert_document_chunks(
                document_id=task.document_id,
                doc_name=doc_name,
                chunks=text_chunks
            )

        # Step 4: Construct Webhook response and notify Express Backend
        webhook_payload = {
            "document_id": task.document_id,
            "status": "success",
            "original_text": extraction["original_text"],
            "translated_text": extraction["translated_text"],
            "language_detected": extraction["language_detected"],
            "summary": extraction["summary"],
            "extracted_fields": extraction["extracted_fields"],
            "error": None
        }

        print(f"📡 Sending webhook callback payload to: {WEBHOOK_URL}")
        res = requests.post(WEBHOOK_URL, json=webhook_payload, timeout=20)
        res.raise_for_status()
        print(f"✅ Background Ingestion Successfully finalized for Doc: {task.document_id}\n")

    except Exception as e:
        print(f"🔥 Background pipeline crashed for Doc: {task.document_id}: {e}")
        # Notify Express backend of processing failure so user is alerted via DLQ retry
        webhook_payload = {
            "document_id": task.document_id,
            "status": "failed",
            "original_text": None,
            "translated_text": None,
            "language_detected": None,
            "summary": None,
            "extracted_fields": None,
            "error": f"AI Microservice failed: {str(e)}"
        }
        try:
            requests.post(WEBHOOK_URL, json=webhook_payload, timeout=20)
        except Exception as err:
            print(f"⚠️ Webhook fail reporting collapsed: {err}")

# 3. HTTP Endpoints
@app.get("/")
def get_service_status():
    return {
        "status": "healthy",
        "service": "Ingexo AI Ingestion Service",
        "port": PORT,
        "engine": "FastAPI (Python)",
        "model": GEMINI_MODEL
    }


@app.post("/process", status_code=status.HTTP_202_ACCEPTED)
def trigger_ingest_pipeline(task: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger the async ingestion pipeline (returns 202 immediately, processes in background)
    """
    background_tasks.add_task(execute_background_ingest, task)
    return {
        "status": "accepted",
        "message": "Ingestion job queued successfully",
        "document_id": task.document_id
    }

@app.post("/query")
def run_semantic_search(query: QueryRequest):
    """
    Search vector index matching semantic queries, applying security document-ID limits
    """
    try:
        matches = query_pinecone_index(
            query_text=query.query_text,
            allowed_document_ids=query.allowed_document_ids,
            top_k=query.top_k
        )
        return {
            "status": "success",
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pinecone Vector query failed: {str(e)}"
        )

@app.post("/chat")
def run_rag_chat(chat_req: ChatRequest):
    """
    Run semantic Pinecone search, retrieve top chunks, and generate a RAG response with citations.
    """
    try:
        # Step 1: Query Pinecone for context chunks
        matches = query_pinecone_index(
            query_text=chat_req.query_text,
            allowed_document_ids=chat_req.allowed_document_ids,
            top_k=chat_req.top_k
        )
        
        # Step 2: Format history list for pipeline
        formatted_history = []
        for h in chat_req.history:
            formatted_history.append({
                "role": h.role,
                "text": h.text
            })
            
        # Step 3: Call Gemini RAG Generator
        answer = generate_rag_response(
            query_text=chat_req.query_text,
            context_chunks=matches,
            history=formatted_history
        )
        
        # Step 4: Extract deduplicated citations matching what was fetched
        citations = []
        seen = set()
        for match in matches:
            quote_snippet = match["text"][:200].strip()
            citation_key = (match["docName"], match["pageNumber"], quote_snippet)
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append({
                    "doc": match["docName"],
                    "page": match["pageNumber"],
                    "quote": quote_snippet
                })

        return {
            "status": "success",
            "text": answer,
            "citations": citations
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG chat processing failed: {str(e)}"
        )

@app.delete("/delete/{doc_id}")
def purge_document_vectors(doc_id: str):
    """
    Remove vectors from Pinecone index matching a specific document ID
    """
    success = delete_document_vectors(doc_id)
    if success:
        return {"status": "success", "message": f"Purged Pinecone vectors for {doc_id}"}
    raise HTTPException(
        status_code=500,
        detail=f"Failed to purge Pinecone vectors for {doc_id}"
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
