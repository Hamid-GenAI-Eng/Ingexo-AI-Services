from pinecone import Pinecone
from app.config import PINECONE_API_KEY, PINECONE_INDEX
from app.vectorizer import generate_embedding
from typing import List, Dict, Any

# Initialize the Pinecone SDK Client
print("[Pinecone] Connecting to Pinecone Vector Database...")
pc = None
index = None
try:
    if PINECONE_API_KEY and "developer_key" not in PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        print(f"[Pinecone] Connected to Pinecone Index: {PINECONE_INDEX}")
    else:
        print("[Pinecone] WARNING: PINECONE_API_KEY is not configured or uses default placeholder. Vector operations will be mocked/unavailable.")
except Exception as e:
    print(f"[Pinecone] WARNING: Failed to connect to Pinecone index: {e}. Vector operations will be mocked/unavailable.")

def upsert_document_chunks(document_id: str, doc_name: str, chunks: List[str]) -> bool:
    """
    Embeds text chunks locally and indexing vectors directly to Pinecone Index.
    """
    if not chunks:
        return False

    vectors = []
    for idx, chunk in enumerate(chunks):
        # 1. Generate embedding locally ($0 CPU cost)
        vector_vals = generate_embedding(chunk)
        
        # 2. Assign unique vector ID
        vector_id = f"{document_id}_c{idx}"
        
        # 3. Compile citations metadata
        metadata = {
            "documentId": document_id,
            "docName": doc_name,
            "pageNumber": 1, # Defaulting first page for simple chunk layouts
            "paragraphIndex": idx,
            "text": chunk
        }
        
        vectors.append({
            "id": vector_id,
            "values": vector_vals,
            "metadata": metadata
        })

    # 4. Batch upsert to Pinecone
    if not index:
        print(f"[Pinecone] ERROR: Cannot upsert vectors for Doc {document_id} - index is not initialized.")
        return False
        
    index.upsert(vectors=vectors)
    print(f"[Pinecone] Successfully indexed {len(vectors)} vectors in Pinecone for Doc: {document_id}")
    return True

def query_pinecone_index(query_text: str, allowed_document_ids: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query Pinecone for matching vector segments, applying role-based ID filters.
    """
    # 1. Embed query locally
    query_vector = generate_embedding(query_text)

    # 2. Build metadata filter payload to enforce security RLS/RBAC access control
    query_options = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }

    if allowed_document_ids:
        query_options["filter"] = {
            "documentId": {"$in": allowed_document_ids}
        }

    # 3. Execute query
    if not index:
        print("[Pinecone] ERROR: Cannot query vectors - index is not initialized.")
        return []
        
    results = index.query(**query_options)
    
    matches = []
    for match in results.get("matches", []):
        matches.append({
            "score": match.score,
            "docName": match.metadata.get("docName", "Unknown"),
            "documentId": match.metadata.get("documentId"),
            "pageNumber": int(match.metadata.get("pageNumber", 1)),
            "text": match.metadata.get("text", "")
        })

    return matches

def delete_document_vectors(document_id: str) -> bool:
    """
    Remove all vectors associated with a deleted document
    """
    try:
        # Delete vectors using metadata filters
        if not index:
            print(f"[Pinecone] ERROR: Cannot delete vectors for Doc {document_id} - index is not initialized.")
            return False
            
        index.delete(filter={"documentId": {"$eq": document_id}})
        print(f"[Pinecone] Cleared all Pinecone vectors matching Document: {document_id}")
        return True
    except Exception as e:
        print(f"[Pinecone] Pinecone vectors delete failed: {e}")
        return False
