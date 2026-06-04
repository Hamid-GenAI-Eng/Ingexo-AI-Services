from sentence_transformers import SentenceTransformer
from typing import List
import re

# Load the local CPU sentence-transformers model (all-MiniLM-L6-v2 is ~120MB)
print("[Vectorizer] Loading local SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Vectorizer] Local SentenceTransformer model loaded successfully.")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Slices document text into paragraph-level segments to preserve citation accuracy.
    """
    if not text:
        return []

    # Clean double spaces or excess carriage returns
    normalized_text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split text into paragraphs
    paragraphs = normalized_text.split('\n\n')
    
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding paragraph exceeds chunk size, save the current chunk
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Append any remaining text
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def generate_embedding(text: str) -> List[float]:
    """
    Computes a 384-dimensional vector embedding locally on CPU
    """
    if not text:
        return [0.0] * 384
        
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
