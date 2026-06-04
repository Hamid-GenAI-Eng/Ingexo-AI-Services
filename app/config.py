import os
from dotenv import load_dotenv

# Load the local environment configuration
load_dotenv()

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "127.0.0.1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "document-embeddings")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
WEBHOOK_URL = f"{BACKEND_URL}/api/documents/webhook-callback"

# Validate critical API keys presence
if not GEMINI_API_KEY or "your_google" in GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not configured correctly.")

if not PINECONE_API_KEY or "your_pinecone" in PINECONE_API_KEY:
    print("⚠️ WARNING: PINECONE_API_KEY environment variable is not configured correctly.")
