import sys
import os
from dotenv import load_dotenv

# Load local env configuration
load_dotenv()

# Verify Python can load dependencies and connect successfully
print("[TEST] Initializing dependencies...")

try:
    import google.generativeai as genai
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    print("[TEST] Dependencies loaded successfully!")
except Exception as e:
    print(f"[TEST] Dependency load failed: {e}")
    sys.exit(1)

# Step 1: Test local embedding model loading
print("[TEST] Loading local SentenceTransformer model...")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed = model.encode("Ingexo AI testing script")
    print(f"[TEST] Local Vector Embedding generated successfully! Dimension: {len(embed)}")
except Exception as e:
    print(f"[TEST] Local Embedding generation failed: {e}")

# Step 2: Test Pinecone Connection
print("[TEST] Connecting to Pinecone...")
try:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "document-embeddings")
    pc = Pinecone(api_key=api_key)
    # Check if index exists or list indexes
    indexes = [idx.name for idx in pc.list_indexes()]
    print(f"[TEST] Connected to Pinecone! Indexes in your account: {indexes}")
    
    if index_name in indexes:
        index = pc.Index(index_name)
        desc = pc.describe_index(index_name)
        print(f"[TEST] Successfully connected to and described active Pinecone index: '{index_name}'!")
        print(f"       Index Info: Dimension: {desc.dimension}, Metric: {desc.metric}, Status: {desc.status['state']}")
    else:
        print(f"[TEST] WARNING: Index '{index_name}' was not found in your list of indexes.")
except Exception as e:
    print(f"[TEST] Pinecone connection failed: {e}")

# Step 3: Test Gemini 1.5 Flash API Key
print("[TEST] Calling Google Gemini API...")
try:
    gemini_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)
    
    print("[TEST] Available models in your key:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"       - {m.name}")
    except Exception as list_err:
        print(f"       Failed to list models: {list_err}")

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say: Ingexo AI System is working perfectly!")
    print(f"[TEST] Google Gemini API Key verified successfully! Response: '{response.text.strip()}'")
except Exception as e:
    print(f"[TEST] Gemini API call failed: {e}")

print("\n[TEST] Integration Verification Complete!")
