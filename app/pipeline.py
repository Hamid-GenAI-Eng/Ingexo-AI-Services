import requests
import google.generativeai as genai
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
from app.config import GEMINI_API_KEY, GEMINI_MODEL

# Configure the Google Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

# Define Pydantic Models for strict JSON schema enforcement in Gemini
class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service rendered")
    amount: float = Field(description="Line item amount or cost")

class ExtractedFields(BaseModel):
    documentType: str = Field(description="Document category e.g. Invoice, Contract, Lease, Receipt, Memo, or Report")
    vendorName: str = Field(description="Name of the issuing company, vendor, or party")
    invoiceDate: Optional[str] = Field(None, description="ISO Date of issue (YYYY-MM-DD)")
    dueDate: Optional[str] = Field(None, description="ISO Due date if applicable (YYYY-MM-DD)")
    grandTotal: Optional[float] = Field(None, description="Total monetary amount due or billed")
    currency: Optional[str] = Field(None, description="ISO Currency code e.g. USD, EUR, GBP")
    lineItems: List[LineItem] = Field(default=[], description="List of specific line items or parsed table rows")
    paymentTerms: Optional[str] = Field(None, description="Payment terms e.g. Net 14, Net 30, Due on Receipt")

def download_file_from_url(url: str) -> bytes:
    """Download Cloudinary file directly into in-memory bytes"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def determine_mime_type(url: str) -> str:
    """Determine MIME type based on file extension from Cloudinary URL"""
    ext = url.split(".")[-1].split("?")[0].lower()
    if ext == "pdf":
        return "application/pdf"
    elif ext in ["png", "jpg", "jpeg", "heic", "tiff", "webp"]:
        return f"image/{ext if ext != 'jpg' else 'jpeg'}"
    return "application/octet-stream"

def process_document_pipeline(
    file_url: str,
    ocr: bool = True,
    translate: bool = False,
    target_lang: str = "en",
    extract: bool = True
) -> Dict[str, Any]:
    """
    Downloads document from Cloudinary URL and executes OCR,
    Translations, Structured extractions and Summarizations using Gemini.
    """
    # 1. Download file bytes
    file_bytes = download_file_from_url(file_url)
    mime_type = determine_mime_type(file_url)

    model = genai.GenerativeModel(GEMINI_MODEL)

    # 2. Extract Plain Text (OCR) preserving layout
    original_text = ""
    if ocr:
        print(f"[Gemini] Executing Multimodal OCR for {mime_type}...")
        prompt = (
            "Analyze this document. Extract the complete plain text, fully "
            "preserving layout alignments, lists, spacing, and tabular structures."
        )
        response = model.generate_content([
            {"mime_type": mime_type, "data": file_bytes},
            prompt
        ])
        original_text = response.text

    # 3. Detect and Translate if requested
    translated_text = ""
    language_detected = "en"
    if translate:
        print(f"[Gemini] Running Language Detection & Translation to {target_lang}...")
        prompt = (
            f"Detect the primary language of this document. Translate its complete "
            f"contents into {target_lang}, retaining original paragraph segments."
        )
        response = model.generate_content([
            {"mime_type": mime_type, "data": file_bytes},
            prompt
        ])
        translated_text = response.text
        
        # Simple language detection hook
        lang_prompt = "Identify the language of this document. Respond with ONLY its two-letter ISO language code (e.g. en, de, es, fr, ja)."
        lang_response = model.generate_content([
            {"mime_type": mime_type, "data": file_bytes},
            lang_prompt
        ])
        language_detected = lang_response.text.strip().lower()

    # 4. Enforce Pydantic Structured Data Extraction
    extracted_fields = {}
    if extract:
        print("[Gemini] Running Pydantic Structured Data Extraction...")
        prompt = (
            "Analyze this document and extract key entities matching the requested "
            "JSON schema. Ensure numeric fields are floats/ints and dates are parsed."
        )
        
        # Native structured JSON output configuration in Gemini Python SDK!
        response = model.generate_content(
            [{"mime_type": mime_type, "data": file_bytes}, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFields
            )
        )
        
        # Parse the strictly structured output
        try:
            extracted_fields = json.loads(response.text)
        except Exception as e:
            print(f"[Gemini] Structured parsing failed: {e}. Fallback to raw string.")
            extracted_fields = {"raw_output": response.text}

    # 5. Generate a dynamic Summary
    print("[Gemini] Generating Summary...")
    summary_prompt = (
        "Generate a brief, 3-sentence summary of this document. "
        "Highlight the primary purpose, parties involved, and key metrics."
    )
    summary_response = model.generate_content([
        {"mime_type": mime_type, "data": file_bytes},
        summary_prompt
    ])
    summary = summary_response.text.strip()

    return {
        "original_text": original_text,
        "translated_text": translated_text if translate else original_text,
        "language_detected": language_detected,
        "extracted_fields": extracted_fields,
        "summary": summary
    }

def generate_rag_response(
    query_text: str,
    context_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]]
) -> str:
    """
    Formulates a RAG prompt with context and history, and calls Gemini to generate an answer.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Format the context chunks
    context_str = ""
    if context_chunks:
        context_str = "\n\n".join([
            f"[Source Document: {c['docName']}, Page: {c['pageNumber']}]\n\"{c['text']}\""
            for c in context_chunks
        ])
    else:
        context_str = "No relevant document chunks found in library. (The user has not uploaded matching documents yet)."

    # Format history
    history_str = ""
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role_label}: {msg['text']}\n"

    # Complete system prompt instruction
    system_prompt = (
        "You are Ingexo AI, a premium, enterprise-grade Document Intelligence and Workspace RAG assistant.\n"
        "Your task is to answer the user's question accurately using the provided Document Context and the Conversation History.\n"
        "Follow these rules strictly:\n"
        "1. Ground your answers ONLY in the provided Document Context if possible. If the context does not contain sufficient details to answer, state this clearly while still attempting to be as helpful as possible based on common knowledge, but make sure to note what is missing from the library.\n"
        "2. Keep your answers beautifully structured, readable, and professional. Use bold text (with **bold**) for headings or key numbers. Keep markdown rendering simple.\n"
        "3. NEVER fabricate page numbers or citations. Speak directly and confidently.\n\n"
        f"Document Context:\n{context_str}\n\n"
        f"Conversation History:\n{history_str}\n"
    )

    response = model.generate_content([
        system_prompt,
        f"User Question: {query_text}\nAssistant Response:"
    ])

    return response.text.strip()

