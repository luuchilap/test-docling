"""
FastAPI main application with PDF upload and RAG query endpoints
"""
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from app.milvus_client import connect_milvus, get_or_create_collection, insert_chunks, inspect_vectors
from app.pdf_utils import extract_text_from_document, chunk_text, is_supported_file, get_file_type, SUPPORTED_EXTENSIONS
from app.rag import generate_embeddings, retrieve_context, generate_answer, calculate_cosine_similarity

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG with Milvus",
    description="MVP for PDF upload, embedding storage, and RAG queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Connect to Milvus on startup
@app.on_event("startup")
async def startup_event():
    try:
        connect_milvus()
        get_or_create_collection()
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Milvus: {e}")
        print("  Make sure Milvus is running: docker run -p 19530:19530 milvusdb/milvus:latest")


# Request/Response models
class QueryRequest(BaseModel):
    file_id: str
    query: str


class QueryResponse(BaseModel):
    answer: str
    file_id: str
    query: str


# Endpoints
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a document file (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images), 
    extract text, generate embeddings, and store in Milvus
    
    Supported formats: PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, PNG, JPEG, TIFF, BMP, WEBP
    """
    # Check if file format is supported
    if not is_supported_file(file.filename):
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {supported}"
        )
    
    # Get file extension for temp file
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_type = get_file_type(file.filename)
    
    # Generate unique file_id
    file_id = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Save uploaded file temporarily with original extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Extract text from document
        print(f"\n{'='*60}")
        print(f"📄 UPLOAD PROCESSING: {file.filename}")
        print(f"{'='*60}")
        print(f"File ID: {file_id}")
        print(f"File Type: {file_type}")
        print()
        
        text = extract_text_from_document(tmp_file_path, file.filename)
        
        if not text.strip():
            raise HTTPException(
                status_code=400, 
                detail=f"{file_type} contains no extractable text"
            )
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail=f"No chunks created from {file_type}"
            )
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Insert into Milvus
        try:
            collection = get_or_create_collection()
            vector_ids = insert_chunks(collection, file_id, chunks, embeddings)
        except Exception as milvus_error:
            error_msg = str(milvus_error)
            if "should create connect first" in error_msg or "ConnectionNotExistException" in str(type(milvus_error)):
                raise HTTPException(
                    status_code=503,
                    detail="Milvus connection not available. Please ensure Milvus is running: docker run -p 19530:19530 milvusdb/milvus:latest"
                )
            raise
        
        print()
        print(f"{'='*60}")
        print(f"✓ UPLOAD COMPLETE: {file_id}")
        print(f"{'='*60}")
        print()
        
        return {
            "file_id": file_id,
            "chunks_created": len(chunks),
            "vector_ids_count": len(vector_ids)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error processing document: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post("/query")
async def query_document(request: QueryRequest, show_similarity: bool = False):
    """
    Query a document using RAG pipeline
    
    Parameters:
    - show_similarity: If True, returns cosine similarity scores for each chunk
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔍 QUERY PROCESSING")
        print(f"{'='*60}")
        print(f"File ID: {request.file_id}")
        print(f"Query: {request.query}")
        print(f"Show Similarity: {show_similarity}")
        print()
        
        # Retrieve relevant context with similarity scores if requested
        if show_similarity:
            context_chunks, similarity_info = retrieve_context(
                request.file_id, 
                request.query, 
                top_k=5, 
                return_similarities=True
            )
        else:
            context_chunks = retrieve_context(request.file_id, request.query, top_k=5)
            similarity_info = None
        
        if not context_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for file_id: {request.file_id}"
            )
        
        # Generate answer
        answer = generate_answer(request.query, context_chunks)
        
        print()
        print(f"{'='*60}")
        print(f"✓ QUERY COMPLETE")
        print(f"{'='*60}")
        print()
        
        response = {
            "answer": answer,
            "file_id": request.file_id,
            "query": request.query
        }
        
        # Add similarity information if requested
        if show_similarity and similarity_info:
            response["similarity_scores"] = similarity_info
            response["similarity_explanation"] = {
                "description": "Cosine similarity measures how similar the query embedding is to each chunk embedding",
                "formula": "cosine_similarity = (A · B) / (||A|| × ||B||)",
                "range": "Values range from -1 (opposite) to 1 (identical), with 0 meaning orthogonal",
                "interpretation": "Higher values (closer to 1) indicate more similar content"
            }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/inspect-vectors")
async def inspect_stored_vectors(
    file_id: Optional[str] = None, 
    limit: int = 10,
    full_content: bool = False,
    show_vectors: bool = False
):
    """
    Inspect stored vectors for educational purposes
    
    Parameters:
    - file_id: Filter by specific file ID (optional)
    - limit: Maximum number of vectors to return
    - full_content: If True, returns full chunk text; if False, returns preview only
    - show_vectors: If True, returns the actual embedding vector values (all 1536 numbers)
    """
    # Normalize file_id - convert empty string to None
    if file_id is not None and not file_id.strip():
        file_id = None
    
    try:
        collection = get_or_create_collection()
        results = inspect_vectors(
            collection, 
            file_id=file_id, 
            limit=limit, 
            full_content=full_content,
            show_vectors=show_vectors
        )
        
        return {
            "count": len(results),
            "full_content": full_content,
            "show_vectors": show_vectors,
            "vectors": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inspecting vectors: {str(e)}")


@app.get("/")
async def root():
    """Serve the HTML UI"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "message": "PDF RAG with Milvus API",
        "endpoints": {
            "upload": "POST /upload-pdf (supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images)",
            "query": "POST /query",
            "inspect": "GET /inspect-vectors?file_id=<file_id>&limit=10",
            "docs": "GET /docs",
            "ui": "GET / (HTML interface)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

