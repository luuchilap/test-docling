"""
FastAPI main application with PDF upload and RAG query endpoints
"""
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from app.milvus_client import connect_milvus, get_or_create_collection, insert_chunks, inspect_vectors
from app.pdf_utils import extract_text_from_document, chunk_text, is_supported_file, get_file_type, SUPPORTED_EXTENSIONS
from app.rag import (
    generate_embeddings,
    retrieve_context,
    generate_answer,
    calculate_cosine_similarity,
    query_embedding,
)
from app.database import init_database, save_file_metadata, get_file_metadata, list_all_files, get_file_statistics

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

# Connect to Milvus and initialize database on startup
@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_database()
    
    # Connect to Milvus
    try:
        connect_milvus()
        get_or_create_collection()
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Milvus: {e}")
        print("  Make sure Milvus is running: docker-compose up -d")


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
    
    # Generate unique file_id (ensure it's safe for Milvus queries)
    # Use only alphanumeric characters, underscores, and hyphens
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    file_id = f"file_{timestamp}_{unique_id}"
    
    # Validate file_id doesn't contain problematic characters
    # Current format should be safe, but add validation as safeguard
    if any(char in file_id for char in ["'", '"', '\\', '\n', '\r', '\t']):
        # Sanitize file_id if needed (shouldn't happen with current format)
        file_id = file_id.replace("'", "").replace('"', "").replace('\\', "").replace('\n', "").replace('\r', "").replace('\t', "")
    
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
            
            # Additional validation before insertion
            print(f"  🔍 Pre-insertion validation:")
            print(f"     - File ID: '{file_id}' (length: {len(file_id)})")
            print(f"     - Number of chunks: {len(chunks)}")
            print(f"     - Number of embeddings: {len(embeddings)}")
            if chunks:
                print(f"     - First chunk length: {len(chunks[0])}")
            if embeddings:
                print(f"     - First embedding dimension: {len(embeddings[0])}")
            
            vector_ids = insert_chunks(collection, file_id, chunks, embeddings)
        except ValueError as ve:
            # Validation errors
            error_msg = str(ve)
            print(f"✗ Validation error: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Data validation error: {error_msg}"
            )
        except Exception as milvus_error:
            error_msg = str(milvus_error)
            error_type = str(type(milvus_error))
            
            print(f"✗ Milvus error details:")
            print(f"   - Type: {error_type}")
            print(f"   - Message: {error_msg}")
            print(f"   - File ID: '{file_id}'")
            
            # Handle connection errors
            if "should create connect first" in error_msg or "ConnectionNotExistException" in error_type:
                raise HTTPException(
                    status_code=503,
                    detail="Milvus connection not available. Please ensure Milvus is running: docker-compose up -d"
                )
            
            # Handle pattern matching errors (usually from query expressions or data format)
            if "string did not match the expected pattern" in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail=f"Milvus data format error. This may indicate: 1) Invalid file_id format, 2) Data type mismatch, 3) Schema mismatch. Try dropping and recreating the collection. Error: {error_msg}"
                )
            
            # Handle other Milvus errors
            raise HTTPException(
                status_code=500,
                detail=f"Error inserting data into Milvus: {error_msg}. Check console logs for details."
            )
        
        # Save file metadata to database
        file_size = os.path.getsize(tmp_file_path)
        save_file_metadata(
            file_id=file_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            chunks_count=len(chunks),
            vectors_count=len(vector_ids),
            status="completed"
        )
        
        print()
        print(f"{'='*60}")
        print(f"✓ UPLOAD COMPLETE: {file_id}")
        print(f"{'='*60}")
        print()
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_type,
            "file_size": file_size,
            "chunks_created": len(chunks),
            "vector_ids_count": len(vector_ids),
            "uploaded_at": datetime.now().isoformat()
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
async def query_document(
    request: QueryRequest,
    show_similarity: bool = False,
    show_query_embedding: bool = False,
):
    """
    Query a document using RAG pipeline
    
    Parameters:
    - show_similarity: If True, returns cosine similarity scores for each chunk
    """
    try:
        # Validate file_id
        if not request.file_id or not request.file_id.strip():
            raise HTTPException(
                status_code=400,
                detail="file_id cannot be empty"
            )
        
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="query cannot be empty"
            )
        
        print(f"\n{'='*60}")
        print(f"🔍 QUERY PROCESSING")
        print(f"{'='*60}")
        print(f"File ID: {request.file_id}")
        print(f"Query: {request.query}")
        print(f"Show Similarity: {show_similarity}")
        print(f"Show Query Embedding: {show_query_embedding}")
        print()
        
        # Optionally compute and expose the query embedding
        query_emb = None
        if show_query_embedding:
            try:
                query_emb = query_embedding(request.query)
            except Exception as embed_error:
                error_msg = str(embed_error)
                print(f"✗ Error generating query embedding for inspection: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating query embedding: {error_msg}",
                )
        
        # Retrieve relevant context with similarity scores if requested
        try:
            if show_similarity:
                context_chunks, similarity_info = retrieve_context(
                    request.file_id,
                    request.query,
                    top_k=5,
                    return_similarities=True,
                    query_embedding_override=query_emb,
                )
            else:
                context_chunks = retrieve_context(
                    request.file_id,
                    request.query,
                    top_k=5,
                    query_embedding_override=query_emb,
                )
                similarity_info = None
        except Exception as search_error:
            error_msg = str(search_error)
            error_type = str(type(search_error))
            
            # Handle pattern matching errors
            if "string did not match the expected pattern" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file_id format or Milvus query error. File ID: '{request.file_id}'. Error: {error_msg}"
                )
            
            # Handle connection errors
            if "should create connect first" in error_msg or "ConnectionNotExistException" in error_type:
                raise HTTPException(
                    status_code=503,
                    detail="Milvus connection not available. Please ensure Milvus is running: docker-compose up -d"
                )
            
            # Re-raise other errors
            raise
        
        if not context_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for file_id: {request.file_id}. The file may not have been uploaded or processed correctly."
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
            "query": request.query,
        }
        
        # Add similarity information if requested
        if show_similarity and similarity_info:
            response["similarity_scores"] = similarity_info
            response["similarity_explanation"] = {
                "description": "Cosine similarity measures how similar the query embedding is to each chunk embedding",
                "formula": "cosine_similarity = (A · B) / (||A|| × ||B||)",
                "range": "Values range from -1 (opposite) to 1 (identical), with 0 meaning orthogonal",
                "interpretation": "Higher values (closer to 1) indicate more similar content",
            }
        
        # Add query embedding if requested
        if query_emb is not None:
            response["query_embedding"] = {
                "dimension": len(query_emb),
                "sample_values": query_emb[:10],
                "vector": query_emb,
            }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        error_type = str(type(e))
        print(f"✗ Error processing query: {error_type} - {error_msg}")
        
        # Handle pattern matching errors
        if "string did not match the expected pattern" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query format or Milvus error. Error: {error_msg}"
            )
        
        # Handle connection errors
        if "should create connect first" in error_msg or "ConnectionNotExistException" in error_type:
            raise HTTPException(
                status_code=503,
                detail="Milvus connection not available. Please ensure Milvus is running: docker-compose up -d"
            )
        
        raise HTTPException(status_code=500, detail=f"Error processing query: {error_msg}")


@app.get("/files")
async def list_files(limit: int = 50):
    """
    List all uploaded files with metadata
    """
    try:
        files = list_all_files(limit=limit)
        statistics = get_file_statistics()
        
        return {
            "files": files,
            "statistics": statistics,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """
    Get metadata for a specific file
    """
    try:
        file_info = get_file_metadata(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        return file_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")


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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML UI"""
    # Get the absolute path to the static directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(current_dir, "static", "index.html")
    
    if os.path.exists(html_path):
        # Read the HTML file and return it as HTMLResponse
        # This ensures the browser displays it instead of downloading
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    
    # Fallback if file doesn't exist
    return HTMLResponse(
        content=f"""
        <html>
        <head><title>PDF RAG with Milvus API</title></head>
        <body>
            <h1>PDF RAG with Milvus API</h1>
            <p>HTML file not found at {html_path}</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li>POST /upload-pdf - Upload documents</li>
                <li>POST /query - Query documents</li>
                <li>GET /inspect-vectors - Inspect stored vectors</li>
                <li>GET /docs - API documentation</li>
            </ul>
        </body>
        </html>
        """
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

