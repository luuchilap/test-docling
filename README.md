# Multi-Format Document RAG with Milvus - MVP

A minimal viable product for document upload (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images), text extraction, embedding generation, and RAG-based querying using Milvus as the vector database.

## 🎯 Features

- **Multi-Format Document Upload**: Upload various document formats (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images) and extract text
- **Embedding Generation**: Generate embeddings using OpenAI's `text-embedding-3-small` model
- **Vector Storage**: Store embeddings in Milvus with HNSW index
- **RAG Querying**: Query documents using vector similarity search and LLM-based answer generation
- **Cosine Similarity**: View cosine similarity scores to understand how queries match document chunks
- **Vector Inspection**: Inspect stored vectors for educational purposes

## 🏗 Tech Stack

- **FastAPI**: HTTP API framework
- **Milvus**: Vector database (standalone)
- **PyMilvus**: Milvus Python SDK
- **OpenAI**: Embeddings and chat completion
- **Docling**: Multi-format document processing (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images)
- **pypdf**: PDF text extraction (fallback)

## 📋 Prerequisites

1. **Python 3.8+**
2. **Docker** (for running Milvus)
3. **OpenAI API Key**

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Start Milvus

**Option A: Using Docker Compose (Recommended)**

Start Milvus with all dependencies (etcd, minio) using docker-compose:

```bash
docker-compose up -d
```

This will start:
- Milvus Standalone on port 19530
- etcd (metadata storage) on port 2379
- MinIO (object storage) on ports 9000 and 9001
- **Attu** (Milvus GUI) on port 3000

Check status:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs -f
```

Stop Milvus:
```bash
docker-compose down
```

### Access Attu (Milvus GUI)

Attu is a web-based graphical interface for visualizing and managing Milvus. After starting the services, access it at:

```
http://localhost:3000
```

**First-time setup:**
1. Open `http://localhost:3000` in your browser
2. You'll see the Attu connection screen
3. Enter the Milvus connection details:
   - **Milvus Address**: `localhost:19530`
   - **Username/Password**: Leave empty (default Milvus setup doesn't require authentication)
4. Click "Connect" to establish the connection

**What you can do with Attu:**
- 📊 **View Collections**: See all your collections and their schemas
- 🔍 **Browse Data**: View stored vectors, chunks, and metadata
- 📈 **Visualize Vectors**: See vector dimensions and statistics
- 🔎 **Query Data**: Execute queries and filter data
- 📉 **Monitor Performance**: View collection statistics and health
- 🛠️ **Manage Collections**: Create, delete, and modify collections

**Option B: Using Milvus Lite (Embedded, No Docker)**

If you prefer not to use Docker, you can use Milvus Lite (embedded version):

1. Set environment variable:
   ```bash
   export USE_MILVUS_LITE=true
   ```

2. The app will automatically use Milvus Lite when started.

### 4. Run the Application

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Access the Web UI

Open your browser and navigate to:
```
http://localhost:8000
```

You'll see a beautiful web interface where you can:
- View all uploaded files with metadata
- Upload documents (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images)
- Query documents with cosine similarity scores
- Inspect stored vectors and embeddings

**Also available:**
- **Attu GUI**: Visualize Milvus data at `http://localhost:3000` (requires Docker Compose)

## 📖 API Endpoints

### 1. Upload Document

Upload a document file (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, or Images) and store its embeddings in Milvus. File metadata is automatically saved to a SQLite database.

**Supported Formats:**
- Documents: PDF, DOCX, PPTX, XLSX
- Web: HTML, Markdown
- Data: CSV
- Images: PNG, JPEG, TIFF, BMP, WEBP (with OCR)

```bash
# Upload a PDF
curl -X POST -F "file=@demo.pdf" http://localhost:8000/upload-pdf

# Upload a Word document
curl -X POST -F "file=@document.docx" http://localhost:8000/upload-pdf

# Upload an image (with OCR)
curl -X POST -F "file=@image.png" http://localhost:8000/upload-pdf
```

**Response:**
```json
{
  "file_id": "file_20251202_123412_abc12345",
  "filename": "document.pdf",
  "file_type": "PDF",
  "file_size": 2567890,
  "chunks_created": 15,
  "vector_ids_count": 15,
  "uploaded_at": "2025-12-02T12:34:12.123456"
}
```

### 2. Query Document

Query a document using RAG pipeline.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "file_20251202_123412_abc12345",
    "query": "What is the conclusion?"
  }'
```

**Response:**
```json
{
  "answer": "Based on the provided context...",
  "file_id": "file_20251202_123412_abc12345",
  "query": "What is the conclusion?"
}
```

### 3. List Uploaded Files

Get a list of all uploaded files with metadata:

```bash
curl "http://localhost:8000/files?limit=50"
```

**Response:**
```json
{
  "files": [
    {
      "id": 1,
      "file_id": "file_20251202_123412_abc12345",
      "filename": "document.pdf",
      "file_type": "PDF",
      "file_size": 2567890,
      "chunks_count": 15,
      "vectors_count": 15,
      "uploaded_at": "2025-12-02T12:34:12",
      "status": "completed"
    }
  ],
  "statistics": {
    "total_files": 5,
    "total_chunks": 75,
    "total_vectors": 75,
    "total_size_bytes": 12839450,
    "total_size_mb": 12.24
  },
  "count": 5
}
```

### 4. Get File Information

Get detailed metadata for a specific file:

```bash
curl "http://localhost:8000/files/file_20251202_123412_abc12345"
```

### 5. Inspect Vectors

Inspect stored vectors for a specific file or all files.

```bash
# Inspect vectors for a specific file
curl "http://localhost:8000/inspect-vectors?file_id=file_20251202_123412_abc12345&limit=10"

# Inspect all vectors
curl "http://localhost:8000/inspect-vectors?limit=10"
```

**Response:**
```json
{
  "count": 10,
  "vectors": [
    {
      "id": 123,
      "file_id": "file_20251202_123412_abc12345",
      "vector_dim": 1536,
      "preview": "The company aims to..."
    }
  ]
}
```

### 6. Web UI

A simple, modern HTML interface is available at:
- Web UI: `http://localhost:8000`

The UI provides:
- Drag-and-drop PDF upload
- Query interface with auto-filled file IDs
- Vector inspection tool
- Real-time feedback and error handling

### 7. API Documentation

Interactive API documentation (Swagger UI) is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📁 Project Structure

```
docling4/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   ├── rag.py               # RAG pipeline (embeddings + query)
│   ├── milvus_client.py    # Milvus connection and operations
│   ├── pdf_utils.py        # PDF text extraction and chunking
│   └── models/
│       └── schema.py       # Milvus collection schema
├── static/
│   └── index.html          # Web UI (HTML/JavaScript)
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Example environment file
├── docker-compose.yml      # Docker Compose config for Milvus (etcd, minio, standalone)
├── requirements.txt        # Python dependencies
├── files.db               # SQLite database (created automatically)
├── README.md              # This file
└── general_requirements.txt
```

## 🔍 Milvus Collection Schema

The collection uses the following schema:

- **id**: INT64 (primary key, auto-generated)
- **file_id**: VARCHAR (255) - Identifier for the uploaded file
- **chunk_text**: VARCHAR (10000) - Text chunk from PDF
- **embedding**: FLOAT_VECTOR (1536) - OpenAI embedding vector

Index: HNSW with L2 distance metric

## 🧪 Example Workflow

1. **Upload a PDF:**
   ```bash
   curl -X POST -F "file=@document.pdf" http://localhost:8000/upload-pdf
   ```
   Save the `file_id` from the response.

2. **Query the document:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"file_id": "YOUR_FILE_ID", "query": "What is this document about?"}'
   ```

3. **Inspect stored vectors:**
   ```bash
   curl "http://localhost:8000/inspect-vectors?file_id=YOUR_FILE_ID"
   ```

## 📝 Notes

- This is an MVP with minimal error handling and no authentication
- Milvus runs locally only (no deployment)
- The system is designed for educational purposes to understand vector storage and RAG pipelines
- Chunks are created with 1000 character size and 200 character overlap by default
- Uses OpenAI's `text-embedding-3-small` model (1536 dimensions)
- Uses GPT-3.5-turbo for answer generation
- Docling handles OCR for images and scanned documents automatically
- Multiple document formats are supported through Docling's conversion pipeline

## 🐛 Troubleshooting

**Milvus connection error:**
- If using docker-compose: `docker-compose ps` to check all services
- Check Milvus logs: `docker-compose logs standalone`
- Verify all services are healthy: `docker-compose ps` should show all services as "healthy"
- If using Milvus Lite: Make sure `milvus` package is installed: `pip install milvus`
- Check if port 19530 is available: `lsof -i :19530`

**OpenAI API error:**
- Verify your API key in `.env` file
- Check your OpenAI account has sufficient credits

**PDF extraction issues:**
- Ensure the PDF contains extractable text (not just images)
- Some PDFs may require OCR preprocessing

