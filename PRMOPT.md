## Goal

Use this document as the single source of truth for structure, stack, and behaviors.

---

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Vector DB**: Milvus Standalone (via Docker Compose) or Milvus Lite (embedded)
- **RAG Logic**: OpenAI `text-embedding-3-small` for embeddings, `gpt-3.5-turbo` (or newer) for answers
- **Document Parsing**: Docling (multi-format: PDF, DOCX, PPTX, XLSX, HTML/MD/CSV, images with OCR)
- **Metadata DB**: SQLite (`files.db`)
- **Frontend**: Single-page HTML/JS UI served by FastAPI

---

## Project Structure (target)

- `app/`
  - `main.py` – FastAPI app, routes, startup logic, static UI mounting
  - `milvus_client.py` – Milvus connection, collection schema management, insert/search/inspect helpers
  - `models/`
    - `schema.py` – Milvus collection schema (`pdf_chunks` with `id`, `file_id`, `chunk_text`, `embedding`)
  - `pdf_utils.py` – Docling-based extraction + text chunking (chunk size ~1000 chars, 200 overlap)
  - `rag.py` – Embedding generation, query embedding, cosine similarity, context retrieval, answer generation
  - `database.py` – SQLite helpers (`files` table, CRUD + stats)
- `static/`
  - `index.html` – Single-page UI (upload, query, inspect vectors, list files)
- `docker-compose.yml` – Milvus Standalone + etcd + MinIO + Attu
- `requirements.txt` – All pinned Python deps
- `README.md` – Setup, run, and troubleshooting guide
- `reset_milvus.py` – Utility to drop/recreate the Milvus collection

---

## Core Behaviors to Preserve

- **Upload endpoint** (`POST /upload-pdf`):
  - Accepts a single file (PDF, DOCX, PPTX, XLSX, HTML/MD/CSV, images).
  - Saves to a temp path.
  - Extracts text via Docling (PDF fallback to `pypdf`).
  - Chunks text (`chunk_size ≈ 1000`, `overlap ≈ 200`, try to split on sentence boundaries).
  - Calls OpenAI embeddings for all chunks (`text-embedding-3-small`).
  - Inserts into Milvus (`pdf_chunks`): `file_id`, `chunk_text`, `embedding`.
  - Writes file metadata into SQLite (`files` table).
  - Returns JSON including `file_id`, chunk/vector counts, file info, timestamp.

- **Query endpoint** (`POST /query`):
  - Body: `{ file_id, query }`, query params: `show_similarity`, `show_query_embedding`.
  - Generates query embedding (optionally exposed in response).
  - Uses Milvus vector search filtered by `file_id` to get top-k chunks.
  - Optionally returns cosine similarity stats per chunk.
  - Calls OpenAI chat completion with retrieved context to produce answer.
  - Returns answer, echo of `file_id/query`, optional similarity scores + query embedding vector.

- **Metadata endpoints**:
  - `GET /files` – list latest files + aggregate stats.
  - `GET /files/{file_id}` – single file metadata.

- **Vector inspection**:
  - `GET /inspect-vectors` with query params `file_id`, `limit`, `full_content`, `show_vectors`.
  - When `show_vectors=true`, use Milvus `search` with a dummy zero vector to fetch embeddings (1536 floats) plus chunk text.

- **UI behaviors** (`static/index.html`):
  - Section 1: Files list (loads `/files`, shows stats, “Use This File” button populating file_id in forms).
  - Section 2: Upload (shows upload result + file_id, buttons to copy ID and refresh list).
  - Section 3: Query:
    - Inputs: `file_id`, `query`.
    - Checkboxes:
      - Show cosine similarity scores.
      - Show query embedding vector (render full array and first 10 sample values).
  - Section 4: Inspect vectors:
    - Inputs: `file_id` (optional), `limit`, `full_content`, `show_vectors`.
    - Displays each chunk, its metadata, and (optionally) the full 1536-dim vector.

---

## Milvus & Storage Requirements

- Milvus collection: `pdf_chunks`
  - `id`: INT64, primary key, auto_id
  - `file_id`: VARCHAR(255)
  - `chunk_text`: VARCHAR(10000)
  - `embedding`: FLOAT_VECTOR(1536)
- Index: HNSW on `embedding` (`metric_type=L2`, `M=16`, `efConstruction=200`), and `collection.load()` on startup.
- Milvus runs either:
  - **Standalone** via `docker-compose up -d` (etcd + MinIO + Attu)
  - Or **Lite** via `milvus` Python package (controlled by `USE_MILVUS_LITE` env).
- SQLite DB:
  - File: `files.db` in project root.
  - `files` table columns: `file_id`, `filename`, `file_type`, `file_size`, `chunks_count`, `vectors_count`, `status`, `uploaded_at`.

---

## Step-by-Step Rebuild Plan

1. **Create new repo + environment**
   - Initialize a new Python project.
   - Create and activate a virtualenv.
   - Add a `requirements.txt` mirroring the current project’s pinned versions (FastAPI, Uvicorn, pymilvus, milvus, openai, docling, pypdf, numpy, python-multipart, sqlite3 stdlib, etc.).

2. **Implement schema and Milvus client**
   - Recreate `app/models/schema.py` with the `pdf_chunks` schema.
   - Recreate `app/milvus_client.py`:
     - `connect_milvus`, `ensure_connection`, `get_or_create_collection`, `insert_chunks`, `search_similar`, `inspect_vectors`.
     - Make sure insert accepts lists: `[file_ids, chunks, embeddings]`.

3. **Implement document extraction and chunking**
   - Recreate `app/pdf_utils.py`:
     - `extract_text_from_document` using Docling + OCR for images.
     - Fallback `extract_text_from_pdf_fallback` with `pypdf`.
     - `chunk_text(text, chunk_size=1000, overlap=200)` with sentence-aware splitting and logging.

4. **Implement RAG logic**
   - Recreate `app/rag.py`:
     - `generate_embeddings(texts)`.
     - `query_embedding(query)`.
     - `calculate_cosine_similarity(vec1, vec2)`.
     - `retrieve_context(file_id, query, top_k, return_similarities, query_embedding_override)` using Milvus search.
     - `generate_answer(query, context_chunks)` using OpenAI chat completions.

5. **Implement SQLite layer**
   - Recreate `app/database.py`:
     - `DB_PATH` pointing to `files.db` in project root.
     - `init_database()` that creates `files` table if not exists.
     - `save_file_metadata`, `get_file_metadata`, `list_all_files`, `delete_file_metadata`, `get_file_statistics`.

6. **Implement FastAPI app**
   - Recreate `app/main.py`:
     - Initialize app, load `.env`, CORS, static mounting.
     - `startup` event: init DB, connect to Milvus, ensure collection.
     - Endpoints:
       - `POST /upload-pdf`.
       - `POST /query` with `show_similarity`, `show_query_embedding`.
       - `GET /files`, `GET /files/{file_id}`.
       - `GET /inspect-vectors`.
       - Optionally `/query-milvus` for direct Milvus expressions.
       - Root `/` returns `static/index.html` through `HTMLResponse`.

7. **Rebuild the UI**
   - Recreate `static/index.html` with:
     - Modern, single-page layout (cards, buttons, etc.).
     - JS `fetch` calls to `/upload-pdf`, `/files`, `/query`, `/inspect-vectors`.
     - Controls for:
       - Upload document.
       - Select file from list.
       - Query with options for similarity + query embedding.
       - Inspect vectors with optional full content and full vector dump.

8. **Milvus Docker & Attu**
   - Recreate `docker-compose.yml`:
     - Services: `etcd`, `minio`, `standalone`, `attu`.
     - Expose Milvus on `19530`, Attu on `3000`.
   - Ensure volumes are mounted under a predictable `volumes/` directory.

9. **Documentation**
   - Rebuild `README.md`:
     - Setup: install deps, set `OPENAI_API_KEY`, start Milvus, run FastAPI.
     - Usage: upload → query → inspect → Attu.
     - Troubleshooting: Milvus connection, marshmallow/environs conflicts, OpenAI errors, port collisions.

10. **Verification Checklist**
   - Upload multiple formats (PDF, DOCX, image) and confirm:
     - Entries in SQLite (`files.db`).
     - Vectors + text in Milvus (`pdf_chunks`), visible in Attu.
   - Run a query with:
     - Similarity scores enabled → see per-chunk cosine similarity.
     - Query embedding enabled → see full 1536-dim query vector.
   - Inspect vectors via `/inspect-vectors` and the UI.

---

## Non-Functional Requirements

- Log clearly every major step (upload, extract, chunk, embed, insert, query, answer).
- Validate inputs: non-empty `file_id`, `query`, matching chunk/embedding lengths, embedding dim = 1536.
- Use robust Milvus expressions (`file_id == '...'`, escape quotes).
- Prefer clear, educational error messages (especially for Milvus and OpenAI issues).


