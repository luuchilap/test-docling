"""
Milvus client for vector database operations
"""
import os
from pymilvus import connections, Collection, utility
from app.models.schema import collection_schema, COLLECTION_NAME

# Use Milvus Lite (embedded) for local development, or Standalone via docker-compose
USE_MILVUS_LITE = os.getenv("USE_MILVUS_LITE", "false").lower() == "true"


def connect_milvus(host="localhost", port=19530):
    """Connect to Milvus instance (Lite or Standalone)"""
    try:
        # Try to disconnect existing connection if any
        try:
            connections.disconnect("default")
        except:
            pass  # Ignore if no connection exists
        
        if USE_MILVUS_LITE:
            # Use Milvus Lite (embedded, no Docker needed)
            from milvus import default_server
            default_server.start()
            connections.connect(
                alias="default",
                host="localhost",
                port=default_server.listen_port
            )
            print(f"✓ Connected to Milvus Lite (embedded) on port {default_server.listen_port}")
        else:
            # Use standalone Milvus (requires Docker with etcd/minio)
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            print(f"✓ Connected to Milvus Standalone at {host}:{port}")
    except Exception as e:
        print(f"✗ Failed to connect to Milvus: {e}")
        if USE_MILVUS_LITE:
            print("  Make sure 'milvus' package is installed: pip install milvus")
        else:
            print("  Make sure Milvus is running with all dependencies (etcd, minio)")
        raise


def ensure_connection(host="localhost", port=19530):
    """Ensure Milvus connection exists, reconnect if needed"""
    try:
        # Try to use the connection by checking if we can list collections
        # This will fail if connection doesn't exist
        utility.list_collections()
    except Exception:
        # Connection doesn't exist or is invalid, reconnect
        print("⚠ No active connection, reconnecting to Milvus...")
        try:
            connect_milvus(host, port)
        except Exception as e:
            print(f"✗ Failed to reconnect to Milvus: {e}")
            raise Exception(f"Milvus connection failed. Make sure Milvus is running: docker run -p 19530:19530 milvusdb/milvus:latest")


def drop_collection_if_exists():
    """Drop the collection if it exists (use with caution!)"""
    ensure_connection()
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"✓ Dropped collection: {COLLECTION_NAME}")
        return True
    return False


def get_or_create_collection():
    """Get existing collection or create a new one"""
    # Ensure connection exists
    ensure_connection()
    
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        print(f"✓ Found existing collection: {COLLECTION_NAME}")
        
        # Verify schema matches (basic check)
        try:
            schema = collection.schema
            field_names = [field.name for field in schema.fields]
            expected_fields = ["id", "file_id", "chunk_text", "embedding"]
            
            if set(field_names) != set(expected_fields):
                print(f"⚠ Warning: Collection schema mismatch. Expected: {expected_fields}, Got: {field_names}")
                print(f"  Consider dropping and recreating the collection if issues persist.")
                print(f"  You can use: from app.milvus_client import drop_collection_if_exists; drop_collection_if_exists()")
        except Exception as e:
            print(f"⚠ Could not verify collection schema: {e}")
    else:
        collection = Collection(
            name=COLLECTION_NAME,
            schema=collection_schema
        )
        print(f"✓ Created new collection: {COLLECTION_NAME}")
    
    # Create index if not exists
    if not collection.has_index():
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print(f"✓ Created HNSW index on embedding field")
    
    # Load collection
    try:
        collection.load()
        print(f"✓ Collection loaded and ready")
    except Exception as e:
        print(f"✗ Failed to load collection: {e}")
        raise
    
    return collection


def insert_chunks(collection, file_id, chunks, embeddings):
    """Insert chunks and embeddings into Milvus"""
    # Validate inputs
    if not file_id or not file_id.strip():
        raise ValueError("file_id cannot be empty")
    if not chunks or len(chunks) == 0:
        raise ValueError("chunks cannot be empty")
    if not embeddings or len(embeddings) == 0:
        raise ValueError("embeddings cannot be empty")
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks and embeddings must have the same length: {len(chunks)} != {len(embeddings)}")
    
    print(f"  🔄 Step 6: Inserting vectors into Milvus...")
    print(f"     - Collection: {collection.name}")
    print(f"     - Chunks to insert: {len(chunks)}")
    print(f"     - Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"     - File ID: {file_id}")
    
    # Clean and validate file_id
    file_id_clean = file_id.strip()
    if len(file_id_clean) > 255:
        raise ValueError(f"file_id exceeds max length of 255: {len(file_id_clean)} characters")
    
    # Validate chunk_text lengths (max 10000 characters per schema)
    for i, chunk in enumerate(chunks):
        if len(chunk) > 10000:
            print(f"     ⚠ Warning: Chunk {i} exceeds max length (10000), truncating...")
            chunks[i] = chunk[:10000]
    
    # Validate embedding dimensions
    expected_dim = 1536
    for i, emb in enumerate(embeddings):
        if len(emb) != expected_dim:
            raise ValueError(f"Embedding {i} has wrong dimension: {len(emb)} != {expected_dim}")
    
    # Prepare entities - ensure all are lists
    file_ids = [file_id_clean] * len(chunks)
    
    # Ensure chunks is a list of strings
    if not isinstance(chunks, list):
        chunks = list(chunks)
    
    # Ensure embeddings is a list of lists
    if not isinstance(embeddings, list):
        embeddings = list(embeddings)
    
    entities = [
        file_ids,  # file_id - list of strings
        chunks,    # chunk_text - list of strings
        embeddings # embedding - list of lists of floats
    ]
    
    print(f"     - Preparing {len(chunks)} entities for insertion...")
    print(f"     - File ID length: {len(file_id_clean)} (max 255)")
    print(f"     - Sample chunk length: {len(chunks[0]) if chunks else 0} (max 10000)")
    print(f"     - Sample embedding length: {len(embeddings[0]) if embeddings else 0} (expected 1536)")
    
    try:
        insert_result = collection.insert(entities)
    except Exception as e:
        error_msg = str(e)
        print(f"     ✗ Insert failed: {error_msg}")
        print(f"     - File ID: '{file_id_clean}'")
        print(f"     - File ID type: {type(file_id_clean)}")
        print(f"     - File IDs list type: {type(file_ids)}")
        print(f"     - File IDs list length: {len(file_ids)}")
        print(f"     - Chunks type: {type(chunks)}")
        print(f"     - Chunks length: {len(chunks)}")
        print(f"     - Embeddings type: {type(embeddings)}")
        print(f"     - Embeddings length: {len(embeddings)}")
        raise
    print(f"     - Entities inserted, flushing to disk...")
    collection.flush()
    print(f"     - Flush complete")
    
    # Calculate storage estimate
    total_vectors = len(insert_result.primary_keys)
    vector_size_bytes = len(embeddings[0]) * 4 if embeddings else 0  # 4 bytes per float
    total_vector_size = total_vectors * vector_size_bytes
    total_text_size = sum(len(c) for c in chunks)
    
    print(f"  ✓ Vector Storage Complete:")
    print(f"     - Vectors inserted: {total_vectors}")
    print(f"     - Vector IDs: {insert_result.primary_keys[:5]}{'...' if len(insert_result.primary_keys) > 5 else ''}")
    print(f"     - Estimated vector storage: {total_vector_size / 1024:.2f} KB")
    print(f"     - Text storage: {total_text_size:,} characters")
    print(f"     - File ID: {file_id}")
    
    return insert_result.primary_keys


def search_similar(collection, query_embedding, file_id, top_k=5, include_embeddings=False):
    """Search for similar chunks using vector similarity"""
    # Ensure connection exists
    ensure_connection()
    
    print(f"     - Collection: {collection.name}")
    print(f"     - Search metric: L2 (Euclidean distance)")
    print(f"     - Top K: {top_k}")
    print(f"     - Query vector dimension: {len(query_embedding)}")
    
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 100}
    }
    
    # Validate file_id
    if not file_id or not file_id.strip():
        raise ValueError("file_id cannot be empty")
    
    # Use single quotes for VARCHAR field comparison in Milvus
    # Escape single quotes in file_id if present
    file_id_clean = file_id.strip()
    escaped_file_id = file_id_clean.replace("'", "\\'").replace('"', '\\"')
    expr = f"file_id == '{escaped_file_id}'"
    print(f"     - Filter expression: {expr}")
    
    # Include embeddings if needed for cosine similarity calculation
    output_fields = ["file_id", "chunk_text"]
    if include_embeddings:
        output_fields.append("embedding")
        print(f"     - Including embeddings in results")
    
    print(f"     - Executing vector search...")
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=output_fields
    )
    
    print(f"     ✓ Search complete, found {len(results[0])} results")
    
    return results[0]  # Return first query result


def inspect_vectors(collection, file_id=None, limit=10, full_content=False, show_vectors=False):
    """Inspect stored vectors for educational purposes"""
    # Ensure connection exists
    ensure_connection()
    
    # If vectors are requested, we must use search() instead of query()
    # because Milvus doesn't allow querying vector fields directly
    if show_vectors:
        # Use search with a dummy zero vector to retrieve embeddings
        # Create a zero vector of dimension 1536
        dummy_vector = [0.0] * 1536
        
        search_params = {
            "metric_type": "L2",
            "params": {"ef": 100}
        }
        
        # Build expression filter
        expr = None
        if file_id and file_id.strip():
            file_id_clean = file_id.strip()
            escaped_file_id = file_id_clean.replace("'", "\\'").replace('"', '\\"')
            expr = f"file_id == '{escaped_file_id}'"
        
        # Use search to get vectors (with a large limit to get all matching)
        output_fields = ["id", "file_id", "chunk_text", "embedding"]
        search_results = collection.search(
            data=[dummy_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields
        )
        
        # Extract results from search (search returns hits)
        results = []
        for hit_list in search_results:
            for hit in hit_list:
                entity = hit.entity
                results.append({
                    "id": entity.get("id"),
                    "file_id": entity.get("file_id"),
                    "chunk_text": entity.get("chunk_text"),
                    "embedding": entity.get("embedding")
                })
    else:
        # No vectors needed, use regular query (faster)
        output_fields = ["id", "file_id", "chunk_text"]
        
        if file_id and file_id.strip():
            file_id_clean = file_id.strip()
            escaped_file_id = file_id_clean.replace("'", "\\'").replace('"', '\\"')
            expr = f"file_id == '{escaped_file_id}'"
            results = collection.query(
                expr=expr,
                limit=limit,
                output_fields=output_fields
            )
        else:
            # Query all vectors (no filter)
            try:
                results = collection.query(
                    expr="id > -1",
                    limit=limit,
                    output_fields=output_fields
                )
            except Exception:
                try:
                    results = collection.query(
                        expr="",
                        limit=limit,
                        output_fields=output_fields
                    )
                except Exception:
                    results = collection.query(
                        expr="id >= 0",
                        limit=limit,
                        output_fields=output_fields
                    )
    
    # Format results
    formatted_results = []
    for result in results:
        chunk_text = result["chunk_text"]
        formatted_result = {
            "id": result["id"],
            "file_id": result["file_id"],
            "vector_dim": 1536,
            "chunk_length": len(chunk_text),
            "preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        }
        
        # Add full content if requested
        if full_content:
            formatted_result["full_content"] = chunk_text
        
        # Add embedding vector if requested
        if show_vectors and "embedding" in result:
            embedding = result["embedding"]
            formatted_result["embedding"] = embedding
            formatted_result["vector_values_count"] = len(embedding)
        
        formatted_results.append(formatted_result)
    
    return formatted_results

