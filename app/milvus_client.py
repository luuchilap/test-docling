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


def get_or_create_collection():
    """Get existing collection or create a new one"""
    # Ensure connection exists
    ensure_connection()
    
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        print(f"✓ Found existing collection: {COLLECTION_NAME}")
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
    collection.load()
    print(f"✓ Collection loaded and ready")
    
    return collection


def insert_chunks(collection, file_id, chunks, embeddings):
    """Insert chunks and embeddings into Milvus"""
    entities = [
        [file_id] * len(chunks),  # file_id
        chunks,  # chunk_text
        embeddings  # embedding
    ]
    
    insert_result = collection.insert(entities)
    collection.flush()
    
    print(f"✓ Inserted {len(chunks)} chunks for file_id: {file_id}")
    print(f"  Vector IDs: {insert_result.primary_keys[:5]}..." if len(insert_result.primary_keys) > 5 else f"  Vector IDs: {insert_result.primary_keys}")
    
    return insert_result.primary_keys


def search_similar(collection, query_embedding, file_id, top_k=5, include_embeddings=False):
    """Search for similar chunks using vector similarity"""
    # Ensure connection exists
    ensure_connection()
    
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 100}
    }
    
    # Use single quotes for VARCHAR field comparison in Milvus
    # Escape single quotes in file_id if present
    escaped_file_id = file_id.replace("'", "\\'")
    expr = f"file_id == '{escaped_file_id}'"
    
    # Include embeddings if needed for cosine similarity calculation
    output_fields = ["file_id", "chunk_text"]
    if include_embeddings:
        output_fields.append("embedding")
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=output_fields
    )
    
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
            escaped_file_id = file_id.strip().replace("'", "\\'")
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
            escaped_file_id = file_id.strip().replace("'", "\\'")
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

