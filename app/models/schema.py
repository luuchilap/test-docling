"""
Milvus collection schema definition
"""
from pymilvus import CollectionSchema, FieldSchema, DataType

# Define fields
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

file_id_field = FieldSchema(
    name="file_id",
    dtype=DataType.VARCHAR,
    max_length=255
)

chunk_text_field = FieldSchema(
    name="chunk_text",
    dtype=DataType.VARCHAR,
    max_length=10000
)

embedding_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=1536
)

# Create collection schema
collection_schema = CollectionSchema(
    fields=[id_field, file_id_field, chunk_text_field, embedding_field],
    description="PDF document chunks with embeddings"
)

# Collection name
COLLECTION_NAME = "pdf_chunks"

