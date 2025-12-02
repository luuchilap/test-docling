"""
RAG pipeline: embeddings generation and query processing
"""
import os
import numpy as np
from openai import OpenAI
from typing import List, Tuple, Union
from app.milvus_client import search_similar, get_or_create_collection


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    print(f"✓ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    return embeddings


def query_embedding(query: str) -> List[float]:
    """Generate embedding for a single query"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Calculate dot product
    dot_product = np.dot(vec1_np, vec2_np)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1_np)
    magnitude2 = np.linalg.norm(vec2_np)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Cosine similarity = dot product / (magnitude1 * magnitude2)
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return float(cosine_sim)


def retrieve_context(file_id: str, query: str, top_k: int = 5, return_similarities: bool = False):
    """
    Retrieve relevant chunks from Milvus for a query
    
    Returns:
    - If return_similarities=False: List of chunk texts
    - If return_similarities=True: Tuple of (chunk texts, similarity info list)
    """
    collection = get_or_create_collection()
    query_emb = query_embedding(query)
    
    results = search_similar(collection, query_emb, file_id, top_k=top_k, include_embeddings=return_similarities)
    
    chunks = []
    similarity_info = []
    
    for hit in results:
        chunk_text = hit.entity.get("chunk_text")
        chunks.append(chunk_text)
        
        if return_similarities:
            # Get the distance from Milvus (L2 distance)
            distance = hit.distance
            
            # Calculate cosine similarity from the query embedding
            # Note: Milvus returns L2 distance, we need to calculate cosine similarity separately
            # For cosine similarity, we can use: cos(θ) = 1 - (L2_distance^2) / 2 for normalized vectors
            # But OpenAI embeddings are normalized, so we can calculate directly
            chunk_embedding = hit.entity.get("embedding")
            if chunk_embedding:
                cosine_sim = calculate_cosine_similarity(query_emb, chunk_embedding)
            else:
                # Fallback: approximate cosine from L2 distance (for normalized vectors)
                # cos(θ) ≈ 1 - (L2^2) / 2 for normalized vectors
                cosine_sim = 1 - (distance ** 2) / 2
                cosine_sim = max(-1.0, min(1.0, cosine_sim))  # Clamp to [-1, 1]
            
            similarity_info.append({
                "chunk_text": chunk_text,
                "l2_distance": float(distance),
                "cosine_similarity": cosine_sim,
                "similarity_percentage": round(cosine_sim * 100, 2)
            })
    
    print(f"✓ Retrieved {len(chunks)} relevant chunks")
    
    if return_similarities:
        return chunks, similarity_info
    return chunks


def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate answer using LLM with retrieved context"""
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You must answer from the provided context only.

Context:
{context}

User question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    return answer

