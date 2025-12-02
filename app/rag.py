"""
RAG pipeline: embeddings generation and query processing
"""
import os
import numpy as np
from openai import OpenAI
from typing import List, Tuple, Union, Optional
from app.milvus_client import search_similar, get_or_create_collection


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI"""
    print(f"  🔄 Step 5: Generating embeddings with OpenAI...")
    print(f"     - Model: text-embedding-3-small")
    print(f"     - Number of texts: {len(texts)}")
    print(f"     - Total characters: {sum(len(t) for t in texts):,}")
    
    # Show progress for large batches
    if len(texts) > 10:
        print(f"     - Processing in batches (this may take a moment)...")
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    
    # Calculate statistics
    embedding_dim = len(embeddings[0]) if embeddings else 0
    total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
    
    print(f"  ✓ Embedding Generation Complete:")
    print(f"     - Embeddings generated: {len(embeddings)}")
    print(f"     - Vector dimension: {embedding_dim}")
    print(f"     - Total tokens used: {total_tokens:,}" if total_tokens > 0 else "")
    print(f"     - Sample vector (first 5 values): {embeddings[0][:5] if embeddings else 'N/A'}")
    
    return embeddings


def query_embedding(query: str) -> List[float]:
    """Generate embedding for a single query"""
    print(f"  🔄 Step 1: Generating query embedding...")
    print(f"     - Query length: {len(query)} characters")
    print(f"     - Query: \"{query[:100]}{'...' if len(query) > 100 else ''}\"")
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    embedding = response.data[0].embedding
    
    print(f"     ✓ Query embedding generated:")
    print(f"       - Dimension: {len(embedding)}")
    print(f"       - Sample values: {embedding[:5]}")
    
    return embedding


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


def retrieve_context(
    file_id: str,
    query: str,
    top_k: int = 5,
    return_similarities: bool = False,
    query_embedding_override: Optional[List[float]] = None,
):
    """
    Retrieve relevant chunks from Milvus for a query
    
    Returns:
    - If return_similarities=False: List of chunk texts
    - If return_similarities=True: Tuple of (chunk texts, similarity info list)
    """
    print(f"  🔄 Step 2: Searching Milvus for similar chunks...")
    print(f"     - File ID: {file_id}")
    print(f"     - Top K: {top_k}")
    
    collection = get_or_create_collection()
    
    # Use precomputed embedding if provided (to avoid duplicate OpenAI calls)
    if query_embedding_override is not None:
        query_emb = query_embedding_override
        print(f"  🔄 Step 2a: Using precomputed query embedding (dim={len(query_emb)})")
    else:
        query_emb = query_embedding(query)
    
    print(f"  🔄 Step 3: Performing vector similarity search...")
    results = search_similar(collection, query_emb, file_id, top_k=top_k, include_embeddings=return_similarities)
    
    chunks = []
    similarity_info = []
    
    print(f"  🔄 Step 4: Processing search results...")
    for i, hit in enumerate(results, 1):
        chunk_text = hit.entity.get("chunk_text")
        chunks.append(chunk_text)
        
        if return_similarities:
            # Get the distance from Milvus (L2 distance)
            distance = hit.distance
            
            # Calculate cosine similarity from the query embedding
            chunk_embedding = hit.entity.get("embedding")
            if chunk_embedding:
                cosine_sim = calculate_cosine_similarity(query_emb, chunk_embedding)
            else:
                # Fallback: approximate cosine from L2 distance (for normalized vectors)
                cosine_sim = 1 - (distance ** 2) / 2
                cosine_sim = max(-1.0, min(1.0, cosine_sim))  # Clamp to [-1, 1]
            
            similarity_info.append({
                "chunk_text": chunk_text,
                "l2_distance": float(distance),
                "cosine_similarity": cosine_sim,
                "similarity_percentage": round(cosine_sim * 100, 2)
            })
            
            print(f"     - Result {i}: L2={distance:.4f}, Cosine={cosine_sim:.4f} ({cosine_sim*100:.1f}%), Chunk length={len(chunk_text)} chars")
        else:
            print(f"     - Result {i}: L2={hit.distance:.4f}, Chunk length={len(chunk_text)} chars")
    
    print(f"  ✓ Retrieved {len(chunks)} relevant chunks")
    if similarity_info:
        avg_sim = sum(s["cosine_similarity"] for s in similarity_info) / len(similarity_info)
        print(f"     - Average cosine similarity: {avg_sim:.4f} ({avg_sim*100:.1f}%)")
        print(f"     - Best match: {max(s['cosine_similarity'] for s in similarity_info):.4f} ({max(s['similarity_percentage'] for s in similarity_info):.1f}%)")
    
    if return_similarities:
        return chunks, similarity_info
    return chunks


def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate answer using LLM with retrieved context"""
    print(f"  🔄 Step 5: Generating answer with LLM...")
    print(f"     - Model: gpt-3.5-turbo")
    print(f"     - Context chunks: {len(context_chunks)}")
    print(f"     - Total context length: {sum(len(c) for c in context_chunks):,} characters")
    
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You must answer from the provided context only.

Context:
{context}

User question: {query}

Answer:"""
    
    prompt_tokens = len(prompt.split())  # Approximate
    print(f"     - Prompt tokens (approx): {prompt_tokens:,}")
    
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
    answer_length = len(answer)
    answer_words = len(answer.split())
    
    # Get token usage if available
    if hasattr(response, 'usage') and response.usage:
        print(f"  ✓ Answer Generation Complete:")
        print(f"     - Answer length: {answer_length} characters, {answer_words} words")
        print(f"     - Tokens used: {response.usage.total_tokens:,} (prompt: {response.usage.prompt_tokens:,}, completion: {response.usage.completion_tokens:,})")
    else:
        print(f"  ✓ Answer Generation Complete:")
        print(f"     - Answer length: {answer_length} characters, {answer_words} words")
    
    return answer

