"""
Busca usando Chroma Cloud
"""

import chromadb
from src.services.embedder import DocumentEmbedder
from src.application.search_validator import verify_search_query
from src.config.chroma_config import CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE

def search_movies_cloud(query, n_results=5):
    """Busca filmes no Chroma Cloud"""
    
    message = verify_search_query(query)
    if message != "OK!":
        return {
            'documents': [[message]],
            'distances': [[0.0]],
            'metadatas': [[{'error': True}]]
        }
    
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    
    collection = client.get_collection("movies_docs")
    
    embedder = DocumentEmbedder(model_name='all-MiniLM-L6-v2')
    query_embedding = embedder.model.encode([query], normalize_embeddings=True)[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return results
