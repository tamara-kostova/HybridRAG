import numpy as np
from src.db.db_client import QdrantWrapper
from src.db.ollama_client import OllamaClient
from typing import List  

def get_index_for_term(term):
    term_to_index_map = {
        "neuroscience": 1,
        "research": 2,
        "on": 3,
        "memory": 4,
        "formation": 5,
    }
    return term_to_index_map.get(term)

def main():
    qdrant_client = QdrantWrapper(
        url="https://7ecf0b14-c826-4ae4-b61b-3bd710fc75d9.europe-west3-0.gcp.cloud.qdrant.io",
        api_key="agxIHD5sPk-2svMtUPmn26Gf3CHZLhmidbz-eOQuOjjushtYCl9aVQ"
    )

    ollama_client = OllamaClient()

    query_text = "Neuroscience research on memory formation"
    
    # Debugging: Check if Ollama service is reachable
    try:
        query_vector = ollama_client.generate_embedding(query_text)
        print("Query vector generated successfully.")
    except Exception as e:
        print(f"Failed to generate query vector in main: {e}")
        return  # Exit if embedding generation fails

    search_results = qdrant_client.hybrid_search(query_vector=query_vector, limit=13)

    print("Hybrid Search Results:")
    print(f"Total Results Found: {len(search_results)}\n")
    for result in search_results:
        print(f"ID: {result['id']}, Score: {result['score']:.4f}, Text: {result['text']}, "
              f"Paper ID: {result.get('paper_id')}, Chunk Index: {result['chunk_index']}")

if __name__ == "__main__":
    main()
