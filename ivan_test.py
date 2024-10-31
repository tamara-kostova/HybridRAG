import numpy as np
from src.db.db_client import QdrantWrapper
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


qdrant_client = QdrantWrapper(
    url="https://7ecf0b14-c826-4ae4-b61b-3bd710fc75d9.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="agxIHD5sPk-2svMtUPmn26Gf3CHZLhmidbz-eOQuOjjushtYCl9aVQ"
)

def generate_query_vector(query_text: str) -> List[float]:
  return np.random.rand(384).tolist()  
if __name__ == "__main__":
    query_text = "Neuroscience research on memory formation"  

    query_vector = generate_query_vector(query_text)

    search_results = qdrant_client.hybrid_search(query_vector=query_vector, limit=13)

    print("Hybrid Search Results:")
    print(f"Total Results Found: {len(search_results)}\n")
    for result in search_results:
        print(f"ID: {result['id']}, Score: {result['score']:.4f}, Text: {result['text']}, "
              f"Paper ID: {result.get('paper_id')}, Chunk Index: {result['chunk_index']}")
