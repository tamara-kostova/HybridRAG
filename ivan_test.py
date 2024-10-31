import numpy as np
from src.db.db_client import QdrantWrapper

# Initialize the Qdrant wrapper with your Qdrant instance URL and API key
qdrant_client = QdrantWrapper(
    url="https://7ecf0b14-c826-4ae4-b61b-3bd710fc75d9.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="agxIHD5sPk-2svMtUPmn26Gf3CHZLhmidbz-eOQuOjjushtYCl9aVQ"
)

# Step 1: Create a dense vector for dense retrieval
query_vector = np.random.rand(384).tolist()  # Random vector as placeholder

# Step 2: Define query text for sparse retrieval
query_text = "Neuroscience research on memory formation"  # Example query text
query_terms = query_text.lower().split()  # Tokenize and lowercase for sparse retrieval

# Step 3: Perform hybrid search
search_results = qdrant_client.hybrid_search(query_vector=query_vector, query_terms=query_terms, limit=13)

# Step 4: Display search results
print("Hybrid Search Results:")
print(f"Total Results Found: {len(search_results)}\n")
for result in search_results:
    print(f"ID: {result['id']}, Score: {result['score']:.4f}, Text: {result['text']}, "
          f"Paper ID: {result.get('paper_id')}, Chunk Index: {result['chunk_index']}")
