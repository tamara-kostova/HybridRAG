import numpy as np
from src.db.db_client import QdrantWrapper


# Usage
import numpy as np

# Initialize the Qdrant wrapper with your Qdrant instance URL and API key
qdrant_client = QdrantWrapper(url="https://7ecf0b14-c826-4ae4-b61b-3bd710fc75d9.europe-west3-0.gcp.cloud.qdrant.io", 
                               api_key="agxIHD5sPk-2svMtUPmn26Gf3CHZLhmidbz-eOQuOjjushtYCl9aVQ")


query_vector = np.random.rand(384).tolist()

search_results = qdrant_client.search(query_vector=query_vector, limit=13)

# Print the search results
print("Search Results:")
print(len(search_results))
for result in search_results:
    print(f"ID: {result['id']}, Score: {result['score']}, Text: {result['text']}, Paper ID: {result.get('paper_id')}, Chunk Index: {result['chunk_index']}")

# Step 3: Fetch all papers from the collection
"""all_papers = qdrant_client.fetch_all_papers(limit=100)

# Print all papers
print("\nAll Papers:")
for paper in all_papers:
    print(f"ID: {paper['id']}, Paper ID: {paper['paper_id']}, Text: {paper['text']}")"""
