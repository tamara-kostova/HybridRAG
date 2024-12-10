import json
import sys
from src.db.db_client import QdrantWrapper

if len(sys.argv) != 3:
    print("Usage: python script.py <QDRANT_HOST> <QDRANT_API_KEY>")
    sys.exit(1)

qdrant_host = sys.argv[1]
qdrant_api_key = sys.argv[2]

qdrant_wrapper = QdrantWrapper(
    url=qdrant_host, api_key=qdrant_api_key
)

# Get collection info
collection_info = qdrant_wrapper.get_collection_info()
print("Collection Info:", collection_info)

# Get all documents
documents = qdrant_wrapper.fetch_all_papers(limit=100)
print(f"Retrieved {len(documents)} documents:")
for doc in documents[:5]:
    print(f"ID: {doc['id']}, Paper ID: {doc['paper_id']}")
    print(f"Text: {doc['text'][:100]}...")
    print()
