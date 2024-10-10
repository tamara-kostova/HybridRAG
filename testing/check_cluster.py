import json
from src.db.db_client import QdrantWrapper

with open("src/db/config.json") as f:
    config = json.load(f)
qdrant_wrapper = QdrantWrapper(url=config["QDRANT_HOST"], api_key=config["QDRANT_API_KEY"])

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