import sys
import os

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client import QdrantClient
from src.db.db_client import QdrantWrapper
from src.db.retrievers.retriever_lexical import LexicalRetriever
from src.db.retrievers.retriever_semantic import SemanticRetriever
from src.db.retrievers.retriever_hybrid import HybridRetriever

qdrant_client = QdrantClient(url="localhost", api_key="")
db_client = QdrantWrapper(url="localhost", api_key="")

# Initialize retrievers
semantic_retriever = SemanticRetriever(db_client=db_client, k=5)
lexical_retriever = LexicalRetriever(db_client=db_client)

# Initialize hybrid retriever
hybrid_retriever = HybridRetriever(
    semantic_retriever=semantic_retriever,
    lexical_retriever=lexical_retriever,
    k=5
)

# Perform a search query
query = "What is the impact of climate change on marine life?"
results = hybrid_retriever.retrieve(query)

# Print the results
for result in results:
    print(f"Paper ID: {result.metadata['paper_id']}")
    print(f"Text: {result.page_content}")
    print()