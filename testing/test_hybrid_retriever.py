import sys
import os
from dotenv import load_dotenv

# Load environment variables from secrets.env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'secrets.env'))

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client import QdrantClient
from src.db.db_client import QdrantWrapper
from src.db.retrievers.retriever_lexical import LexicalRetriever
from src.db.retrievers.retriever_semantic import SemanticRetriever
from src.db.retrievers.retriever_hybrid import HybridRetriever

# Get Qdrant connection details from environment variables
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
llm_url = "http://localhost:11434"
model = "llama3:8b"

# Initialize Qdrant client and wrapper with cloud details
qdrant_client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
db_client = QdrantWrapper(url=qdrant_host, api_key=qdrant_api_key)

# Debugging: Check if the API key is correct and has the necessary permissions
try:
    collections = qdrant_client.get_collections()
    print(f"Collections: {collections}")
except Exception as e:
    print(f"Error accessing collections: {e}")

# Initialize retrievers
try:
    semantic_retriever = SemanticRetriever(db_client=db_client, llm_url=llm_url, model=model, k=5)
    lexical_retriever = LexicalRetriever(db_client=db_client)
except Exception as e:
    print(f"Error initializing retrievers: {e}")

# Initialize hybrid retriever
try:
    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
        k=5
    )
    
except Exception as e:
    print('TUKA')
    print(f"Error initializing hybrid retriever: {e}")

# Perform a search query
try:
    query = "What is alzheimers and is there any sort of connection between bananas and alzheimers?"
    results = hybrid_retriever.retrieve(query)

    # Print the results
    for result in results:
        print(f"Paper ID: {result.metadata['paper_id']}")
        print(f"Text: {result.page_content}")
        print()
except Exception as e:
    print(f"Error performing search query: {e}")