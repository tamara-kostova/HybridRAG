import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'secrets.env'))

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client import QdrantClient
from src.db.db_client import QdrantWrapper
from src.db.retrievers.retriever_semantic import SemanticRetriever
from src.db.retrievers.retriever_lexical import LexicalRetriever
from src.db.retrievers.retriever_hybrid import HybridRetriever

def test_hybrid_retriever():
    # Retrieve configuration from environment variables
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    llm_url = "http://localhost:11434"  # Local model server
    model = "llama3:8b"

    # Initialize Qdrant client and wrapper
    try:
        qdrant_client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        db_client = QdrantWrapper(url=qdrant_host, api_key=qdrant_api_key)
        collections = qdrant_client.get_collections()
        print(f"Collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        sys.exit(1)

    # Initialize semantic retriever
    try:
        semantic_retriever = SemanticRetriever(
            db_client=db_client
        )
        print("Semantic retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing semantic retriever: {e}")
        sys.exit(1)

    # Initialize lexical retriever
    try:
        lexical_retriever = LexicalRetriever(db_client=db_client)
        print("Lexical retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing lexical retriever: {e}")
        sys.exit(1)

    # Initialize hybrid retriever
    try:
        hybrid_retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            k=5
        )
        print("Hybrid retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing hybrid retriever: {e}")
        sys.exit(1)

    # Perform a search query
    try:
        print('Performing search query...')
        query = "What is Alzheimer's, and is there any sort of connection between bananas and Alzheimer's?"
        results = hybrid_retriever.retrieve(query)
        
        # Print the results
        if results:
            for result in results:
                print(f"Paper Name: {result.metadata.get('paper_name', 'Unknown')}")
                print(f"Text: {result.page_content}")
                print("-" * 50)
        else:
            print("No results found.")
    except Exception as e:
        print(f"Error performing search query: {e}")

if __name__ == "__main__":
    test_hybrid_retriever()