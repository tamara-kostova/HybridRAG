import sys
import os
import requests
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), "..", "secrets.env"))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from src.db.db_client import QdrantWrapper
from hybridrag.retrievers.retriever_semantic import SemanticRetriever
from hybridrag.retrievers.retriever_lexical import LexicalRetriever
from hybridrag.retrievers.retriever_hybrid import HybridRetriever


def test_hybrid_retriever(qdrant_host: str, qdrant_api_key: str):
    llm_url = "http://localhost:11434/v1/chat/completions"
    model = "llama3:8b"

    try:
        qdrant_client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        db_client = QdrantWrapper(url=qdrant_host, api_key=qdrant_api_key)
        collections = qdrant_client.get_collections()
        print(f"Collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        sys.exit(1)

    try:
        semantic_retriever = SemanticRetriever(db_client=db_client)
        print("Semantic retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing semantic retriever: {e}")
        sys.exit(1)

    try:
        lexical_retriever = LexicalRetriever(db_client=db_client)
        print("Lexical retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing lexical retriever: {e}")
        sys.exit(1)

    try:
        hybrid_retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            k=10,
        )
        print("Hybrid retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing hybrid retriever: {e}")
        sys.exit(1)

    try:
        print("Performing search query...")
        query = "What is Alzheimer's, and is there any sort of connection between bananas and Alzheimer's?"
        results = hybrid_retriever.retrieve(query)

        context = "\n".join([result.page_content for result in results])

        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        print(f"Sending prompt to LLM: {prompt}")

        response = requests.post(
            llm_url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            llm_answer = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No answer generated.")
            )
            print(f"Answer from LLM: {llm_answer}")
        else:
            print(f"Error querying LLM: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error performing search query: {e}")


# instructions to run inside docker container:
# sudo docker exec -it hybridrag-app-1 bash
# python3 -m testing.test_local <url> <api_key>
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <QDRANT_HOST> <QDRANT_API_KEY>")
        sys.exit(1)
    qdrant_host = sys.argv[1]
    qdrant_api_key = sys.argv[2]
    test_hybrid_retriever(qdrant_host, qdrant_api_key)
