import requests
from typing import List
from src.db.db_client import QdrantWrapper
from langchain.schema import Document

class SemanticRetriever:
    def __init__(self, db_client: QdrantWrapper, llm_url: str, model: str, k: int = 5):
        self.db_client = db_client
        self.llm_url = llm_url
        self.model = model
        self.k = k

    def generate_embedding(self, text: str) -> List[float]:
        # Ensure the local model endpoint is correct
        response = requests.post(
            f"{self.llm_url}/api/generate",  # Replace with your local model's API
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def retrieve(self, query: str) -> List[Document]:
        query_embedding = self.generate_embedding(query)
        results = self.db_client.search(query_vector=query_embedding, limit=self.k)
        return [
            Document(
                page_content=res["text"], metadata={"paper_name": res["paper_name"]}
            )
            for res in results
        ]