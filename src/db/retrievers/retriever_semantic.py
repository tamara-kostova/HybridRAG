from typing import List
from src.db.db_client import QdrantWrapper
from langchain.schema import Document


class SemanticRetriever:
    def __init__(self, db_client: QdrantWrapper, k: int = 5):
        self.db_client = db_client
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        query_embedding = self.db_client.model.encode(query)
        results = self.db_client.search(query_vector=query_embedding, limit=self.k)
        return [
            Document(
                page_content=res["text"], metadata={"paper_name": res["paper_name"]}
            )
            for res in results
        ]