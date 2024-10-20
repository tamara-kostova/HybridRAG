from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from src.db.db_client import QdrantWrapper


class LexicalRetriever:
    def __init__(self, db_client: QdrantWrapper):
        texts: List[Document] = self.fetch_texts_from_db(db_client)
        self.retriever = BM25Retriever.from_documents(texts)

    def fetch_texts_from_db(self, db_client: QdrantWrapper) -> List[Document]:
        fetched_documents = db_client.fetch_all_papers()
        documents = [
            Document(page_content=doc["text"], metadata={"paper_name": doc["paper_name"], "chunk_index": doc["chunk_index"]})
            for doc in fetched_documents
            if doc.get("text")
        ]
        return documents

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        return self.retriever.retrieve(query, top_k=top_k)
