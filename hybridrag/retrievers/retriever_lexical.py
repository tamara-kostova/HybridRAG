from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from src.db.db_client import QdrantWrapper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexicalRetriever:
    def __init__(self, db_client: QdrantWrapper):
        texts: List[Document] = self.fetch_texts_from_db(db_client)
        if not texts:
            raise ValueError("No documents retrieved from the database.")
        self.retriever = BM25Retriever.from_documents(texts)

    def fetch_texts_from_db(self, db_client: QdrantWrapper) -> List[Document]:
        fetched_documents = db_client.fetch_all_papers()
        if not fetched_documents:
            logger.info("No documents fetched from the database.")
        documents = [
            Document(
                page_content=doc["text"],
                metadata={
                    "paper_name": doc.get("paper_name", "Unknown"),
                    "chunk_index": doc.get("chunk_index", 0),
                },
            )
            for doc in fetched_documents
            if doc.get("text")
        ]
        if not documents:
            logger.info("No valid documents found.")
        return documents

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        results = self.retriever.invoke(query, top_k=top_k)
        return results
