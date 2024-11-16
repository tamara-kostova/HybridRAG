from typing import List
import nest_asyncio
from llama_index.core.retrievers import QueryFusionRetriever
from langchain.schema import Document
from src.db.retrievers.retriever_lexical import LexicalRetriever
from src.db.db_client import QdrantWrapper

nest_asyncio.apply()

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

class HybridRetriever:
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        lexical_retriever: LexicalRetriever,
        k: int = 5,
    ):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.k = k
        self.retriever = QueryFusionRetriever(
            [
                self.semantic_retriever,
                self.lexical_retriever,
            ],
            num_queries=1,
            use_async=True,
        )

    def retrieve(self, query: str) -> List[Document]:
        results = self.retriever.retrieve(query)
        combined_results = {
            doc.metadata["paper_id"]: doc
            for doc in results
        }
        return list(combined_results.values())