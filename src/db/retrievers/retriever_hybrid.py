from typing import List
import nest_asyncio
from llama_index.core.retrievers import QueryFusionRetriever
from langchain.schema import Document
from src.db.retrievers.retriever_lexical import LexicalRetriever
from src.db.retrievers.retriever_semantic import SemanticRetriever
nest_asyncio.apply()

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
                self.semantic_retriever.retrieve(similarity_top_k=k),
                self.lexical_retriever.retrieve(similarity_top_k=k),
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