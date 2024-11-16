from typing import List
import nest_asyncio
from llama_index.core.retrievers import QueryFusionRetriever
from langchain.schema import Document

nest_asyncio.apply()

class HybridRetriever:
    def __init__(
        self,
        semantic_retriever,
        lexical_retriever,
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
            llm=None  # Disable the use of OpenAI LLM
        )

    def retrieve(self, query: str) -> List[Document]:
        results = self.retriever.retrieve(query)
        combined_results = {
            doc.metadata["paper_id"]: doc
            for doc in results
        }
        return list(combined_results.values())