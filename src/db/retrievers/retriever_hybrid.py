from typing import List
import nest_asyncio
from src.db.retrievers.local_fusion import LocalQueryFusionRetriever  # Import the new class
from langchain.schema import Document

nest_asyncio.apply()

class HybridRetriever:
    def __init__(
        self,
        semantic_retriever,
        lexical_retriever,
        llm_url: str,
        model: str,
        k: int = 5,
    ):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.k = k
        self.retriever = LocalQueryFusionRetriever(
            retrievers=[
                self.semantic_retriever,
                self.lexical_retriever,
            ],
            llm_url=llm_url,  # Use your local LLM
            model=model,  # Use your local LLM model
            num_queries=1,
            use_async=True,
            verbose=True,  # Enable verbose for debugging if needed
        )

    def retrieve(self, query: str) -> List[Document]:
        results = self.retriever.retrieve(query)
        # Combine results while avoiding duplicates
        combined_results = {
            doc.metadata["paper_name"]: doc
            for doc in results
        }
        return list(combined_results.values())