from typing import List
import nest_asyncio
from src.db.retrievers.local_fusion import LocalQueryFusionRetriever  # Import the correct class
from langchain.schema import Document

nest_asyncio.apply()

class HybridRetriever:
    def __init__(
        self,
        semantic_retriever,
        lexical_retriever,
        local_llm_callable,  # Function to call your local LLM
        k: int = 5,
    ):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.k = k

        # Initialize LocalQueryFusionRetriever with the local LLM callable
        self.retriever = LocalQueryFusionRetriever(
            retrievers=[
                self.semantic_retriever,
                self.lexical_retriever,
            ],
            local_llm=local_llm_callable,  # Pass the local LLM callable
            num_queries=1,  # Generate one query variation
            use_async=True,
            verbose=True,  # Enable verbose mode for debugging
        )

    def retrieve(self, query: str) -> List[Document]:
        results = self.retriever.retrieve(query)
        # Combine results while avoiding duplicates
        combined_results = {
            doc.metadata["paper_name"]: doc
            for doc in results
        }
        return list(combined_results.values())