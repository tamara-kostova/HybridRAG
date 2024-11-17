from typing import List
import nest_asyncio
from src.db.retrievers.local_fusion import LocalQueryFusionRetriever  # Import the correct class
from langchain.schema import Document
from llama_index.core.schema import QueryBundle

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
            local_llm_callable=local_llm_callable,  # Pass the local LLM callable
            num_queries=1,  # Generate one query variation
            use_async=True,
            verbose=True,  # Enable verbose mode for debugging
        )

    def retrieve(self, query: str) -> List[Document]:
        # Check if query is a string, and wrap it into a QueryBundle if necessary
        if isinstance(query, str):
            query = QueryBundle(query)  # Wrap the string into QueryBundle
        print('ovde')
        results = self.retriever.retrieve(query)  # Call the retriever's retrieve method
        print('POSLE RESULTS')

        # Combine results while avoiding duplicates based on "paper_name"
        combined_results = {
            doc.metadata["paper_name"]: doc
            for doc in results
        }
        print('TUKA')

        return list(combined_results.values())