from typing import List
from hybridrag.lexical_retriever import LexicalRetriever
from hybridrag.semantic_retriever import SemanticRetriever
from langchain.schema import Document


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

    def retrieve(self, query: str) -> List[Document]:
        semantic_results = self.semantic_retriever.retrieve(query)
        lexical_results = self.lexical_retriever.retrieve(query)

        combined_results = {
            doc.metadata["paper_id"]: doc
            for doc in (semantic_results + lexical_results)
        }

        return list(combined_results.values())
