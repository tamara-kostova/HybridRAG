from typing import List
from langchain.schema import Document
import logging

from hybridrag.retrievers.retriever_lexical import LexicalRetriever
from hybridrag.retrievers.retriever_semantic import SemanticRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        lexical_retriever: LexicalRetriever,
        k: int = 10,
    ):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        logger.info("Performing semantic retrieval...")
        semantic_results = self.semantic_retriever.retrieve(query)[: self.k]
        logger.info("Semantic retrieval completed.")
        logger.info(len(semantic_results), "results")

        logger.info("Performing lexical retrieval...")
        lexical_results = self.lexical_retriever.retrieve(query)[: self.k]
        logger.info("Lexical retrieval completed.")
        logger.info(len(lexical_results), "results")

        combined_results = {
            (doc.metadata["paper_name"], doc.metadata.get("chunk_index", 0)): doc
            for doc in (semantic_results + lexical_results)
        }

        return list(combined_results.values())
