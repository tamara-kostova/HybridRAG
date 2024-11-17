from typing import List
from src.db.retrievers.retriever_lexical import LexicalRetriever 
from src.db.retrievers.retriever_semantic import SemanticRetriever
from langchain.schema import Document

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
        print('Performing semantic retrieval...')
        semantic_results = self.semantic_retriever.retrieve(query)
        print('Semantic retrieval completed.')
        print(len(semantic_results))
        print(semantic_results)
        print('Performing lexical retrieval...')
        lexical_results = self.lexical_retriever.retrieve(query)
        print('Lexical retrieval completed.')

        combined_results = {
            doc.metadata["paper_name"]: doc
            for doc in (semantic_results + lexical_results)
        }

        return list(combined_results.values())