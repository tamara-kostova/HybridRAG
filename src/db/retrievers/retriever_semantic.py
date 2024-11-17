from typing import List
from src.db.db_client import QdrantWrapper
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

class SemanticRetriever:
    def __init__(self, db_client: QdrantWrapper, model_name: str = "all-MiniLM-L6-v2", k: int = 10):

        self.db_client = db_client
        self.model = SentenceTransformer(model_name)  
        self.k = k

    def generate_embedding(self, text: str) -> List[float]:

        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def retrieve(self, query: str) -> List[Document]:


        query_embedding = self.generate_embedding(query)
        


        if not all(isinstance(x, float) for x in query_embedding):
            query_embedding = [float(x) for x in query_embedding]

        # Search the database with the generated query embedding
        results = self.db_client.search(query_vector=query_embedding, limit=self.k)

        # Return a list of Document objects from the results
        return [
            Document(
                page_content=res.get("text", "No text available"),
                metadata={"paper_name": res.get("paper_name", "Unknown")},
            )
            for res in results
        ]

