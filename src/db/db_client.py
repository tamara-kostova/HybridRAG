from typing import List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantWrapper:
    def __init__(self, url: str = "localhost", port: int = 6333, api_key:str = ""):
        self.client = QdrantClient(url=url, port=port, api_key=api_key)
        self.collection_name = "neurology_papers"

    def create_collection(self, vector_size: int = 384) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def insert_paper(self, paper_id: str, chunks: List[str], embeddings) -> None:
        print(f"Paper id {paper_id}")
        print(f"Chunks {chunks}")
        print(f"Embeddings {embeddings}")
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": chunk, "paper_id": paper_id}
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        print(f"Points {points}")
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], limit:int =5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return [{"id": hit.id, "score": hit.score, "text": hit.payload["text"]} for hit in results]