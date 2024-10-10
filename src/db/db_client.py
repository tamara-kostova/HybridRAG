from typing import List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantWrapper:
    def __init__(self, url: str = "localhost", api_key:str = ""):
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.collection_name = "neurology_papers"
            self.create_collection()
        except Exception as e:
            logger.error(f"Error initializing QdrantWrapper: {e}")
            raise

    def create_collection(self, vector_size: int = 384) -> None:
        try:
            existing_collections = self.client.get_collections()
            if self.collection_name in [c.name for c in existing_collections.collections]:
                logger.info(f"Collection {self.collection_name} already exists. Skipping creation.")
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logger.info(f"Collection {self.collection_name} created successfully.")
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                logger.info(f"Collection {self.collection_name} already exists. Skipping creation.")
            else:
                logger.error(f"Error creating collection: {e}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    def insert_paper(self, paper_id: str, chunks: List[str], embeddings) -> None:
        try:
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": chunk, "paper_id": paper_id}
                )
                for chunk, embedding in zip(chunks, embeddings) if embedding is not None
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            logger.error(f"Error inserting paper {paper_id}: {e}")

    def search(self, query_vector: List[float], limit:int =5):
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [{"id": hit.id, "score": hit.score, "text": hit.payload["text"]} for hit in results]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
        
    def fetch_all_papers(self, limit: int = 100):
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            documents = []
            for point in results[0]:
                documents.append({
                    "id": point.id,
                    "paper_id": point.payload.get("paper_id"),
                    "text": point.payload.get("text")
                })
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
        
    def get_collection_info(self):
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return None