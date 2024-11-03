import numpy as np
from typing import List, Dict, Any
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantWrapper:
    def __init__(self, url: str = "localhost", api_key: str = "", timeout: float = 30.0):
        try:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
            self.collection_name = "alzheimers_papers"
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
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            logger.info(f"Collection {self.collection_name} created successfully.")
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                logger.info(f"Collection {self.collection_name} already exists. Skipping creation.")
            else:
                logger.error(f"Error creating collection: {e}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    def insert_paper(self, paper_name: str, chunks: List[str], embeddings, token_frequencies: List[Dict[str, int]]) -> None:
        try:
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": chunk, "paper_name": paper_name, "chunk_index": i, "tf_idf": tf_idf},
                )
                for i, (chunk, embedding, tf_idf) in enumerate(zip(chunks, embeddings, token_frequencies))
                if embedding is not None
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            logger.error(f"Error inserting paper {paper_name}: {e}")

    def search_dense(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "paper_id": hit.payload.get("paper_name"),
                    "chunk_index": hit.payload["chunk_index"],
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []

    def search_sparse(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,  
                limit=limit,
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "paper_id": hit.payload.get("paper_name"),
                    "chunk_index": hit.payload["chunk_index"],
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    def hybrid_search(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        dense_results = self.search_dense(query_vector, limit=limit)
        sparse_results = self.search_sparse(query_vector, limit=limit)  

        combined_results = {}
        
        for result in dense_results + sparse_results:
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = result
            else:
                combined_results[doc_id]["score"] = (
                    combined_results[doc_id]["score"] + result["score"]
                ) / 2

        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)

        return sorted_results[:limit]
