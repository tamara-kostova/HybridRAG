from typing import List, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantWrapper:
    def __init__(self, url: str = "localhost", api_key: str = "", 
                 collection_name: str = "selected_alzheimers_papers",
                 vector_size: int = 384):
        try:
            self.client = QdrantClient(url=url)
            self.collection_name = collection_name
            self.create_collection(vector_size=vector_size)
        except Exception as e:
            logger.error(f"Error initializing QdrantWrapper: {e}")
            raise

    def create_collection(self, vector_size: int = 384) -> None:
        try:
            existing_collections = self.client.get_collections()
            if self.collection_name in [
                c.name for c in existing_collections.collections
            ]:
                logger.info(
                    f"Collection {self.collection_name} already exists. Skipping creation."
                )
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            print(f"Collection: {self.collection_name} created")
            logger.info(f"Collection {self.collection_name} created successfully.")
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                logger.info(
                    f"Collection {self.collection_name} already exists. Skipping creation."
                )
            else:
                logger.error(f"Error creating collection: {e}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    def insert_paper(self, paper_name: str, chunks: List[str], embeddings) -> None:
        try:
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": chunk, "paper_name": paper_name, "chunk_index": i},
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                if embedding is not None
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            logger.error(f"Error inserting paper {paper_name}: {e}")

    def search(self, query_vector: List[float], limit: int = 5, timeout: int = 30):
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                timeout=timeout  
            )
            result_list = []
            for hit in results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", "No text available"),  
                    "paper_name": hit.payload.get("paper_name", "Unknown"),  
                    "chunk_index": hit.payload.get("chunk_index", -1), 
                }
                result_list.append(result)

            return result_list

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []


    def fetch(self, 
              id: Optional[uuid.UUID] = None, 
              paper_name: Optional[str] = None, 
              text: Optional[str] = None, 
              limit: int = 100,
              timeout: int = 60):
        try:
            scroll_filter = None
            if id:
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=id)
                        )
                    ]
                )
            if paper_name:
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="paper_name",
                            match=models.MatchValue(value=paper_name)
                        )
                    ]
                )
            if text:
                text_condition = models.FieldCondition(
                    key="text",
                    match=models.MatchValue(value=text)
                )
                if scroll_filter:
                    scroll_filter.must.append(text_condition)
                else:
                    scroll_filter = models.Filter(must=[text_condition])
            batch_size = min(100, limit)
            offset = None
            documents = []
        
            while len(documents) < limit:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                    timeout=timeout
                )
                
                if not results[0]:
                    break
                    
                for point in results[0]:
                    documents.append({
                        "id": point.id,
                        "paper_name": point.payload.get("paper_name"),
                        "text": point.payload.get("text"),
                    })
                    print(point)
                    if len(documents) >= limit:
                        break
                
                offset = results[1]
                
                if offset is None:
                    break
            
                return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []


    def fetch_all_papers(self, limit: int = 100):
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=0,
            )

            documents = []
            for point in results[0]:
                documents.append(
                    {
                        "id": point.id,
                        "paper_name": point.payload.get("paper_name"),
                        "text": point.payload.get("text"),
                    }
                )
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
        
    def fetch_all_documents_in_batches(self, batch_size: int = 1000):
        try:
            all_documents = []
            offset = None
            
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                
                batch_documents = results[0]
                next_offset = results[1]
                
                if not batch_documents:
                    break
                for point in batch_documents:
                    all_documents.append({
                        "id": point.id,
                        "paper_name": point.payload.get("paper_name"),
                        "paper_id": point.payload.get("paper_id", None),
                        "text": point.payload.get("text"),
                        "chunk_index": point.payload.get("chunk_index", -1)
                    })
                
                logger.info(f"Fetched batch of {len(batch_documents)} documents. Total so far: {len(all_documents)}")
                
                if next_offset is None:
                    break
                    
                offset = next_offset
                
            logger.info(f"Completed fetching all documents. Total documents: {len(all_documents)}")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error in batch fetching documents: {e}")
            return []

    def get_collection_info(self):
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return None

    def delete(self,
           id: Optional[str] = None,
           paper_name: Optional[str] = None,
           text: Optional[str] = None,
           timeout: int = 120):
        try:
            points_filter = None

            if id:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=[id]
                    )
                )
                logger.info(f"Deleted document with ID: {id}")
                return 1
                
            if paper_name:
                points_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="paper_name",
                            match=models.MatchValue(value=paper_name)
                        )
                    ]
                )
                
            if text:
                text_condition = models.FieldCondition(
                    key="text",
                    match=models.MatchValue(value=text)
                )
                if points_filter:
                    points_filter.must.append(text_condition)
                else:
                    points_filter = models.Filter(must=[text_condition])
            if points_filter:
                delete_result = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=points_filter
                    ),
                    wait=True
                )
                deleted_count = delete_result.status.get("deleted_count", 0)
                logger.info(f"Deleted approximately {deleted_count} document(s) matching the criteria")
                return deleted_count
                
            logger.warning("No deletion criteria provided")
            return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0