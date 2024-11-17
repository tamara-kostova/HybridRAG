import requests
import json
from typing import List
from src.db.db_client import QdrantWrapper
from langchain.schema import Document

class SemanticRetriever:
    def __init__(self, db_client: QdrantWrapper, llm_url: str, model: str, k: int = 5):
        self.db_client = db_client
        self.llm_url = llm_url
        self.model = model
        self.k = k

    def generate_embedding(self, text: str, max_tokens: int = 1) -> List[float]:
        """
        Generates an embedding for the given text using the specified LLM API.
        """
        try:
            # Send the request to generate the embedding
            response = requests.post(
                f"{self.llm_url}/api/generate",
                json={"model": self.model, "prompt": text, "max_tokens": max_tokens},
            )

            response.raise_for_status()

            # Decode the response content
            
            response_text = response.content.decode('utf-8')
            print(f"Decoded Response Content: {response_text}")  # Debug print

            # Try to parse the response content
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                print(f"JSON decoding error: {json_err}")
                print(f"Raw response content: {response_text}")
                raise

            # Check if the "embedding" key is present in the response
            if "embedding" not in response_data:
                raise ValueError(
                    f"'embedding' key not found in the response. Response data: {response_data}"
                )

            # Return the embedding (list of floats)
            return response_data["embedding"]

        except requests.exceptions.RequestException as req_err:
            print(f"HTTP request error: {req_err}")
            raise
        except ValueError as val_err:
            print(f"Error in response format: {val_err}")
            raise

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves documents from the database using a hybrid approach.
        """
        print("Generating embedding for query...")
        query_embedding = self.generate_embedding(query)
        print("Embedding generated successfully.")

        # Search the database with the generated embedding
        results = self.db_client.search(query_vector=query_embedding, limit=self.k)

        # Create and return a list of Document objects from the results
        return [
            Document(
                page_content=res.get("text", "No text available"),
                metadata={"paper_name": res.get("paper_name", "Unknown")},
            )
            for res in results
        ]
