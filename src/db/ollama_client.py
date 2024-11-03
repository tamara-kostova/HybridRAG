import requests
from typing import List, Optional

class OllamaClient:
    def __init__(self, host: str = "localhost", port: int = 11434, endpoint: str = "/"):
        """Initialize Ollama client with the host, port, and endpoint of the service."""
        self.url = f"http://{host}:{port}{endpoint}"

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates embeddings for the given text by sending a request to the Ollama service.
        
        Parameters:
            text (str): The input text for which to generate embeddings.

        Returns:
            List[float]: The embeddings if successful; otherwise, returns None.
        """
        payload = {"model": "llama", "input": text}
        try:
            print(f"Connecting to Ollama at {self.url}")
            response = requests.post(self.url, json=payload, timeout=5)
            response.raise_for_status()  # Ensure we catch HTTP errors

            # Attempt to parse the JSON response
            data = response.json()
            
            # Check if 'embedding' key exists in the response
            if "embedding" in data and isinstance(data["embedding"], list):
                print("Embedding generated successfully.")
                return data["embedding"]
            else:
                print("Error: Unexpected response format. 'embedding' key not found.")
                return None  # Return None if the structure is unexpected

        except requests.ConnectionError:
            print(f"Connection error: Failed to reach Ollama server at {self.url}. Ensure the server is running.")
            return None
        except requests.Timeout:
            print("Timeout error: The request to Ollama server timed out.")
            return None
        except requests.JSONDecodeError:
            print("Error: Failed to decode JSON response from Ollama server.")
            return None
        except requests.RequestException as e:
            print(f"Request error: An error occurred - {e}")
            return None


if __name__ == "__main__":
    
    client = OllamaClient(endpoint="/your_actual_endpoint_here")
    text = "Neuroscience research on memory formation"
    embedding = client.generate_embedding(text)
    if embedding:
        print("Embedding:", embedding)
    else:
        print("Failed to retrieve embedding.")
