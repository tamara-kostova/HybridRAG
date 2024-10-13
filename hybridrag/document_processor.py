from typing import List
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {e}")
            raise

    def chunk_text(self, text, max_chunk_size=1000) -> List[str]:
        try:
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0

            for word in words:
                if current_size + len(word) > max_chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []

    def embed_text(self, text):
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    def process_paper(self, paper):
        try:
            full_text = f"{paper['title']} {paper['abstract']}"
            if not full_text.strip():
                logger.warning(f"Empty text for paper {paper['pmid']}")
                return None
            chunks = self.chunk_text(full_text)
            if not chunks:
                logger.warning(f"No chunks generated for paper {paper['pmid']}")
                return None
            embeddings = [self.embed_text(chunk) for chunk in chunks]
            valid_chunks_and_embeddings = [
                (chunk, emb)
                for chunk, emb in zip(chunks, embeddings)
                if emb is not None and len(emb) > 0
            ]

            if not valid_chunks_and_embeddings:
                logger.warning(
                    f"No valid embeddings generated for paper {paper['pmid']}"
                )
                return None

            return {
                "pmid": paper["pmid"],
                "chunks": [chunk for chunk, _ in valid_chunks_and_embeddings],
                "embeddings": [emb for _, emb in valid_chunks_and_embeddings],
            }
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            return None
