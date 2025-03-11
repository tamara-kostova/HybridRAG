import os
from typing import List
from src.db.db_client import QdrantWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(
        self,
        db_client: QdrantWrapper,
        model_name: str = "all-MiniLM-L6-v2",
        directory_path: str = "",
    ):
        try:
            self.model = SentenceTransformer(model_name)
            self.directory_path = directory_path
            self.db_client = db_client
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {e}")
            raise

    def procces_directory(self):
        file_paths = os.listdir(self.directory_path)
        inserted_count = 0
        for file_path in file_paths:
            pdf_path = os.path.join(self.directory_path, file_path)
            if pdf_path.lower().endswith(".pdf"):
                self.process_pdf(pdf_path)
                inserted_count += 1
        return inserted_count

    def process_pdf(self, pdf_path: str):
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            full_text = "".join([doc.page_content for doc in documents])
            chunks = self.chunk_text(full_text)
            embeddings = [self.embed_text(chunk) for chunk in chunks]
            valid_chunks_and_embeddings = [
                (chunk, emb)
                for chunk, emb in zip(chunks, embeddings)
                if emb is not None and len(emb) > 0
            ]
            pdf_filename = os.path.basename(pdf_path)
            if valid_chunks_and_embeddings:
                self.db_client.insert_paper(
                    paper_name=pdf_filename,
                    chunks=[chunk for chunk, emb in valid_chunks_and_embeddings],
                    embeddings=[emb for chunk, emb in valid_chunks_and_embeddings],
                )
                logger.info(
                    f"Processed and inserted PDF {pdf_filename} into the database."
                )
            else:
                logger.warning(f"No valid chunks or embeddings for PDF {pdf_filename}.")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")

    def chunk_text(self, text: str, max_chunk_size=500) -> List[str]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size, chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []

    def embed_text(self, text: str):
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m testing.check_cluster <QDRANT_HOST> <QDRANT_API_KEY>")
        sys.exit(1)

    qdrant_host = sys.argv[1]
    qdrant_api_key = sys.argv[2]
    db_client = QdrantWrapper(
        url=qdrant_host, api_key=qdrant_api_key
    )    
    dp = DocumentProcessor(db_client=db_client, directory_path="selected_pdfs")
    dp.procces_directory()
