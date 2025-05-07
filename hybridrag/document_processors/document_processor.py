import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging

from src.db.db_client import QdrantWrapper

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
        inserted_count = 0
        logger.info(f"Walking through directory: {self.directory_path}")

        for root, _, files in os.walk(self.directory_path):
            logger.info(f"In directory: {root}")
            logger.info(f"Files found: {files}")

            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    logger.info(f"Found PDF: {pdf_path}")
                    try:
                        self.process_pdf(pdf_path)
                        inserted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to process {pdf_path}: {e}")

        logger.info(f"Total inserted: {inserted_count}")
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
                    chunks=[chunk for chunk, _ in valid_chunks_and_embeddings],
                    embeddings=[emb for _, emb in valid_chunks_and_embeddings],
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
