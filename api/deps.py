from fastapi import Request

from hybridrag.document_processor import DocumentProcessor
from hybridrag.scraper import PubMedScraper
from src.db.db_client import QdrantWrapper


def get_db_client(request: Request) -> QdrantWrapper:
    return request.app.state.db_client  # type: ignore


def get_document_processor(request: Request) -> DocumentProcessor:
    return request.app.state.document_processor  # type: ignore


def get_scraper(request: Request) -> PubMedScraper:
    return request.app.state.scraper  # type: ignore
