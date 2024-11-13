from fastapi import Request

from hybridrag.src.document_processors.document_processor_ingest import DocumentProcessorIngest
from hybridrag.src.document_processors.scraper import PubMedScraper
from src.db.db_client import QdrantWrapper
from hybridrag.src.graph.extractor import NodeExtractor

def get_db_client(request: Request) -> QdrantWrapper:
    return request.app.state.db_client  # type: ignore


def get_document_processor_ingest(request: Request) -> DocumentProcessorIngest:
    return request.app.state.document_processor_ingest  # type: ignore


def get_scraper(request: Request) -> PubMedScraper:
    return request.app.state.scraper  # type: ignore

def get_graph(request: Request) -> NodeExtractor: 
    return request.app.state.graph # type: ignore 