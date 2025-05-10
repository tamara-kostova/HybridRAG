from fastapi import Request

from hybridrag.document_processors.document_processor_ingest import DocumentProcessorIngest
from hybridrag.document_processors.scraper import PubMedScraper
from hybridrag.retrievers.multihop import MultiHop
from hybridrag.retrievers.retriever_hybrid import HybridRetriever
from hybridrag.retrievers.retriever_lexical import LexicalRetriever
from hybridrag.retrievers.retriever_semantic import SemanticRetriever
from src.db.db_client import QdrantWrapper
from hybridrag.graph.extractor import NodeExtractor

def get_db_client(request: Request) -> QdrantWrapper:
    return request.app.state.db_client  # type: ignore


def get_document_processor_ingest(request: Request) -> DocumentProcessorIngest:
    return request.app.state.document_processor_ingest  # type: ignore


def get_scraper(request: Request) -> PubMedScraper:
    return request.app.state.scraper  # type: ignore

def get_graph(request: Request) -> NodeExtractor:
    return request.app.state.graph # type: ignore

def get_lexical_retriever(request: Request) -> LexicalRetriever:
    return request.app.state.lexical_retriever # type: ignore

def get_semantic_retriever(request: Request) -> SemanticRetriever:
    return request.app.state.semantic_retriever # type: ignore

def get_hybrid_retriever(request: Request) -> HybridRetriever:
    return request.app.state.hybrid_retriever # type: ignore

def get_multihop_retriever(request: Request) -> MultiHop:
    return request.app.state.multihop_retriever # type: ignore