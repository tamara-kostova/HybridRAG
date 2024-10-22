import os
from platform import processor
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Response

import requests

from api.deps import get_db_client, get_document_processor_ingest, get_scraper
from hybridrag.document_processor import DocumentProcessor
from src.db.models.search_result import SearchResult
from hybridrag.scraper import PubMedScraper
from hybridrag.document_processor_ingest import DocumentProcessorIngest
from src.db.db_client import QdrantWrapper
from src.db.models.query import Query


router = APIRouter()


@router.get("/")
def home():
    return {"health": "healthy"}


@router.get("/chat/v1/completions")
def chat_completion(prompt: str) -> Response:
    res = requests.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False,
        },
    )

    return Response(content=res.text, media_type="application/json")


@router.post("/insert")
async def ingest_papers(
    directory_path: str,
    db_client: QdrantWrapper = Depends(get_db_client),
):
    document_processor = DocumentProcessor(
        db_client=db_client, directory_path=directory_path
    )
    papers = document_processor.procces_directory()
    return {"message": f"Inserted {papers} papers"}


@router.post("/ingest")
async def ingest_papers(
    query: Query,
    scraper: PubMedScraper = Depends(get_scraper),
    db_client: QdrantWrapper = Depends(get_db_client),
    document_processor: DocumentProcessorIngest = Depends(
        get_document_processor_ingest
    ),
):
    papers = scraper.scrape_papers(query.text, max_results=10)
    for paper in papers:
        processed_paper = document_processor.process_paper(paper)
        if processed_paper:
            db_client.insert_paper(
                processed_paper["pmid"],
                processed_paper["chunks"],
                processed_paper["embeddings"],
            )
    return {"message": f"Ingested {len(papers)} papers"}


@router.post("/query", response_model=List[SearchResult])
async def query_endpoint(
    query: Query,
    db_client: QdrantWrapper = Depends(get_db_client),
    document_processor: DocumentProcessorIngest = Depends(
        get_document_processor_ingest
    ),
) -> List[SearchResult]:
    query_embedding: List[float] = document_processor.embed_text(query.text)
    results: List[Dict[str, Any]] = db_client.search(query_embedding)
    return [
        SearchResult(
            id=result["id"],
            score=result["score"],
            text=result["text"],
            paper_id=result["paper_id"],
            chunk_index=result["chunk_index"],
        )
        for result in results
    ]
