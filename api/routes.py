import os
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Form, Request, Response

import requests

from api.deps import get_db_client, get_document_processor_ingest, get_scraper
from hybridrag.document_processors.document_processor import DocumentProcessor
from api.utils import format_message, get_session_history
from src.db.models.search_result import SearchResult
from hybridrag.document_processors.scraper import PubMedScraper
from hybridrag.document_processors.document_processor_ingest import DocumentProcessorIngest
from src.db.db_client import QdrantWrapper
from src.db.models.query import Query
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()


@router.get("/")
def home():
    return {"health": "healthy"}


@router.post("/generate")
def chat_completion(request: Request, question: str = Form(...)) -> Response:
    session_id = request.client.host
    chat_history = get_session_history(session_id)
    with open("hybridrag/instructions.txt", "r") as f:
        instructions = f.read()
    message = format_message(instructions, question, chat_history)
    chat_history.add_message(HumanMessage(content=question))
    try:
        llm_ip_address = os.getenv("LLM_IP_ADDRESS")
        res = requests.post(
            url=f"http://{llm_ip_address}:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": message,
                "stream": False,
            },
        )
        response_text = res.text
        chat_history.add_message(AIMessage(content=response_text))
        return Response(content=response_text, media_type="application/json")
    except Exception as e:
        return Response(
            content=f'{{"message": "Failed to generate a response", "description": "{e}"}}',
            media_type="application/json",
            status_code=500,
        )


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
