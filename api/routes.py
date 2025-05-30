import os
from typing import Any, Dict, List
import uuid
from fastapi import APIRouter, Depends, Form, Request, Response

import requests

from api.deps import (
    get_db_client,
    get_document_processor_ingest,
    get_hybrid_retriever,
    get_multihop_retriever,
    get_scraper,
)
from hybridrag.document_processors.document_processor import DocumentProcessor
from api.utils import format_message, get_session_history
from hybridrag.retrievers.multihop import MultiHop
from hybridrag.retrievers.retriever_hybrid import HybridRetriever
from src.db.models.search_result import SearchResult
from hybridrag.document_processors.scraper import PubMedScraper
from hybridrag.document_processors.document_processor_ingest import (
    DocumentProcessorIngest,
)
from src.db.db_client import QdrantWrapper
from src.db.models.query import Query
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()


@router.get("/")
def home():
    return {"health": "healthy"}


@router.post("/generate")
def chat_completion(
    request: Request,
    question: str = Form(...),
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> Response:

    session_id = request.client.host
    chat_history = get_session_history(session_id)
    with open("hybridrag/instructions.txt", "r") as f:
        instructions = f.read()
    chat_history.add_message(HumanMessage(content=question))

    retrieved_documents = hybrid_retriever.retrieve(question)
    retrieved_context = "\n\n".join(
        [
            f"Document {i+1}: {doc.page_content}"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    enhanced_message = f"""
    Context for answering the question:
    {retrieved_context}

    {format_message(instructions, question, chat_history)}
    """

    try:
        llm_ip_address = os.getenv("LLM_IP_ADDRESS")
        res = requests.post(
            url=f"http://{llm_ip_address}:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": enhanced_message,
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


@router.post("/multihop", response_model=List[SearchResult])
async def query_endpoint(
    query: str,
    multihop_retriever: MultiHop = Depends(get_multihop_retriever),
) -> List[SearchResult]:

    result = multihop_retriever.forward(query)

    search_results = []
    for idx, context_passage in enumerate(result.context):
        search_results.append(
            SearchResult(
                id=str(uuid.uuid4()),
                score=1.0,
                text=context_passage,
                paper_id=None,
                chunk_index=idx,
            )
        )

    return search_results


@router.post("/insert")
async def insert_papers(
    directory_path: str,
    db_client: QdrantWrapper = Depends(get_db_client),
):
    document_processor = DocumentProcessor(
        db_client=db_client, directory_path=directory_path
    )
    papers = document_processor.procces_directory()
    return {"message": f"Inserted {papers} papers"}


@router.post("/old/ingest")
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
