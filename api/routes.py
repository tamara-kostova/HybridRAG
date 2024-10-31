import os
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Form, Request, Response

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from api.deps import get_db_client, get_document_processor_ingest, get_scraper
from api.utils import format_message, get_session_history
from hybridrag.document_processor import DocumentProcessor
from src.db.models.search_result import SearchResult
from hybridrag.scraper import PubMedScraper
from hybridrag.document_processor_ingest import DocumentProcessorIngest
from src.db.db_client import QdrantWrapper
from src.db.models.query import Query
from langchain_core.messages import HumanMessage, AIMessage


router = APIRouter()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@router.get("/")
def home():
    return {"health": "healthy"}

bm25_retriever = BM25Retriever(docs=[])

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

@router.on_event("startup")
async def startup_event():

    db_client = QdrantWrapper(
        url="https://7ecf0b14-c826-4ae4-b61b-3bd710fc75d9.europe-west3-0.gcp.cloud.qdrant.io", 
        api_key="agxIHD5sPk-2svMtUPmn26Gf3CHZLhmidbz-eOQuOjjushtYCl9aVQ"
    )


    documents = db_client.fetch_all_papers() 


    transformed_documents = [
        Document(page_content=doc["text"], metadata=doc.get("metadata", {})) 
        for doc in documents
    ]


    global bm25_retriever  
    if transformed_documents:
        bm25_retriever = BM25Retriever.from_documents(transformed_documents)  

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
    document_processor: DocumentProcessorIngest = Depends(get_document_processor_ingest),
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
    document_processor: DocumentProcessorIngest = Depends(get_document_processor_ingest),
) -> List[SearchResult]:
    query_embedding: List[float] = document_processor.embed_text(query.text)

    dense_results: List[Dict[str, Any]] = db_client.search(query_embedding, limit=10)

    sparse_results: List[Dict[str, Any]] = bm25_retriever.get_relevant_documents(query.text)

    combined_results = {res["id"]: res for res in dense_results + sparse_results}.values()

    return [
        SearchResult(
            id=result["id"],
            score=result["score"],
            text=result["text"],
            paper_id=result.get("paper_id"),
            chunk_index=result["chunk_index"],
        )
        for result in combined_results
    ]
