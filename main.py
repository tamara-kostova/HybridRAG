from contextlib import asynccontextmanager
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from api import routes

from hybridrag.document_processor_ingest import DocumentProcessorIngest
from hybridrag.hybrid_retriever import HybridRetriever
from hybridrag.lexical_retriever import LexicalRetriever
from hybridrag.semantic_retriever import SemanticRetriever
from src.db.db_client import QdrantWrapper
from hybridrag.scraper import PubMedScraper


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("src/db/config.json") as f:
        config = json.load(f)
    app.state.scraper = PubMedScraper()
    app.state.document_processor_ingest = DocumentProcessorIngest()
    app.state.db_client = QdrantWrapper(
        url=config["QDRANT_HOST"], api_key=config["QDRANT_API_KEY"]
    )
    app.state.lexical_retriever = LexicalRetriever(db_client=app.state.db_client)
    app.state.semantic_retriever = SemanticRetriever(db_client=app.state.db_client)
    app.state.hybrid_retriever = HybridRetriever(
        semantic_retriever=app.state.semantic_retriever,
        lexical_retriever=app.state.lexical_retriever,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.include_router(routes.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
