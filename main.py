from contextlib import asynccontextmanager
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from api import routes

from hybridrag.src.document_processors.document_processor_ingest import DocumentProcessorIngest
from src.db.db_client import QdrantWrapper
from hybridrag.src.document_processors.scraper import PubMedScraper
from hybridrag.src.graph.extractor import NodeExtractor

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("src/db/config.json") as f:
        config = json.load(f)
    app.state.scraper = PubMedScraper()
    app.state.document_processor_ingest = DocumentProcessorIngest()
    app.state.db_client = QdrantWrapper(
        url=config["QDRANT_HOST"], api_key=config["QDRANT_API_KEY"]
    )
    app.state.graph = NodeExtractor(url=config["NEO4J_URI"], username =config["NEO4J_USERNAME"], password=config["NEO4J_PASSWORD"])
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
