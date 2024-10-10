from typing import List
from pydantic import BaseModel


class ProcessedPaper(BaseModel):
    pmid: str
    chunks: List[str]
    embeddings: List[List[float]]
