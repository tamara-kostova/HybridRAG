from typing import Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    paper_id: Optional[str] = None
    chunk_index: int
