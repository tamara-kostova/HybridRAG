from pydantic import BaseModel


class SearchResult(BaseModel):
    id: str
    score: float
    text: str