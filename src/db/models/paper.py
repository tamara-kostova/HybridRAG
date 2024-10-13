from pydantic import BaseModel


class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
