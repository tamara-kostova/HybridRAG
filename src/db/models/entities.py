from typing import List
from pydantic import BaseModel, Field

class Entities(BaseModel):
    ids: List[str] = Field(
        ...,
        description = "The names (ID fields) of all entities appearing in the text."
    )
