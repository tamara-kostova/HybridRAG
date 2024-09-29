from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_db(documents, embeddings: Embeddings):
    db = Chroma.from_documents(documents=documents, embeddings=embeddings)
    return db


def get_answer(db, query: str):
    similar_docs = db.similarity_search(query, k=3)
    pass
