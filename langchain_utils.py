from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral
from langchain_core.documents import Document


def get_document_splits(web_paths) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=(web_paths,),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits


def get_chunks(text: str, chunk_size: int):
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def get_text_embedding(client: Mistral, input):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed", inputs=input
    )
    return embeddings_batch_response.data[0].embedding
