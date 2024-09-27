import json
from mistralai import Mistral
import os
from getpass import getpass
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub

from langchain_utils import get_chunks, get_document_splits, get_text_embedding
from llm_utils import run_mistral

with open("config.json", "r") as f:
    config = json.load(f)

os.environ["LANGCHAIN_TRACING_V2"] = config["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]

if __name__ == "__main__":
    api_key = config["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    web_paths = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6139814/"

    chunks = get_chunks(web_paths, 2048)
    embeddings = [get_text_embedding(client, chunk) for chunk in chunks]

    embeddings_model = MistralAIEmbeddings(
        model="mistral-embed", mistral_api_key=api_key
    )

    documents = get_document_splits(web_paths)
    # Load into a vector database
    vector = FAISS.from_documents(documents, embeddings_model)

    # Define a retriever interface
    retriever = vector.as_retriever()

    question = (
        "What can the standardizing of the assessment of patients with DoC assist with?"
    )
    prompt = hub.pull("prompt")

    run_mistral(client, question, prompt)
