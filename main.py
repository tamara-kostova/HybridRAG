import json
import os
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.messages import HumanMessage
from langchain_utils import get_document_splits
from llm_utils import format_docs

with open("config.json", "r") as f:
    config = json.load(f)

os.environ["LANGCHAIN_TRACING_V2"] = config["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
os.environ["MISTRAL_API_KEY"] = config["MISTRAL_API_KEY"]
os.environ["USER_AGENT"] = config["USER_AGENT"]
os.environ["HF_TOKEN"] = config["HF_TOKEN"]

if __name__ == "__main__":
    try:
        api_key = config["MISTRAL_API_KEY"]
        client = Mistral(api_key=api_key)
        urls = [
            "https://pub.towardsai.net/not-rag-but-rag-fusion-understanding-next-gen-info-retrieval-477788da02e2",
            "https://carloarg02.medium.com/my-favorite-coding-question-to-give-candidates-17ea4758880c",
            "https://medium.com/aiguys/why-gen-ai-boom-is-fading-and-whats-next-7f1363b92696",
            "https://towardsdatascience.com/what-nobody-tells-you-about-rags-b35f017e1570",
            "https://medium.com/codex/ai-does-not-need-intelligence-to-become-intelligent-e317c9e1a3bb",
        ]
        doc_splits = get_document_splits(urls)

        # Create embeddings
        embeddings = MistralAIEmbeddings()

        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embeddings,
        )

        # Create retriever
        retriever = vectorstore.as_retriever()
        mistral_model = "mistral-small-latest"
        llm = ChatMistralAI(model=mistral_model, temperature=0)
        question = (
            "What is the author's favourite coding question for candidate interviews?"
        )
        docs = retriever.get_relevant_documents(question)
        docs_txt = format_docs(docs)

        with open("prompt.txt", "r") as f:
            rag_prompt = f.read()

        rag_prompt_formatted = rag_prompt.format(document=docs_txt, question=question)
        generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        print(generation)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
