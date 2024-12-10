from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
import re
from tqdm import tqdm
import os


class NodeExtractor:
    def __init__(self, url:str = 'bolt://localhost:7687',username:str = 'neo4j',password = '',chunk_size:int = 1200,chunk_overlap:int = 200):
        try:
            self.graph = Neo4jGraph(url=url, username=username, password=password)
        except Exception as e:
            print(f'Unable to connect to Neo4j database: {e}')

        with open(file='hybridrag/graph/extraction_system_prompt.txt', mode='r') as file:
            prompt_content = file.read()

            self.system_prompt = prompt_content

        with open(file='hybridrag/graph/extraction_human_prompt.txt', mode ='r') as file:
            prompt_content = file.read()

            self.human_prompt = prompt_content

        self.extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_prompt,
                ),
                (
                    "human",
                    self.human_prompt
                ),
            ]
        )

        try:
            self.llm = ChatOllama(model='llama3:8b')
        except Exception as e:
            print(f'Unable to get LLM instance: {e}')

        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(
        self,
        text:str
    ) -> List[str]:
        try:
            return self.chunker.split_text(text)
        except Exception as e:
            print(f'Error chunking {text} : {e}')


    def parse_extraction_output(self, response: str):
        nodes = {}
        relationships = []

        nodes_block = re.search(r"\[\[START_NODES\]\](.*?)\[\[STOP_NODES\]\]", response, re.DOTALL)
        relationships_block = re.search(r"\[\[START_RELATIONSHIPS\]\](.*?)\[\[STOP_RELATIONSHIPS\]\]", response, re.DOTALL)

        if nodes_block:
            nodes_lines = nodes_block.group(1).strip().split("\n")
            for line in nodes_lines:
                match = re.match(r'<"(.*?)", "(.*?)", "(.*?)">', line.strip())
                if match:
                    node_id, label, text = match.groups()
                    nodes[node_id] = Node(id=node_id, type=label, properties={"text": text})

        if relationships_block:
            relationships_lines = relationships_block.group(1).strip().split("\n")
            for line in relationships_lines:
                match = re.match(r'<"(.*?)", "(.*?)", "(.*?)">', line.strip())
                if match:
                    node_id1, rel_type, node_id2 = match.groups()
                    if node_id1 in nodes and node_id2 in nodes:
                        relationships.append(Relationship(
                            source=nodes[node_id1],
                            type=rel_type,
                            target=nodes[node_id2]
                        ))

        return list(nodes.values()), relationships


    def extract_entities_from_pdf(
        self,
        pdf_path: str
    ) -> List[GraphDocument]:

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        graph_documents = []

        for document in documents:
            full_text = document.page_content
            chunks = self.chunk_text(full_text)

            nodes, relationships = [], []

            for chunk in tqdm(chunks):
                chain = self.extraction_prompt | self.llm
                response = chain.invoke(input={"input_chunk": chunk}).content
                try:
                    extracted_nodes, extracted_relationships = self.parse_extraction_output(response)
                    nodes.extend(extracted_nodes)
                    relationships.extend(extracted_relationships)
                except Exception as e:
                    print(f'Unable to parse input for chunk: {e}')

            graph_document = GraphDocument(
                nodes=nodes,
                relationships=relationships,
                source=document
            )

            graph_documents.append(graph_document)

        return graph_documents

    def extract_entities_from_folder(self, folder_path) -> None:

        for pdf_path in tqdm(os.listdir(folder_path)):

            pdf = os.path.join(folder_path,pdf_path)
            graph_documents = self.extract_entities_from_pdf(pdf)
            self.graph.add_graph_documents(graph_documents)
