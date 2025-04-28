from src.db.models.entities import Entities
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.graphs import Neo4jGraph
import os

from sentence_transformers import SentenceTransformer


class GraphRetriever: 
    def __init__(self, url:str = 'bolt://localhost:7687',username:str = 'neo4j',password = ''):
        try: 
            self.graph = Neo4jGraph(url=url, username=username, password=password)
        except Exception as e:
            print(f'Unable to connect to Neo4j database: {e}')

        current_dir = os.path.dirname(os.path.abspath(__file__))  
        prompts_dir = os.path.join(current_dir, "prompts")

        system_prompt_path = os.path.join(prompts_dir, "entities_extraction_system_prompt.txt")
        human_prompt_path = os.path.join(prompts_dir, "entities_extraction_human_prompt.txt")

        with open(system_prompt_path, mode='r') as file:
            self.entity_extraction_system_prompt = file.read()

        with open(human_prompt_path, mode='r') as file:
            self.entity_extraction_human_prompt = file.read()


        self.entity_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.entity_extraction_system_prompt,
                ),
                (
                    "human",
                    self.entity_extraction_human_prompt
                ),
            ]
        )

        try: 
            self.llm = ChatOllama(model='llama3.1:8b')
                
        except Exception as e:
                print(f'Unable to get LLM instance: {e}')

        self.entity_extract_chain = self.entity_extraction_prompt | self.llm.with_structured_output(Entities)


    def retrieve(self, question: str) -> str:
        """Retrieves entities in the neighborhood of the entity mentioned in the question"""
        result = ""
        entities = self.entity_extract_chain.invoke(input={"question": question})
        print(entities)
        
        for entity in entities.ids:
            response = self.graph.query(
                """
                MATCH (node)
                WHERE node.id = $entity_id
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                } 

                RETURN output LIMIT 50
                """,
                {"entity_id": entity},
            )
            result += "\n".join([el['output'] for el in response])
        
        return result