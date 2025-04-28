from src.graph.extractor import NodeExtractor
import os 

neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
PATH_TO_FOLDER = '/home/lukar/selected_pdfs'


extractor = NodeExtractor(url='bolt://localhost:7687', username="neo4j", chunk_size=2000,chunk_overlap=100,password="assistance-shock-mops")

extractor.extract_entities_from_folder(PATH_TO_FOLDER)