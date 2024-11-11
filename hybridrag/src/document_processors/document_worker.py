from hybridrag.src.document_processors.document_processor import DocumentProcessor
from src.db.db_client import QdrantWrapper
from dotenv import load_dotenv
import os 


load_dotenv()

PATH_TO_PDFS = "/home/lukar/projects/hybridrag/hybridrag/src/scraper/pdfs/"


processor = DocumentProcessor(
    db_client= QdrantWrapper(url="https://7ecf0b14-c826-4ae4-b61b-3bd710fc75d9.europe-west3-0.gcp.cloud.qdrant.io", api_key="agxIHD5sPk-2svMtUPmn26Gf3CHZLhmidbz-eOQuOjjushtYCl9aVQ"),
    directory_path=PATH_TO_PDFS
)

n_processed_pdfs = processor.procces_directory()

print(f'Inserted {n_processed_pdfs} into database!')

