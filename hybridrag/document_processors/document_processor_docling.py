from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    smoldocling_vlm_mlx_conversion_options,
)

from docling_core.types.doc.labels import DocItemLabel
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from sentence_transformers import SentenceTransformer

from pathlib import Path
import time
from typing import List, Dict
import logging

from src.db.db_client import QdrantWrapper

class DoclingDocumentProcessor:
    """
    A class to process documents for Docling with support for:
    - Batch PDF processing
    - Markdown export
    - Hybrid chunking
    - Vector database integration
    """

    def __init__(self, db_client=None, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 256,
                 merge_peers: bool = False):
        """Initialize the document processor with optional vector DB client."""
        self.db_client = db_client
        self.pipeline_options = self._setup_pipeline_options()
        self.converter = self._setup_converter()
        self.chunker = HybridChunker(tokenizer=model_name, 
                                     max_tokens=max_tokens, 
                                     merge_peers=merge_peers)
        self.model = SentenceTransformer(model_name)
        
    def _setup_pipeline_options(self) -> VlmPipelineOptions:
        """Configure pipeline options for document processing."""
        options = VlmPipelineOptions()
        options.force_backend_text = False
        options.vlm_options = smoldocling_vlm_mlx_conversion_options
        return options
    
    def _setup_converter(self) -> DocumentConverter:
        """Set up the document converter with pipeline options."""
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self.pipeline_options,
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self.pipeline_options,
                ),
            }
        )

    def process_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Process all PDF files in the given directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            Dictionary mapping filenames to processed DoclingDocuments
        """
        pdf_files = Path(directory_path).glob("*.pdf")
        processed_docs = {}
        
        for pdf_file in pdf_files:
            try:
                start_time = time.time()
                logging.info(f"Processing {pdf_file}")
                
                result = self.converter.convert(str(pdf_file))
                doc_pages = []
                
                i = 0
                for page in result.pages:
                    doctags_page = DocTagsDocument.from_doctags_and_image_pairs([page.predictions.vlm_response.text], [page.image])
                    print(f"Page number: {i}")
                    markdown_document_page = DoclingDocument.load_from_doctags(doctags_page,
                                                                        document_name=f"Document_{i}")

                    doc_pages.append(f"Page Number: {i}\n" + markdown_document_page.export_to_markdown() + "\n")
                    
                    i += 1

                # for i, page in enumerate(result.pages):
                #     doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                #         [page.predictions.vlm_response.text], 
                #         [page.image]
                #     )
                #     doc_pages.append(doctags_doc)
                
                # # Combine all pages into one document
                # document = DoclingDocument.load_from_doctags(
                #     doc_pages,
                #     document_name=pdf_file.stem
                # )
                
                processed_docs[pdf_file.name] = "".join(doc_pages)
                
                inference_time = time.time() - start_time
                logging.info(f"Processed {pdf_file.name} in {inference_time:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {str(e)}")
                
        return processed_docs

    def export_to_markdown(self, document: DoclingDocument) -> str:
        """
        Export a DoclingDocument to markdown format.
        
        Args:
            document: DoclingDocument to export
            
        Returns:
            Markdown string representation of the document
        """
        return document.export_to_markdown()

    def create_hybrid_chunks(self, doc_name, document, chunk_size=256) -> List[Dict]:
        """
        Create hybrid chunks from markdown text using semantic and structure-aware chunking.
        
        Args:
            document: List of documents to chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        markdown_text = document
        print(f"Processing document: {doc_name}")

        docling_document = DoclingDocument(name=doc_name)
        docling_document.add_text(label=DocItemLabel.TEXT, text=markdown_text)

        document_chunks = self.chunker.chunk(
            docling_document,
            chunk_size=chunk_size,
            chunk_overlap=50
        )

        for chunk in document_chunks:
            chunks.append(chunk)

        print(f"Created {len(chunks)} chunks for document: {doc_name}")
        print(f"Total chunks created: {len(chunks)}")
        return chunks

    def embed_chunks(self, doc_name:str , chunks: List) -> bool:
        """
        Embed chunks into the vector database.
        
        Args:
            chunks: List of document chunks to embed
            
        Returns:
            Boolean indicating success
        """
        try:
            embeddings = []
            chunks_embeddings = []
            for chunk in chunks:
                text_to_embed = self.chunker.contextualize(chunk=chunk)
                embedding = self.embed_text(text_to_embed)
                chunks_embeddings.append(text_to_embed)
                
                if embedding is not None:
                    embeddings.append(embedding)

            return self.db_client.insert_paper(
                paper_name=doc_name,
                chunks=chunks_embeddings,
                embeddings=embeddings
            )
            
        except Exception as e:
            logging.error(f"Error embedding chunks: {str(e)}")
            return False
    
    def embed_text(self, text: str):
        try:
            return self.model.encode(text)
        except Exception as e:
            logging.error(f"Error embedding text: {e}")
            return None
        
if __name__ == "__main__":
    print("Docling Document Processor Example")

    db_client = QdrantWrapper(collection_name="docling_documents",
                              vector_size=384)

    # Initialize processor with your vector DB client
    processor = DoclingDocumentProcessor(db_client=db_client, max_tokens=256, merge_peers=True)

    # Process a directory of PDFs
    docs = processor.process_directory("./EEML")

    # For each document
    for doc_name, document in docs.items():
        # Create chunks
        chunks = processor.create_hybrid_chunks(doc_name, document, chunk_size=256)
        
        # Store in vector DB
        update_result = processor.embed_chunks(doc_name, chunks)
        print(f"Update result for {doc_name}: {update_result}")