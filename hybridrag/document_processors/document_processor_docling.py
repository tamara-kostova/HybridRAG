import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List
import os
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    VlmPipelineOptions,
    smoldocling_vlm_mlx_conversion_options,
)
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from docling_core.types.doc.labels import DocItemLabel
from sentence_transformers import SentenceTransformer

from src.db.db_client import QdrantWrapper

from .logging_config import setup_logging

logger = logging.getLogger(__name__)


class DoclingDocumentProcessor:
    """
    A class to process documents for Docling with support for:
    - Batch PDF processing
    - Markdown export
    - Hybrid chunking
    - Vector database integration
    """

    def __init__(
        self,
        db_client=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 256,
        merge_peers: bool = False,
    ):
        """Initialize the document processor with optional vector DB client."""
        self.db_client = db_client
        self.pipeline_options = self._setup_pipeline_options()
        self.vlm_converter = self._setup_vlm_converter()
        self.converter = self._setup_converter()
        self.chunker = HybridChunker(
            tokenizer=model_name, max_tokens=max_tokens, merge_peers=merge_peers
        )
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized DoclingDocumentProcessor with model: {model_name}")

    def _setup_pipeline_options(self) -> VlmPipelineOptions:
        """Configure pipeline options for document processing."""
        options = VlmPipelineOptions()
        options.force_backend_text = False
        options.vlm_options = smoldocling_vlm_mlx_conversion_options
        return options

    def _setup_vlm_converter(self) -> DocumentConverter:
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

    def _setup_converter(self) -> DocumentConverter:
        """Set up the document converter with pipeline options."""

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        return doc_converter

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
                logger.info(f"Processing {pdf_file}")

                result = self.vlm_converter.convert(str(pdf_file))
                doc_pages = []

                for i, page in enumerate(result.pages):
                    doctags_page = DocTagsDocument.from_doctags_and_image_pairs(
                        [page.predictions.vlm_response.text], [page.image]
                    )
                    markdown_document_page = DoclingDocument.load_from_doctags(
                        doctags_page, document_name=f"Document_{i}"
                    )

                    logger.info("tables in page: " + str(markdown_document_page.tables))
                    page_md = f"Page Number: {i}\n"
                    page_md += markdown_document_page.export_to_markdown(image_placeholder = "", image_mode=ImageRefMode.REFERENCED) + "\n"

                    # Tables
                    for table_ix, table in enumerate(markdown_document_page.tables):
                        table_df: pd.DataFrame = table.export_to_dataframe()
                        logger.info(f"Table {table_ix}:")
                        logger.info(table_df)
                        logger.info("================")
                        # Generate summary with LLM
                        table_summary = self.call_llm_to_summarize_table(table_df)
                        page_md += f"\n**Table {table_ix} Summary:**\n{table_summary}\n"

                    images_filenames = []
                    for pic_ix, picture in enumerate(markdown_document_page.pictures):
                        image = picture.get_image(markdown_document_page)
                        # Skip images with very small resolution
                        if image.width < 30 or image.height < 30:
                            logger.info(f"Skipping image {pic_ix} on page {i} due to small size: {image.width}x{image.height}")
                            continue
                        output_dir = Path("scratch")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        image_filename = output_dir / f"{pdf_file.stem}-page{i}-picture{pic_ix}.png"
                        image.save(image_filename, format="PNG")
                        images_filenames.append(image_filename)
                        # Generate description with LLM
                        picture_description = self.call_llm_to_describe_picture(image_filename)
                        page_md += f"\n**Picture {pic_ix} Description:**\n{picture_description}\n, Referenced as: <image src=\"{image_filename}\" />.\n"

                    doc_pages.append(page_md)

                # # Process images and tables
                # result = self.converter.convert(str(pdf_file))

                # # save to txt file (string representation)
                # with open("./temp_new.txt", "w") as f:
                #     f.write(str(result.document))

                processed_docs[pdf_file.name] = "".join(doc_pages)

                inference_time = time.time() - start_time
                logger.info(
                    f"Processed {pdf_file.name} in {inference_time:.2f} seconds"
                )

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")

        return processed_docs

    def export_to_markdown(self, document: DoclingDocument) -> str:
        """
        Export a DoclingDocument to markdown format.

        Args:
            document: DoclingDocument to export

        Returns:
            Markdown string representation of the document
        """
        return document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)

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

        # save to md
        with open("./markdown_text.md", "w") as f:
            f.write(markdown_text)

        logger.info(f"Processing document: {doc_name}")

        # save a temp md file
        with open("./temp_md.md", "w") as f:
            f.write(markdown_text)

        conv_res: ConversionResult = self.converter.convert(str("./temp_md.md"))
        docling_document = conv_res.document

        # delete the temp md file
        os.remove("./temp_md.md")

        document_chunks = self.chunker.chunk(
            docling_document, chunk_size=chunk_size, chunk_overlap=50
        )

        for chunk in document_chunks:
            # detect if the chunk is a table
            print("================")
            for it in chunk.meta.doc_items:
                print(it.label)
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document: {doc_name}")
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def embed_chunks(self, doc_name: str, chunks: List) -> bool:
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

                logger.debug("==== Embedding chunk ====")
                logger.debug(
                    f"Chunk text: {text_to_embed[:100]}..."
                )  # First 100 chars for brevity

                if embedding is not None:
                    embeddings.append(embedding)

            return self.db_client.insert_paper(
                paper_name=doc_name, chunks=chunks_embeddings, embeddings=embeddings
            )

        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            return False

    def embed_text(self, text: str):
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    def call_llm_to_summarize_table(self, table_df):
        table_str = table_df.to_markdown()
        prompt = f"Summarize the following table:\n{table_str}"
        return "This is a table summary"
        # return self.llm_api_call(prompt)

    def call_llm_to_describe_picture(self, image_filename):
        # If your LLM supports image input, send the image; otherwise, use context
        prompt = f"Describe the content of the image at {image_filename}."
        return "This is a picture description"
        # return self.llm_api_call(prompt)


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    logger.info("Docling Document Processor Example")

    db_client = QdrantWrapper(collection_name="docling_documents", vector_size=384)

    # Initialize processor with your vector DB client
    processor = DoclingDocumentProcessor(
        db_client=db_client, max_tokens=256, merge_peers=True
    )

    # Process a directory of PDFs
    docs = processor.process_directory("./EEML")

    # For each document
    for doc_name, document in docs.items():
        # Create chunks
        chunks = processor.create_hybrid_chunks(doc_name, document, chunk_size=256)

        # Store in vector DB
        update_result = processor.embed_chunks(doc_name, chunks)
        logger.info(f"Update result for {doc_name}: {update_result}")
