from pathlib import Path
from typing import Dict
from config.settings import settings
from src.utils.logger import get_logger
from src.utils.helpers import time_it
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.chunker import DocumentChunker
from src.ingestion.meta_manager import MetadataManager
from src.retrieval.dense_index import DenseIndexManager
from src.retrieval.sparse_index import SparseIndexManager
logger = get_logger(__name__)

class IngestionPipeline:
    """
    Coordinates the complete PDF ingestion and indexing pipeline.
    Pipeline:
        PDF Files
            ↓
        Metadata Registration
            ↓
        PDF Loading
            ↓
        Text Cleaning
            ↓
        Chunking
            ↓
        Metadata Enrichment
            ↓
        FAISS Dense Index
            +
        BM25 Sparse Index
    Both FAISS and BM25 are built from the exact same ordered
    collection of chunks to maintain consistency between the
    dense and sparse retrieval systems.
    """
    def __init__(
        self,
        raw_data_dir: Path = settings.DATA_DIR / "raw_pdfs"
    ):
        self.raw_data_dir = raw_data_dir
        logger.info("Initializing PDF Ingestion Pipeline...")

        self.metadata_manager = MetadataManager()
        self.temp_upload_dir = settings.DATA_DIR / "temp_uploads"
        self.temp_upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        self.pdf_loader = PDFLoader(
            directory_path=[
                self.raw_data_dir,
                self.temp_upload_dir
            ]
        )
        self.cleaner = DocumentCleaner()
        self.chunker = DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.dense_manager = DenseIndexManager()
        self.sparse_manager = SparseIndexManager()
        logger.info("PDF Ingestion Pipeline initialized successfully.")

    @time_it
    def run_pipeline(self) -> Dict[str, int]:
        """
        Executes the complete PDF ingestion pipeline.
        Returns:
            Dictionary containing ingestion statistics.
        """
        
        logger.info("STARTING PDF INGESTION PIPELINE")

        # Discover PDFs from both permanent and temporary directories
        pdf_files = []

        for directory in [self.raw_data_dir, self.temp_upload_dir]:
            if not directory.exists():
                logger.warning(f"PDF directory does not exist: {directory}")
                continue

            pdf_files.extend(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            )

        pdf_files = sorted(
            pdf_files,
            key=lambda path: str(path).lower(),
        )

        if not pdf_files:
            logger.warning(
                "No PDFs found in permanent or temporary directories."
            )
            return {
                "pdf_files": 0,
                "new_documents": 0,
                "pages": 0,
                "chunks": 0,
            }

        logger.info(
            f"Discovered {len(pdf_files)} PDF(s). "
            "Beginning corpus processing..."
        )

        document_id_map: Dict[str, str] = {}
        new_documents = 0

        # Sequential extraction, cleaning, and chunking across all PDFs
        for pdf_path in pdf_files:
            try:
                file_hash = self.metadata_manager.compute_file_hash(pdf_path)
                # Check whether this document already exists
                existing_doc_id = next(
                        (
                            doc_id
                            for doc_id, metadata
                            in self.metadata_manager.registry.items()
                            if metadata.get("file_hash") == file_hash
                        ),
                        None
                    )
                if existing_doc_id:
                    document_id_map[file_hash] = existing_doc_id
                    logger.info(
                            f"Document already registered: "
                            f"'{pdf_path.name}' "
                            f"[{existing_doc_id}]"
                        )
                else:
                    is_temporary = (
                        pdf_path.parent.resolve()
                        == self.temp_upload_dir.resolve()
                    )
                    source_type = (
                        "temporary_upload" if is_temporary else "pdf"
                    )
                    doc_id = self.metadata_manager.register_document(
                        file_path=pdf_path,
                        source_type=source_type
                    )
                    if doc_id:
                        document_id_map[file_hash] = doc_id
                        new_documents += 1
            except Exception as e:
                logger.error(
                    f"Failed to register document "
                    f"'{pdf_path.name}': {e}",
                    exc_info=True
                )
        logger.info(
            f"Document registration completed | "
            f"New documents: {new_documents}"
        )


        # Load ALL PDF Pages
        logger.info("Loading PDF documents...")
        raw_documents = self.pdf_loader.load_documents()
        if not raw_documents:
            logger.error(
                "No PDF pages were extracted. "
                "Pipeline execution stopped."
            )
            return {
                "pdf_files": len(pdf_files),
                "new_documents": new_documents,
                "pages": 0,
                "chunks": 0
            }
        logger.info(
            f"PDF loading completed | "
            f"Pages extracted: {len(raw_documents)}"
        )

        # Clean ALL Documents
        logger.info("Cleaning extracted documents...")
        cleaned_documents = self.cleaner.clean_documents(
            raw_documents
        )
        if not cleaned_documents:
            logger.error(
                "No documents remained after cleaning. "
                "Pipeline execution stopped."
            )
            return {
                "pdf_files": len(pdf_files),
                "new_documents": new_documents,
                "pages": 0,
                "chunks": 0
            }
        logger.info(
            f"Document cleaning completed | "
            f"Valid pages: {len(cleaned_documents)}"
        )

        # Chunk ALL Documents
        logger.info("Creating document chunks...")
        all_corpus_chunks = self.chunker.split_documents(
            cleaned_documents
        )
        if not all_corpus_chunks:
            logger.error(
                "No chunks were generated. "
                "Pipeline execution stopped."
            )
            return {
                "pdf_files": len(pdf_files),
                "new_documents": new_documents,
                "pages": len(cleaned_documents),
                "chunks": 0
            }
        logger.info(
            f"Chunking completed | "
            f"Total chunks: {len(all_corpus_chunks)}"
        )

        # Enrich Chunk Metadata
        logger.info("Enriching chunk metadata...")
        enriched_count = 0
        missing_doc_count = 0
        for chunk in all_corpus_chunks:
            file_hash = chunk.metadata.get("file_hash")
            doc_id = document_id_map.get(file_hash)
            if doc_id:
                chunk.metadata["doc_id"] = doc_id
                enriched_count += 1
            else:
                missing_doc_count += 1
                logger.warning(
                    f"No document ID found for "
                    f"file hash: {file_hash}"
                )
            # Keep File Name Explicitly Available
            chunk.metadata["filename"] = chunk.metadata.get(
                "source",
                "unnknown"
            )
        logger.info(
            f"Chunk metadata enrichment completed | "
            f"Enriched: {enriched_count} | "
            f"Missing document IDs: {missing_doc_count}"
        )

        # Build FAISS Dense Index
        logger.info("BUILDING FAISS DENSE INDEX")
        self.dense_manager.build_and_save_index(
            all_corpus_chunks
        )
        logger.info(
            "FAISS dense index successfully built and saved."
        )

        # Build BM25 Sparse Index
        logger.info("BUILDING BM25 SPARSE INDEX")
        self.sparse_manager.build_and_save_index(
            all_corpus_chunks
        )
        logger.info(
            "BM25 sparse index successfully built and saved."
        )

        # Count Chunks Per Document
        document_chunk_counts: Dict[str, int] = {}
        for chunk in all_corpus_chunks:
            doc_id = chunk.metadata.get("doc_id")
            if doc_id:
                document_chunk_counts[doc_id] = (
                    document_chunk_counts.get(doc_id, 0) + 1
                )

        # Update Document Registry
        logger.info(
            "Updating document registry..."
        )
        for doc_id, chunk_count in document_chunk_counts.items():
            self.metadata_manager.update_document_status(
                doc_id=doc_id,
                chunk_count=chunk_count,
                status="completed"
            )
        logger.info(
            "Document registry updated successfully."
        )

        # Final Statistics
        statistics = {
            "pdf_files": len(pdf_files),
            "new_documents": new_documents,
            "pages": len(cleaned_documents),
            "chunks": len(all_corpus_chunks)
        }
        logger.info("PDF INGESTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(
            f"PDF Files     : {statistics['pdf_files']}"
        )
        logger.info(
            f"New Documents : {statistics['new_documents']}"
        )
        logger.info(
            f"Pages         : {statistics['pages']}"
        )
        logger.info(
            f"Chunks        : {statistics['chunks']}"
        )

        return statistics

if __name__ == "__main__":

    pipeline = IngestionPipeline()

    results = pipeline.run_pipeline()

    print("\n")
    print("=" * 50)
    print("INGESTION RESULTS")
    print("=" * 50)

    print(
        f"PDF Files     : {results['pdf_files']}"
    )

    print(
        f"New Documents : {results['new_documents']}"
    )

    print(
        f"Pages         : {results['pages']}"
    )

    print(
        f"Chunks        : {results['chunks']}"
    ) 