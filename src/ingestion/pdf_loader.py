from pathlib import Path
from typing import List, Optional, Sequence
from pypdf import PdfReader
from langchain_core.documents import Document
from config.settings import settings
from src.utils.logger import get_logger
from src.utils.helpers import time_it
from src.ingestion.meta_manager import MetadataManager
logger = get_logger(__name__)
TEMP_UPLOAD_DIR = settings.DATA_DIR / "temp_uploads"

class PDFLoader:
    """
    It Handles the Reading and Extracting the Text from PDF
    Generates File Hashes to track Processed Files and Extracts
    """
    def __init__(self, directory_path: Optional[Sequence[Path | str]] = None):
        self.directory_path = [
            Path(path)
            for path in (
                directory_path
                if directory_path is not None
                else [settings.RAW_PDF_DIR]
            )
        ]

    @time_it
    def load_documents(self) -> List[Document]:
        """
        Iterates through all PDFs Data in Target Directory, and Extracts the 
        Text data Page by Page and Wraps them in Langchain Document Schemas with Rich Meta data
        """
        documents: List[Document] = []
        pdf_files = []

        for directory in self.directory_path:
            if not directory.exists():
                logger.warning(
                    f"PDF directory does not exist: {directory}"
                )
                continue

            pdf_files.extend(
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".pdf"
            )

        pdf_files = sorted(
            pdf_files,
            key=lambda path: str(path).lower()
        )


        if not pdf_files:
            logger.warning("No PDFs were found in the configured directories.")
            return documents

        logger.info(
            f"PDF extraction started for {len(pdf_files)} file(s)."
        )

        for pdf_path in pdf_files:
            try:
                # Generates hash to uniquely identify this file state
                file_hash = MetadataManager.compute_file_hash(pdf_path)
                reader = PdfReader(pdf_path)

                is_temporary = (
                    pdf_path.parent.resolve()
                    == TEMP_UPLOAD_DIR.resolve()
                )

                extracted_pages = 0
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    # Only ingest(read) the Pages which Contains Text Skip only Contains Images Pages
                    if not text or not text.strip():
                        continue

                    metadata = {
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "file_hash": file_hash,
                        "source_type": (
                            "temporary_upload"
                            if is_temporary
                            else "pdf"
                        )
                    }
                    documents.append(
                        Document(
                            page_content=text,
                            metadata=metadata
                        )
                    )
                    extracted_pages += 1
                logger.info(
                    f"Processed {pdf_path.name}: "
                    f"{extracted_pages} page(s)."
                )
            except Exception:
                logger.exception(
                    f"Failed to load PDF: {pdf_path.name}"
                )
        logger.info(
            f"Ingestion completed. Total pages extracted across all documents: "
            f"{len(documents)}"
        )
        return documents

if __name__ == "__main__":
    loader = PDFLoader(
        directory_paths=[
            settings.RAW_PDF_DIR,
            TEMP_UPLOAD_DIR,
        ]
    )

    docs = loader.load_documents()
    print(f"Loaded {len(docs)} page documents.")

    if docs:
        print(f"Sample text: {docs[0].page_content[:200]}")
