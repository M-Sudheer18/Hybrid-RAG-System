from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
from src.utils.logger import get_logger
from src.utils.helpers import time_it
logger = get_logger(__name__)

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(
            f"Document chunker initialized | "
            f"Chunk size: {self.chunk_size} | "
            f"Chunk overlap: {self.chunk_overlap}"
        )

    @time_it
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits cleaned Document objects into smaller chunks.
        Preserves existing metadata and adds:
        - chunk_index
        - chunk_id
        """
        if not documents:
            logger.warning("No Document Provided for Chunking")
            return []
        logger.info(f"Splitting {len(documents)} documents pages to chunks")
        raw_chunks = self.splitter.split_documents(documents)
        # Enrich chunk metadata with sequential identifiers per document
        enriched_chunks: List[Document] = []
        doc_chunk_counter = {}
        for chunk in raw_chunks:
            source = chunk.metadata.get("source", "unknown")
            doc_chunk_counter[source] = doc_chunk_counter.get(source, 0) + 1
            chunk.metadata["chunk_index"] = doc_chunk_counter[source]
            chunk.metadata["chunk_id"] = f"{source}_chunk_{doc_chunk_counter[source]}"
            enriched_chunks.append(chunk)
        logger.info(
            f"Chunking complete | "
            f"Produced {len(enriched_chunks)} chunks "
            f"from {len(documents)} documents"
        )
        return enriched_chunks

if __name__ == "__main__":
    test_doc = Document(
        page_content="This is sentence one. " * 30 + "This is sentence two. " * 30,
        metadata={
            "source": "test.pdf", 
            "page": 1
        }
    )
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.split_documents([test_doc])
    for c in chunks[:3]:
        print(c.metadata, len(c.page_content))