import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config.settings import settings
from src.utils.helpers import time_it
from src.utils.logger import get_logger
logger = get_logger(__name__)

class SparseIndexManager:
    """
    Manages the BM25 sparse keyword index.
    Responsibilities:
    - Tokenize document chunks and queries
    - Build and persist a BM25Okapi index
    - Manage integer-mapped metadata storage
    - Perform lexical keyword search and return raw BM25 scores for RRF fusion
    """
    def __init__(self):
        self.index_path: Path = settings.BM25_INDEX_DIR / "bm25.pkl"
        self.metadata_path: Path = settings.BM25_INDEX_DIR / "metadata.json"
        self.bm25: Optional[BM25Okapi] = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Tokenizes input text into normalized alphanumeric terms.
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def _validate_consistency(self) -> bool:
        # Verifies that the BM25 corpus size matches the metadata store count.
        if self.bm25 is None:
            logger.error("Consistency Check Failed: BM25 is Not Loaded..")
            return False

        corpus_size = self.bm25.corpus_size
        metadata_count = len(self.metadata_store)
        if corpus_size != metadata_count:
            logger.error(
                f"BM25 state Mismatch! Indexed Docs: {corpus_size}, "
                f"Metadata record: {metadata_count}"
            )
            return False
        logger.debug(f"BM25 consistency Verified: {corpus_size} entries synchronized..")
        return True

    @time_it
    def build_and_save_index(self, chunks: List[Document]) -> None:
        """
        Tokenizes document chunks, constructs the BM25Okapi model,
        and saves both the model and metadata map to disk.
        """
        if not chunks:
            raise ValueError("Chunk list can't be Empty when building the BM25 Index..")

        logger.info(f"Building BM25 Sparse index from {len(chunks)} Chunks..")
        self.metadata_store = {
            i : {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for i, chunk in enumerate(chunks)
        }

        # Tokenize each chunk for BM25 frequency calculations
        tokenized_corpus = [self.tokenize(chunk.page_content) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        if not self._validate_consistency():
            raise RuntimeError("Validation Consistency Failed metadata and BM25 Corpus misaligned...")

        self._save_to_disk()
        logger.info(
            f"BM25 index build and Saved | Documents: {self.bm25.corpus_size} |"
            f"Path: {settings.BM25_INDEX_DIR}"
        )

    def _save_to_disk(self) -> None:
        # Serializes BM25 Model via Pickle and metadata via json
        if self.bm25 is None:
            raise RuntimeError("Cannot persist uninitialized BM25 index.")

        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

    @time_it
    def load_index(self) -> bool:
        # Loads the Serialized BM25 Model and Metadata Store in Memory
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("BM25 index or Metadata Store Was Missing on the Disk. Build first..")
            return False
        try:
            with open(self.index_path, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                loaded_store = json.load(f)

            self.metadata_store = {int(k): v for k, v in loaded_store.items()}
            if not self._validate_consistency():
                logger.warning("Loaded BM25 Index Failed Consistency Check..")
                return False

            logger.info(f"Loaded Valid BM25 index with {self.bm25.corpus_size} Documents..")
            return True
        except Exception as e:
            logger.error(f"Failed to Load BM25 Index: {e}", exc_info=True)
            return False

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K_SPARSE,
    ) -> List[Dict[str, Any]]:
        # Executes BM25 lexical search and returns ranked chunks with raw scores.
        if not query or not query.strip():
            raise ValueError("Query Can't be Empty...")
        if top_k <= 0:
            raise ValueError("Top_k Must be Greater than Zero(0)..")
        if self.bm25 is None:
            if not self.load_index():
                raise RuntimeError("BM25 Index is not Initialized or Loaded..")

        tokenized_query  = self.tokenize(query)
        if not tokenized_query:
            logger.warning(f"Query {query} Produced No Valid Alpha numeric tokens..")
            return []

        # Get raw BM25 Scores for all Raw Documents...
        scores = self.bm25.get_scores(tokenized_query)

        # Retrieve indices sorted by score in descending Order..
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            score = scores[idx]
            # Filter out chunks with Zero Lexical Overlap..
            if score <= 0.0:
                continue

            chunk_data = self.metadata_store.get(int(idx))
            if chunk_data:
                results.append({
                    "chunk_id": chunk_data["metadata"].get("chunk_id", f"idx_{idx}"),
                    "content": chunk_data["page_content"],
                    "metadata": chunk_data["metadata"],
                    "sparse_score": float(score)
                })
        logger.info(f"BM25 Search Completed | Query: {query} | Results: {len(results)}")
        return results

if __name__ == "__main__":
    manager = SparseIndexManager()
    sample_chunks = [
        Document(
            page_content="Econocom provides certified enterprise infrastructure services.",
            metadata={"chunk_id": "sparse_1"}
        ),
        Document(
            page_content="Weather in Hyderabad is pleasant during monsoon.",
            metadata={"chunk_id": "sparse_2"}
        ),
        Document(
            page_content="Enterprise infrastructure and cloud modernization with Econocom.",
            metadata={"chunk_id": "sparse_3"}
        )
    ]
    manager.build_and_save_index(sample_chunks)
    query_text = "Economic Infrastructure"
    ranked_chunks = manager.search(query_text, top_k=2)
    for item in ranked_chunks:
        print(f"[{item['chunk_id']}] Score: {item['sparse_score']:.4f} | Content: {item['content']}")