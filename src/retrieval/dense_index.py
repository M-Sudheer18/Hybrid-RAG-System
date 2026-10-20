import json
import faiss
import numpy as np
from config.settings import settings
from src.utils.logger import get_logger
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.utils.helpers import time_it
from src.embeddings.embedding_model import EmbeddingModelWrapper
logger = get_logger(__name__)

class DenseIndexManager:
    """
    Manages the FAISS dense vector index.
    Responsibilities:
    - Generate embeddings for document chunks
    - Build a FAISS inner-product index
    - Normalize vectors for cosine similarity
    - Persist the FAISS index and metadata
    - Load an existing index
    - Perform dense similarity search
    """
    def __init__(self):
        self.index_path    = (settings.FAISS_INDEX_DIR / "dense.faiss")
        self.metadata_path = (settings.FAISS_INDEX_DIR / "metadata.json")
        self.embedder      = EmbeddingModelWrapper.get_embeddings()
        self.index         = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}

    def _validate_consistency(self) -> bool:
        if self.index is None:
            logger.error("Consistency is Failed, Index is Not Loaded..")
            return False
        vector_count = self.index.ntotal
        metadata_count = len(self.metadata_store)
        if vector_count != metadata_count:
            logger.error(
                f"Index state mismatch! FAISS vectors: {vector_count}, "
                f"Metadata records: {metadata_count}"
            )
            return False
        logger.debug(f"Index consistency verified: {vector_count} entries matched.")
        return True

    @time_it
    def build_and_save_index(self, chunks: List[Document]) -> None:
        if not chunks:
            raise ValueError("Chunk List is Empty...")
        logger.info(f"Building Faiss Index From {len(chunks)} Chunks")

        # Extract the Text and Prepare Metadata Mapping
        texts = [
            chunk.page_content 
            for chunk in chunks
        ]
        # Store content + metadata using FAISS integer IDs
        self.metadata_store = {
            i: {
                "page_content": chunk.page_content, 
                "metadata": chunk.metadata
            }
            for i, chunk in enumerate(chunks)
        }

        # Generate embeddings
        embeddings_list = self.embedder.embed_documents(texts)

        # Convert to FAISS-compatible matrix
        embeddings_matrix = np.asarray(
            embeddings_list,
            dtype=np.float32
        )
        # Ensure 2D matrix
        if embeddings_matrix.ndim != 2:
            raise ValueError(
                "Embeddings Matrix Must be 2 - Dim"
            )

        # Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings_matrix)

        # Normalize embeddings so inner product becomes cosine similarity
        dimension = embeddings_matrix.shape[1]
        logger.info(
            f"Embedding matrix prepared | "
            f"Vectors: {len(embeddings_matrix)} | "
            f"Dimension: {dimension}"
        )

        # Inner Product + normalized vectors = cosine similarity
        base_index = faiss.IndexFlatIP(dimension)

        # Allows explicit integer IDs
        self.index = faiss.IndexIDMap(base_index)
        vector_ids = np.arange(
            len(chunks),
            dtype=np.int64
        )

        self.index.add_with_ids(embeddings_matrix, vector_ids)
        self._save_to_disk()
        logger.info(
            f"FAISS index saved successfully | "
            f"Vectors: {self.index.ntotal} | "
            f"Dimension: {dimension}"
        )

    def _save_to_disk(self) -> None:
        faiss.write_index(
            self.index, 
            str(self.index_path)
        )
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

    @time_it
    def load_index(self) -> bool:
        # Loads the pre-computed FAISS index and metadata store from disk into memory.
        if (
            not self.index_path.exists() or not self.metadata_path.exists()
        ):
            logger.warning(
                "FAISS index or metadata is missing. "
                "Build the index first."
            )
            return False
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                loaded_store = json.load(f)

            self.metadata_store = {int(key): value for key, value in loaded_store.items()}
            logger.info(f"Loaded Faiss Index with {self.index.ntotal} Vectors")
            return True
        except Exception as e:
            logger.error(
                f"Failed to load Faiss Index: {e}",
                exc_info=True
            )
            return False

    def search(
            self, 
            query: str, 
            top_k: int = settings.TOP_K_SEARCH
        ) -> List[Dict[str, Any]]:
        # Performs dense vector similarity search.
        if not query or not query.strip():
            raise ValueError(
                "Search Query Can't be Empty.."
            )
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("FAISS index is not Initialized or Loaded..")

        # Embed Query
        query_vector = self.embedder.embed_query(query)
        query_matrix = np.asarray(
            [query_vector],
            dtype=np.float32
        )

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_matrix)
        # Execute Search 
        distances, indices = self.index.search(query_matrix, top_k)

        results = []
        # Flatten the 2D O/P from Faiss
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_data = self.metadata_store.get(int(idx))
            if chunk_data:
                results.append({
                    "chunk_id": chunk_data["metadata"].get("chunk_id", f"idx_{idx}"),
                    "content" : chunk_data["page_content"],
                    "metadata": chunk_data["metadata"],
                    "dense_score": float(score)
                })
        logger.info(
            f"Dense search completed | "
            f"Query: '{query}' | "
            f"Results: {len(results)}"
        )
        return results

if __name__ == "__main__":
    from langchain_core.documents import Document
    # Test execution
    manager = DenseIndexManager()
    # Mock data
    mock_chunks = [
        Document(page_content="Econocom manages recycled IT assets.", metadata={"chunk_id": "test_1"}),
        Document(page_content="The sky is blue today.", metadata={"chunk_id": "test_2"}),
        Document(page_content="FastEmbed is a lightweight embedding library.", metadata={"chunk_id": "test_3"})
    ]
    manager.build_and_save_index(mock_chunks)    
    query = "What does Econocom do?"
    results = manager.search(query, top_k=2)
    print(f"Top matches for '{query}':")
    for res in results:
        print(f"- {res['content']} (Score: {res['dense_score']:.4f})")