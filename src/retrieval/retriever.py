from typing import List, Dict, Any
from config.settings import settings
from src.utils.helpers import time_it
from src.utils.logger import get_logger
from src.retrieval.dense_index import DenseIndexManager
from src.retrieval.sparse_index import SparseIndexManager
from src.retrieval.hybrid_fusion import HybridRetriever
logger = get_logger(__name__)

class Retriever:
    """
    Orchestration layer for Stage 1 candidate selection.
    Coordinates parallel Dense (FAISS) and Sparse (BM25) searches, 
    fusing the results via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        self.dense_retriever  = DenseIndexManager()
        self.sparse_retriever = SparseIndexManager()
        self.hybrid_retriever = HybridRetriever()

    @time_it
    def retrieve(
        self,
        query: str,
        dense_top_k: int  = settings.TOP_K_SEARCH,
        sparse_top_k: int = settings.TOP_K_SPARSE,
        final_top_n: int  = settings.TOP_N_RERANK
    ) -> List[Dict[str, Any]]:
        # Executes the dual-retrieval and fusion pipeline.
        if not query or not query.strip():
            raise ValueError("Query can't be empty.")
        logger.info(f"Hybrid retrieval started for query: '{query}'")

        # 2. Dense Retrieval (FAISS)
        try:
            dense_results = self.dense_retriever.search(
                query=query,
                top_k=dense_top_k
            )
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            dense_results = []

        try:
            sparse_results = self.sparse_retriever.search(
                query=query,
                top_k=sparse_top_k
            )
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            sparse_results = []

        # RRF Fusion
        if not dense_results and not sparse_results:
            logger.warning("Both retrievers returned empty lists. No results to fuse.")
            return []

        hybrid_results = self.hybrid_retriever.apply_rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_n = final_top_n
        )
        logger.info(
            f"Retrieval complete | Dense: {len(dense_results)} | "
            f"Sparse: {len(sparse_results)} | RRF Final: {len(hybrid_results)}"
        )
        return hybrid_results

if __name__ == "__main__":
    retriever = Retriever()
    query_text = "What does Econocom do?"

    try:
        results = retriever.retrieve(query_text)
        
        print(f"\n--- Final RRF Results for: '{query_text}' ---")
        for result in results:
            print(
                f"[{result['chunk_id']}] "
                f"RRF: {result.get('rrf_score', 0):.6f} | "
                f"Content: {result['content'][:80]}..."
            )
    except Exception as e:
        print(f"Test failed. Ensure mock indexes are built first. Error: {e}")