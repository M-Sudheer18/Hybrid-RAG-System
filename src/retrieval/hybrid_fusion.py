from collections import defaultdict
from typing import List, Dict, Any
from config.settings import settings
from src.utils.helpers import time_it
from src.utils.logger import get_logger
logger = get_logger(__name__)

class HybridRetriever:
    """
    Fuses results from Dense (FAISS) and Sparse (BM25) retrievers
    using Reciprocal Rank Fusion (RRF). This standardizes rankings 
    across completely different scoring scales.
    """
    def __init__(self, k_constant: int = settings.RRF_K_CONSTANT):
        self.k = k_constant

    @time_it
    def apply_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_n: int = settings.TOP_N_RERANK,
    ) -> List[Dict[str, Any]]:
        """
        Merges and reranks result lists using the RRF algorithm.
        Retains all original payload data (content, metadata, raw scores).
        """
        # Accumulate RRF scores keyed by unique chunk_id
        rrf_scores = defaultdict(float)

        # Central Map to Store the Uniques chunk Payloads
        chunk_map = {}
        # Iterate through both lists (expandable to 3+ retrievers if needed)
        for result_list in [dense_results, sparse_results]:
            # RRF rank is 1-Indexed
            for rank, chunk in enumerate(result_list, start=1):
                chunk_id = chunk["chunk_id"]

                # RRF Formula
                rrf_scores[chunk_id] += 1.0 / (self.k + rank)

                # Store the payload upon first encounter
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = {
                        "chunk_id": chunk_id,
                        "content" : chunk["content"],
                        "metadata": chunk["metadata"],
                        "dense_score": None,
                        "sparse_score": None 
                    }
                # Carry over original raw scores if present for debugging
                if "dense_score" in chunk:
                    chunk_map[chunk_id]["dense_score"] = chunk["dense_score"]
                if "sparse_score" in chunk:
                    chunk_map[chunk_id]["sparse_score"] = chunk["sparse_score"]

                # Sort the accumulated chunks by their RRF score in descending order
        sorted_items = sorted(rrf_scores.items(), key = lambda x: x[1], reverse=True)

        # Construct the Final Ranked List
        fused_results = []
        for chunk_id, score in sorted_items[:top_n]:
            fused_chunk = chunk_map[chunk_id]
            fused_chunk["rrf_score"] = round(score, 6)
            fused_results.append(fused_chunk)

        logger.info(
            f"RRF fusion complete | Dense inputs: {len(dense_results)} | "
            f"Sparse inputs: {len(sparse_results)} | Output: {len(fused_results)}"
        )
        return fused_results
            
if __name__ == "__main__":
    # Test execution for RRF Fusion
    retriever = HybridRetriever(k_constant=60)
    
    # Mock data to simulate unaligned score scales
    mock_dense = [
        {"chunk_id": "doc_A", "content": "RAG is cool.", "metadata": {}, "dense_score": 0.89},
        {"chunk_id": "doc_B", "content": "FAISS is fast.", "metadata": {}, "dense_score": 0.75}
    ]
    
    mock_sparse = [
        {"chunk_id": "doc_C", "content": "BM25 uses lexical matching.", "metadata": {}, "sparse_score": 14.5},
        {"chunk_id": "doc_A", "content": "RAG is cool.", "metadata": {}, "sparse_score": 11.2}
    ]
    
    fused = retriever.apply_rrf(mock_dense, mock_sparse, top_n=3)
    
    for item in fused:
        print(f"[{item['chunk_id']}] RRF Score: {item['rrf_score']} | "
              f"Dense: {item['dense_score']} | Sparse: {item['sparse_score']}")