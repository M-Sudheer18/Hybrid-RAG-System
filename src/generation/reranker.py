from typing import List, Dict, Any
from fastembed.rerank.cross_encoder import TextCrossEncoder
from config.settings import settings
from src.utils.helpers import time_it
from src.utils.logger import get_logger
logger = get_logger(__name__)

class CrossEncoderReranker:
    """
    Reranking Layer
    Uses an ONNX-optimized Cross-Encoder model to calculate deep, token-level
    relevance scores between the user query and the RRF-fused candidate chunks.
    """
    _instance = None
    def __new__(cls):
        """
        Ensure the cross-encoder model is only loaded into memory once.
        --> Becoz Heavier than a standard embedding model
        """
        if cls._instance is None:
            logger.info(f"Loading FastEmbed Cross-Encoder: [{settings.RERANKER_MODEL_NAME}]...")
            try:
                cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
                # Initialize the ONNX-backed Cross-Encoder
                cls._instance.model = TextCrossEncoder(model_name=settings.RERANKER_MODEL_NAME)
                logger.info("Cross-Encoder model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Cross-Encoder: {e}", exc_info=True)
                raise
        return cls._instance

    @classmethod
    def get_instance(cls) -> "CrossEncoderReranker":
        # Retrieves the instance.
        return cls()

    @staticmethod
    def _deduplicate_chunks(
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove chunks with identical text, ignoring case and whitespace."""
        unique_chunks = []
        seen_content = set()

        for chunk in chunks:
            content = str(chunk.get("content", ""))
            normalized_content = " ".join(content.casefold().split())

            # Keep empty-content chunks distinct rather than merging them.
            if not normalized_content:
                unique_chunks.append(chunk)
                continue

            if normalized_content in seen_content:
                continue

            seen_content.add(normalized_content)
            unique_chunks.append(chunk)

        return unique_chunks

    @time_it
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = settings.TOP_N_RERANK
    ) -> List[Dict[str, Any]]:
        """
        Takes the query and candidate chunks, scores them simultaneously using 
        cross-attention, and returns the strictly ordered list.
        """
        if not query or not query.strip():
            raise ValueError("Query can't be empty.")
        if not chunks:
            logger.warning("No Chunks Provided to reranking..")
            return []
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0")
        logger.info(f"Cross-Encoder evaluating {len(chunks)} candidate chunks for query: '{query}'")

        # Extract the text Content for the Model
        documents = [chunk["content"] for chunk in chunks]
        try:
            scores_generator = self.model.rerank(query, documents)
            scores = list(scores_generator) 
        except Exception as e:
            logger.error(
                f"Reranking failed: {e}. "
                f"Falling back to original input order.",
                exc_info=True
            )
            unique_chunks = self._deduplicate_chunks(chunks)
            return unique_chunks[:top_n]

        # Attach the new cross-encoder score to the payload
        for chunk, score in zip(chunks, scores):
            # It Stores the Scores in float Fromat Only...
            chunk["cross_encoder_score"] = float(score) 

        # Sort the chunks descending based on the precise cross-encoder score
        reranked_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.get(
                "cross_encoder_score",
                float("-inf"),
            ),
            reverse=True,
        )
        # Preserve the best-ranked occurrence of each unique passage.
        unique_chunks = self._deduplicate_chunks(reranked_chunks)
        
        final_chunks = unique_chunks[:top_n]
        duplicates_removed = len(reranked_chunks) - len(unique_chunks)
        logger.info(
            f"Reranking complete | "
            f"Candidates: {len(chunks)} | "
            f"Duplicates removed: {duplicates_removed} | "
            f"Unique chunks sent to LLM: {len(final_chunks)}"
        )

        return final_chunks

# For Test Purpose Only
if __name__ == "__main__":
    # Isolated test execution
    reranker = CrossEncoderReranker.get_instance()
    
    test_query = "What is Econocom's main focus?"
    mock_candidates = [
        {"chunk_id": "c1", "content": "The weather in Hyderabad is pleasant.", "rrf_score": 0.03},
        {"chunk_id": "c2", "content": "Econocom focuses on digital transition and enterprise IT.", "rrf_score": 0.02},
        {"chunk_id": "c3", "content": "Many companies are moving to the cloud.", "rrf_score": 0.025}
    ]
    
    refined_results = reranker.rerank(test_query, mock_candidates, top_n=2)
    
    print("\n--- Cross-Encoder Refined Order ---")
    for r in refined_results:
        print(f"[{r['chunk_id']}] CE Score: {r['cross_encoder_score']:.4f} | Content: {r['content']}")