from typing import Dict, Any
from config.settings import settings
from src.utils.logger import get_logger
from src.utils.helpers import time_it

from src.retrieval.retriever import Retriever
from src.orchestration.memory import SessionMemoryManager
from src.orchestration.query_processor import QueryProcessor
from src.generation.llm_client import GroqLLMClient
from src.generation.prompt_templates import RAGPromptBuilder
from src.generation.reranker import CrossEncoderReranker

logger = get_logger(__name__)

class RAGPipeline:
    """
    The unified end-to-end RAG application interface.
    Coordinates memory, query rewriting, dual-retrieval, reranking, and LLM generation.
    Designed to be directly injected into FastAPI route handlers.
    """
    def __init__(self):
        logger.info("Initializing complete end-to-end RAG Pipeline...")
        self.memory_manager = SessionMemoryManager.get_instance()
        self.query_processor = QueryProcessor()
        self.retriever = Retriever()
        self.reranker = CrossEncoderReranker.get_instance()
        self.llm_client = GroqLLMClient.get_instance()

    @time_it
    def execute(
        self,
        session_id: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Executes a single conversational turn through the RAG pipeline.
        Returns the finalized answer alongside the source metadata for UI rendering.
        """
        if not session_id or not session_id.strip():
            raise ValueError("Session ID Can't be Empty..")
        if not user_query or not user_query.strip():
            raise ValueError("User Query Can't be Empty..")
        logger.info(f"--- Starting Pipeline Execution for Session: {session_id} ---")

        # Fetch Session Memory
        session = self.memory_manager.get_or_create_session(session_id)
        chat_history = session.get_history()

        # Query Condensation ( Resolve Pronouns/ Context)
        standalone_query = self.query_processor.process_query(
            query=user_query,
            chat_history=chat_history
        )

        # Hybrid Retrieval (FAISS + BM25 -> RRF)
        # Fetching a larger pool (e.g., top 15) for high recall
        hybrid_candidates = self.retriever.retrieve(
            query=standalone_query,
            dense_top_k = settings.TOP_K_SEARCH,
            sparse_top_k = settings.TOP_K_SPARSE,
            final_top_n = 15
        )
        if not hybrid_candidates:
            logger.warning(
                f"No retrieval results found for session: {session_id}"
            )

        # Stage 2: Cross-Encoder Reranking
        # Refining the pool down to the absolute best chunks (e.g., top 5)
        top_chunks = self.reranker.rerank(
            query = standalone_query,
            chunks = hybrid_candidates,
            top_n = settings.TOP_N_RERANK 
        )

        # Prompt Construction
        messages = RAGPromptBuilder.build_messages(
            query=standalone_query,
            chunks=top_chunks
        )

        # LLM Generation
        answer = self.llm_client.generate(messages)

        # Update Session Memory
        # We store the *original* user query to keep the conversation feeling natural,
        session.add_user_message(user_query)
        session.add_assistant_message(answer)

        # Package the Response Payload
        sources = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "score": chunk.get("cross_encoder_score"),
                "metadata": chunk.get("metadata", {})
            }
            for chunk in top_chunks
        ]
        logger.info(f"--- Pipeline Execution Complete for Session: {session_id} ---")

        return {
            "session_id": session_id,
            "original_query": user_query,
            "standalone_query": standalone_query,
            "answer": answer,
            "sources": sources
        }

# For testing
if __name__ == "__main__":
    # Integration test for the full pipeline
    pipeline = RAGPipeline()
    test_session = "test_user_001"
    
    print("\n" + "="*50)
    print("TURN 1: Initial Query")
    print("="*50)
    response_1 = pipeline.execute(test_session, "What does Econocom do?")
    print(f"\nAssistant: {response_1['answer']}")
    print(f"\nSources Used: {[s['chunk_id'] for s in response_1['sources']]}")
    
    print("\n" + "="*50)
    print("TURN 2: Contextual Follow-up")
    print("="*50)
    response_2 = pipeline.execute(test_session, "How does it impact enterprise infrastructure?")
    print(f"\nRewritten Query: {response_2['standalone_query']}")
    print(f"Assistant: {response_2['answer']}")
    print(f"\nSources Used: {[s['chunk_id'] for s in response_2['sources']]}")