from typing import List, Dict, Optional
from src.generation.llm_client import GroqLLMClient
from src.utils.helpers import time_it
from src.utils.logger import get_logger
logger = get_logger(__name__)

class QueryProcessor:
    """
    Pre-processes user queries before they enter the retrieval pipeline.
    Handles contextual query reformulation (e.g., resolving pronouns
    from conversation history into a standalone search query).
    """
    def __init__(self):
        self.llm_client = GroqLLMClient.get_instance()

        # Highly constrained system prompt strictly for query rewriting
        self.condensation_prompt = (
            "You are an expert search query reformulator operating in a RAG system. "
            "Your task is to analyze the provided conversation history and the latest user query. "
            "If the latest query relies on context from the history (e.g., uses pronouns like "
            "'it', 'they', 'their', or refers to previously mentioned entities), rewrite it into a "
            "standalone, self-contained search query. "
            "If the query is already standalone, return it exactly as is. "
            "Do NOT answer the question. Output ONLY the rewritten query without quotes or explanations."
        )

    @time_it
    def process_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Condenses a conversational follow-up into a standalone search query.
        If no history is provided, it bypasses the LLM entirely to save latency.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query Can't be Empty...")
        
        # Fast path: No contextual history to resolve, skip LLM call
        if not chat_history:
            logger.debug("No chat history provided. Skipping query reformulation.")
            return query
        logger.info("Chat history detected. Reformulating query for contextual retrieval...")
        return self._rewrite_query(query, chat_history)

    def _rewrite_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """Executes a low-latency LLM call to rewrite the query based on context."""
        # Format the history explicitly so the LLM understands the conversational flow
        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
        ])
        user_message = (
            f"Conversation History: \n{history_text}\n\n"
            f"Latest Query: {query}\n\n"
            "Standalone Query: " 
        )
        messages = [
            {"role": "system", "content": self.condensation_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            # We use a temperature of 0.0 for deterministic outputs and 
            # cap tokens at 60 since a search query should be short.
            rewritten_query = self.llm_client.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=128,
                reasoning_effort="low"
            )

            # Clean up any potential quotation marks the LLM might have appended
            rewritten_query = rewritten_query.strip().strip('"\'').strip()
            if not rewritten_query:
                logger.error(
                    f"Query reformulation failed: {e}. "
                    f"Falling back to original query.",
                    exc_info=True
                )
                return query
            logger.info(f"Query reformulated: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            # Fallback mechanism ensures the system doesn't crash if the Groq API hiccups
            logger.error(f"Query reformulation failed: {e}. Falling back to original query.")
            return query
        
if __name__ == "__main__":
    # Isolated test execution
    processor = QueryProcessor()
    
    print("--- Test 1: Fast Path (No History) ---")
    q1 = processor.process_query("What does Econocom do?")
    print(f"Processed: {q1}\n")
    
    print("--- Test 2: Conversational Context ---")
    mock_history = [
        {"role": "user", "content": "What is Econocom's main focus?"},
        {"role": "assistant", "content": "Econocom focuses on digital transition and enterprise IT orchestration."}
    ]
    
    follow_up = "What about their cloud services?"
    q2 = processor.process_query(follow_up, chat_history=mock_history)
    print(f"Original: {follow_up}")
    print(f"Processed: {q2}")