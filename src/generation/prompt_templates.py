from typing import List, Any, Dict
from src.utils.logger import get_logger
logger = get_logger(__name__)

class RAGPromptBuilder:
    SYSTEM_PROMPT = (
        "You are a highly precise, professional AI assistant operating "
        "within a strict Retrieval-Augmented Generation (RAG) system. "
        "Your sole purpose is to answer the user's query based "
        "EXCLUSIVELY on the provided context snippets.\n\n"

        "CRITICAL INSTRUCTIONS:\n"

        "1. GROUNDING: Do not use outside knowledge. "
        "If the answer is not contained within the context, "
        "state that the information is unavailable.\n"

        "2. CITATION: Every factual claim you make MUST be "
        "immediately followed by a citation referencing the "
        "specific chunk it came from. Use the exact format: "
        "[Source: <chunk_id>].\n"

        "3. TONE: Be direct, objective, and professional. "
        "Avoid filler introductions such as "
        "\"Based on the context provided...\".\n"

        "4. SYNTHESIS: If multiple chunks contain relevant "
        "information, synthesize them logically while citing "
        "each distinct factual claim.\n"

        "5. SOURCE INTEGRITY: Use only chunk IDs that actually "
        "appear in the provided context. Never invent, modify, "
        "or fabricate a source ID."
    )

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Transforms the list of reranked candidate dictionaries into a unified,
        """
        if not chunks:
            logger.debug("No context chunks provided for formatting.")
            return "No relevant context found."

        context_part = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = chunk.get("chunk_id", f"unknown_chunk_{index}")
            content = chunk.get("content", "").strip()
            if not content:
                logger.warning(
                    f"Skipping empty context chunk: {chunk_id}"
                )
                continue

            # Formats each piece of context clearly for the LLM's attention mechanism
            formatted_chunk = (
                f"--- BEGIN CHUNK [{chunk_id}] ---\n"
                f"{content}\n"
                f"--- END CHUNK [{chunk_id}] ---"
            )
            context_part.append(formatted_chunk)

        return "\n\n".join(context_part)

    @classmethod
    def build_messages(
        cls,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Constructs the final messages array ready to be dispatched to the Model(Groq API).
        """
        if not query or not query.strip():
                raise ValueError("Query can't be empty.")
        logger.debug(f"Building LLM prompt payload for query: '{query}'")
        formatted_context = cls.format_context(chunks)
        user_content = (
            f"USER QUERY:\n{query}\n\n"
            f"AVAILABLE CONTEXT:\n{formatted_context}\n\n"
            "INSTRUCTIONS:\n"
            "Answer the query using only the context above. Remember to append citations "
            "(e.g., [Source: doc_name_chunk_2]) to your facts."
        )
        messages = [
            {
                "role": "system",
                "content": cls.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        return messages

if __name__ == "__main__":
    # Test execution for the prompt builder
    test_query = "What is the primary product of Econocom?"
    
    mock_reranked_chunks = [
        {
            "chunk_id": "econocom_annual_report_chunk_42",
            "content": "Econocom primarily focuses on financing and orchestrating digital transformation for enterprises.",
            "cross_encoder_score": 0.98
        }
    ]
    
    messages_payload = RAGPromptBuilder.build_messages(test_query, mock_reranked_chunks)
    
    print("\n--- System Message ---")
    print(messages_payload[0]["content"])
    
    print("\n--- User Message ---")
    print(messages_payload[1]["content"])