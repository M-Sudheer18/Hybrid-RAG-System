from typing import List, Dict, Optional
from collections import deque
from config.settings import settings
from src.utils.logger import get_logger
logger = get_logger(__name__)

class ConversationMemory:
    """
    Manages conversational history for a single dialogue session using a sliding window.
    Retains the most recent interaction turns to preserve relevant context while
    preventing prompt bloat and latency spikes.
    """
    def __init__(
        self,
        max_turns: int = getattr(settings, "MAX_HISTORY_TURNS", 5)
    ):
        # Each Consists of Two Messages ( user + assistant )
        self.max_messages = max_turns * 2
        self.history: deque = deque(maxlen=self.max_messages)

    def add_user_message(
        self, 
        content: str
    ) -> None:
        if not content or not content.strip():
            logger.warning("Attempted to add empty user message to memory.")
            return
        self.history.append({
            "role": "user",
            "content": content.strip()
        })

    def add_assistant_message(
        self,
        content: str
    ) -> None:
        # Appends the assistants Response to the History..
        if not content or not content.strip():
            logger.warning("Attempted to add empty assistant Message to Memory..")
            return
        self.history.append({
            "role": "assistant",
            "content": content.strip()
        })

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def clear_history(self) -> None:
        self.history.clear()
        logger.info("Conversation Memory Cleared..")

    def __len__(self) -> int:
        return len(self.history)

class SessionMemoryManager:
    """
    In-memory session registry for conversation sessions.
    """
    _instance: Optional["SessionMemoryManager"] = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionMemoryManager, cls).__new__(cls)
            cls._instance.sessions: Dict[str, ConversationMemory] = {}
            logger.info("SessionMemoryManager initialized.")
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SessionMemoryManager":
        return cls()

    def get_or_create_session(
        self,
        session_id: str,
        max_turns: int = getattr(settings, "MAX_HISTORY_TURNS", 5)
    ) -> ConversationMemory:
        """
        Retrieves an existing memory instance or creates a new one for the session.
        """
        if not session_id or not session_id.strip():
            raise ValueError("Session_id Can't be Empty..")

        if session_id not in self.sessions:
            logger.info(f"Creating new conversation session: '{session_id}'")
            self.sessions[session_id] = ConversationMemory(max_turns=max_turns)
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> bool:
        """
        Removes a session buffer from memory.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session '{session_id}' removed from memory.")
            return True
        return False

if __name__ == "__main__":
    manager = SessionMemoryManager.get_instance()
    session = manager.get_or_create_session("session_user_123", max_turns=2)

    # Turn 1
    session.add_user_message("What is Econocom's revenue?")
    session.add_assistant_message("Econocom reported revenue of approximately €2.7B.")

    # Turn 2
    session.add_user_message("What about their employees?")
    session.add_assistant_message("They employ over 8,500 people across Europe.")

    # Turn 3 (Triggers sliding window eviction of Turn 1)
    session.add_user_message("Where are they headquartered?")
    session.add_assistant_message("Their headquarters are located in Brussels, Belgium.")

    current_history = session.get_history()
    print(f"Active message count: {len(current_history)} (Expected: 4 messages / 2 turns)")
    for msg in current_history:
        print(f"[{msg['role'].upper()}]: {msg['content']}")