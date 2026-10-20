from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    """Payload Receives From the Frontend/User..."""
    session_id: str = Field(
        ...,
        description="Unique identifier to retrieve conversation memory."
    )
    query: str = Field(
        ...,
        description="The Raw User Question"
    )

class SourceChunk(BaseModel):
    """Data structure for returning the specific chunks used to build the answer."""
    chunk_id: str
    score: float
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    """Payload Sent Back to Frontend/User"""
    session_id: str
    original_query: str
    standalone_query: str
    answer: str
    sources: List[SourceChunk]