from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentInfo(BaseModel):
    """Schema representing the metadata of the Single ingested Document"""
    doc_id: str = Field(
        ...,
        description="Unique Identifier Generated during registration"
    )
    file_name: str = Field(
        ...,
        description="Original Name of the Uploaded File"
    )
    file_hash: str = Field(
        ...,
        description="SHA-256 hash used to prevent duplicate ingestion"
    )
    source_type: str = Field(
        ...,
        description="Format of the Source File"
    )
    ingestion_time: datetime = Field(
        ...,
        description="Timestamp of When the Document is Registered.."
    )
    status: str = Field(
        ...,
        description="Current state (e.g., 'processing', 'chunked', 'failed')."
    )
    chunk_count: int = Field(
        default=0,
        description="Total number of vector chunks extracted."
    )

class DocumentListResponse(BaseModel):
    """Payload for endpoints that return the entire ingested corpus registry."""
    documents: List[DocumentInfo]
    total_count: int = Field(
        ...,
        description="Total number of documents in the system."
    )

class UploadResponse(BaseModel):
    """Pay Load Sent Back to the Frontend/User after a Document Upload Finishes"""
    message: str = Field(
        ...,
        description="Status Message detailing the Output.."
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="The UUID if the document was successfully ingested."
    )
    filename: str = Field(
        ...,
        description="Name of the Processed File"
    )
    status: str = Field(
        ..., 
        description="Final ingestion status (e.g., 'chunked', 'skipped')."
    )
    chunk_count: int = Field(
        ...,
        description="No.of the Chunks Successfully indexed.."
    )
