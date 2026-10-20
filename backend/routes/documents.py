import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from config.settings import settings
from backend.schemas.document_schema import (
    UploadResponse,
    DocumentInfo,
    DocumentListResponse,
)
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.meta_manager import MetadataManager
from src.utils.logger import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])
TEMP_UPLOAD_DIR = settings.DATA_DIR / "temp_uploads"

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Saves an uploaded PDF to the raw data directory and triggers the ingestion pipeline 
    to rebuild the FAISS and BM25 indexes synchronously in a background thread.
    """
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only Pdf Files are Allowed"
        )
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Saves the file to Raw pdf Directory..
    safe_filename = Path(file.filename).name
    file_path = TEMP_UPLOAD_DIR / safe_filename

    try:
        # Avoid overwriting an existing upload with the same filename.
        if file_path.exists():
            file_path = TEMP_UPLOAD_DIR / (
                f"{file_path.stem}_{uuid.uuid4().hex[:8]}.pdf"
            )
        with file_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
    except Exception as e:
        logger.exception("Failed to save uploaded PDF.")
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded PDF."
        ) from e
    finally:
        await file.close()

    # Trigger the heavy ingestion process in a thread pool to avoid blocking the API
    try:
        pipeline = IngestionPipeline()
        await run_in_threadpool(pipeline.run_pipeline)
        metadata_manager = MetadataManager()
        file_hash = metadata_manager.compute_file_hash(file_path)

        record = next(
            (
                item
                for item in metadata_manager.registry.values()
                if item.get("file_hash") == file_hash
            ),
            None
        )

        if not record:
            raise HTTPException(
                status_code=500,
                detail="PDF was saved, but its metadata was not registered."
            )

        status = record.get("status", "Unknown")
        return UploadResponse(
            message=(
                "Temporary PDF successfully indexed."
                if status == "completed"
                else f"PDF Processing Status: {status}"
            ),
            doc_id=record.get("doc_id"),
            filename=record.get("filename", safe_filename),
            status=status,
            chunk_count=record.get("chunk_count", 0)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PDF Ingestion Failed")
        raise HTTPException(
            status_code=500,
            detail="PDF was saved, but indexing failed."
        ) from exc


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    Retrieves the complete registry of ingested documents
    """
    try:
        metadat_manager = MetadataManager()
        documents = []
        for doc_id, data in metadat_manager.registry.items():
            documents.append(DocumentInfo(**data))
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
    except Exception as e:
        logger.error(f"Failed to fetch the document registry: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieved the Document List"
        )

@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """
    Retrieves metadata for a single specific document by its ID.
    """
    metadata_manager = MetadataManager()
    doc_data = metadata_manager.registry.get(doc_id)
    if not doc_data:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{doc_id}' not found."
        )
    return DocumentInfo(**doc_data)