from fastapi import APIRouter, Request
from src.utils.logger import get_logger
logger = get_logger(__name__)
router = APIRouter(
    prefix="/api/health",
    tags=["health"]
)

@router.get("/")
async def health_check(request: Request):
    """
    Liveness and status probe.
    Verifies the API server is running and checks if the RAG pipeline 
    has finished mounting its models and indexes into application state.
    """
    # Safety Checking app state without triggering the dependency exceptions
    pipeline_loaded = getattr(
        request.app.state,
        "rag_pipeline",
        None
    ) is not None
    status_payload = {
        "status": "healthy",
        "pipeline_payload": pipeline_loaded
    }
    logger.debug(f"Health check pinged. Pipeline loaded: {pipeline_loaded}")
    return status_payload