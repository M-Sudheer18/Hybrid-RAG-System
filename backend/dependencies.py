from fastapi import HTTPException, Request
from src.orchestration.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger
logger = get_logger(__name__)

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """
    FastAPI Dependency to retrieve the initialized RAG Pipeline from the application state.
    This ensures that all incoming API requests reuse the exact same pipeline instance
    (and its heavy ML models / FAISS indexes) created during the server startup lifespan.
    """
    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None
    )
    if pipeline is None:
        logger.error("Rag Pipeline Request by Endpoint but Not found in Application..")
        raise HTTPException(
            status_code=503,
            detail="Service is Unavailable: Rag Pipeline is Missing or Unable to Load.."
        )
    return pipeline



# FastAPI Startup
#       ↓
# Create RAGPipeline()
#       ↓
# app.state.rag_pipeline
#       ↓
#      API
#       ↓
# get_rag_pipeline()
#       ↓
# Same RAGPipeline instance