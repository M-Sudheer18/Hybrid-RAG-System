from fastapi import HTTPException, APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.dependencies import get_rag_pipeline
from src.orchestration.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger
logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"]
)
@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Main conversational endpoint. 
    Injects the shared RAGPipeline instance, processes the user query with 
    session memory, and returns the generated answer alongside source citations.
    """
    logger.info(f"Received chat request for session_id: {request.session_id}")
    try:
        # Offload the heavy synchronous execution to the pipeline   
        result = await run_in_threadpool(
            rag_pipeline.execute,
            session_id = request.session_id,
            user_query = request.query
        )
        # Pydantic will automatically map the dict keys to the ChatResponse schema
        return ChatResponse(**result)
    except ValueError as ve:
        logger.warning(f"Validation Error Due to Chat Execution: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(
            f"Critical Failure Occurred during chat Execution: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An Internal error Occurs while processing request.."
        )


