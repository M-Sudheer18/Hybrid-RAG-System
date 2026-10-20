import shutil
from pathlib import Path
from config.settings import settings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.utils.logger import get_logger
from src.ingestion.meta_manager import MetadataManager
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.orchestration.rag_pipeline import RAGPipeline


# API Routes
from backend.routes.health import router as health_router
from backend.routes.chat import router as chat_router
from backend.routes.documents import router as document_router
logger = get_logger(__name__)


# Frontend Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "web_app"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

# Auto-Create
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Jinja2 Templates
templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)

TEMP_UPLOAD_DIR = settings.DATA_DIR / "temp_uploads"
RAW_PDF_DIR = settings.DATA_DIR / "raw_pdfs"

def remove_saved_indexes():
    index_files = [
        settings.FAISS_INDEX_DIR / "dense.faiss",
        settings.FAISS_INDEX_DIR / "metadata.json",
        settings.BM25_INDEX_DIR / "bm25.pkl",
        settings.BM25_INDEX_DIR / "metadata.json"
    ]
    for index_file in index_files:
        index_file.unlink(missing_ok=True)

def cleanup_temporary_uploads():
    logger.info("Starting temporary PDF cleanup..")
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    metadata_manager = MetadataManager()
    
    # Delete files and folders inside the temporary directory only.
    for path in TEMP_UPLOAD_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    # Remove temporary upload records from the document registry.
    original_count = len(metadata_manager.registry)

    metadata_manager.registry = {
        doc_id: record
        for doc_id, record in metadata_manager.registry.items()
        if (
            doc_id != "doc_6e469e71" and
            str(record.get("source_type", "")).strip().lower()
            != "temporary_upload"
        )
    }
    metadata_manager._save_registry()
    removed_records = original_count - len(metadata_manager.registry)
    logger.info(
        f"Removed {removed_records} AI&ML or temporary-upload records."
    )

    # Rebuild the indexes exclusively from permanent PDFs.
    permanent_pdfs = (
        sorted(
            path for path in RAW_PDF_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if RAW_PDF_DIR.exists()
        else []
    )

    if not permanent_pdfs:
        logger.info("No permanent PDFs found. Removing stale indexes.")
        remove_saved_indexes()
        return
    try:
        pipeline = IngestionPipeline(raw_data_dir=RAW_PDF_DIR)
        result = pipeline.run_pipeline()

        if result.get("chunks", 0) == 0:
            logger.warning(
                "No permanent document chunks were indexed. "
                "Removing potentially stale indexes."
            )
            remove_saved_indexes()
        else:
            logger.info(
                f"Permanent document indexes rebuilt: "
                f"{result['chunks']} chunks."
            )
    except Exception:
        # Never leave old indexes available if rebuilding fails.
        remove_saved_indexes()
        logger.exception("Failed to rebuild permanent document indexes.")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager. 
    Initializes the shared RAGPipeline (which loads the embedding model, 
    cross-encoder, and FAISS/BM25 indexes) before accepting HTTP traffic.
    """
    logger.info("Starting up FastAPI App")
    try:
        cleanup_temporary_uploads()
        app.state.rag_pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized Successfully..")
    except Exception as e:
        logger.error(f"Failed to Initialize the RAG Pipeline during Startup: {e}")
        raise RuntimeError(
            "Failed to initialize the application."
        ) from e
    # The application runs and accepts requests while yielded here
    yield

    # App Shutdown
    logger.info("Shutting Down FastAPI application..")
    app.state.rag_pipeline = None

# Create the Fast API Application
app = FastAPI(
    title="RAG API",
    description="High-performance dual-retrieval RAG backend using FastEmbed, FAISS, BM25, and Groq.",
    version='1.0.0',
    lifespan=lifespan
)

# Configure Cors for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], # For production, replace "*" with your frontend's specific URL
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Mount the Static Directory to Serve the css, js and images
app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ), name='static'
)

# Register the health, chat, and document routes
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(document_router)

# UI Route
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_frontend(request: Request):
    # Serves the main frontend application HTML.
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/home", response_class=HTMLResponse, tags=["UI"])
async def home_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html"
    )

@app.get("/chat", response_class=HTMLResponse, tags=["UI"])
async def chat_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html"
    )

@app.get("/pdf", response_class=HTMLResponse, tags=["UI"])
async def pdf_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pdf.html"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True  
        )