from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hybrid Rag System"
    VERSION: str = "1.0.0"

    # Path Config
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_PDF_DIR: Path = DATA_DIR / "raw_pdfs"
    SCRAPED_DATA_DIR: Path = DATA_DIR / "scraped_data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
    METADATA_DIR: Path = DATA_DIR / "metadata" 

    # Vector Store Directories
    FAISS_INDEX_DIR: Path = VECTOR_STORE_DIR / "faiss_index"
    BM25_INDEX_DIR: Path = VECTOR_STORE_DIR / "bm25_index" 

    # # Temporary Files
    # TEMP_UPLOAD_DIR = DATA_DIR / "temp_upload"
    # TEMP_UPLOAD_DIR.mkdir(
    #     parents=True,
    #     exist_ok=True
    # )

    # Chunking Parameters
    CHUNK_SIZE: int = Field(default=800, description="Max Characters per Chunk")
    CHUNK_OVERLAP: int = Field(default=150, description="Overlap to Remain Semantic Meaning")

    # Embeddings and Reranker Models
    EMBEDDING_MODEL_NAME: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local Embeddings Model From Hugging Face.."
    )
    RERANKER_MODEL_NAME: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        description="FastEmbed ONNX Cross-Encoder Model for Hybrid Search Reranking"
    )

    # Hybrid Retrieval Parameters
    TOP_K_SEARCH: int = Field(default=15, description="No.of results from FAISS Vector Store")
    TOP_K_SPARSE: int = Field(default=15, description="No.of results from BM25 Hybrid Search")
    TOP_N_RERANK: int = Field(default=5, description="Final No.of Chunks Sent to LM")
    RRF_K_CONSTANT: int = Field(default=60, description="Smoothing Constant for Reciprocal Rank Fusion")

    # Groq Api
    # Default as None to get Error if Key is Not There
    GROQ_API_KEY: str | None = Field(
        default=None, 
        description="API Key From GROQ CLOUD"
    )
    GROQ_MODEL_NAME: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq LLM for grounded RAG generation"
    )
    TEMPERATURE: float = Field(
        default=0.0, 
        description="0.0 -> Ensures that Factual Grouding Generation"
    )
    MAX_TOKENS: int = Field(
        default=2048, 
        description="Max response length"
    )

    # Connect to .env File
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Validators + Directory Setup
    @model_validator(mode="after")
    def setup_directories(self) -> "Settings":
        # Chunks Validation
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                "Chunking Overlap Must be Smaller than Chunks Size"
            )
        directories = [
            self.DATA_DIR,
            self.RAW_PDF_DIR,
            self.SCRAPED_DATA_DIR,
            self.PROCESSED_DIR,
            self.METADATA_DIR,
            self.VECTOR_STORE_DIR,
            self.FAISS_INDEX_DIR,
            self.BM25_INDEX_DIR
        ]
        for dic in directories:
            dic.mkdir(
                parents=True, 
                exist_ok=True
            )

        if not self.GROQ_API_KEY:
            print("WARNING: GROQ_API_KEY is not set in the .env file. Generation will fail.")
        return self

# Initantiate it 
settings = Settings()