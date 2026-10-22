from config.settings import settings
from src.utils.logger import get_logger
from src.utils.helpers import time_it
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
logger = get_logger(__name__)

class EmbeddingModelWrapper:
    # Because it uses FastEmbed (ONNX Runtime), it runs exceptionally fast on CPU without PyTorch.
    _instance = None
    def __new__(cls):
        # Ensure Embedding Model is Loads Only One time into Memory
        if cls._instance is None:
            logger.info(f"Loading Fast Embedding Model: [{settings.EMBEDDING_MODEL_NAME}]")
            try:
                cls._instance = FastEmbedEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    batch_size=8,
                    parallel=None,
                    threads=1,
                )
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to load embedding model: {e}",
                    exc_info=True
                )
                raise
        return cls._instance
    @classmethod
    def get_embeddings(cls) -> FastEmbedEmbeddings:
        # retrieves the initialized langchain Embedding Model
        return cls()
    @classmethod
    def embed_query(cls, text: str):
        return cls.get_embeddings().embed_query(text)
    @classmethod
    def embed_documents(cls, texts):
        return cls.get_embeddings().embed_documents(texts)
    @classmethod
    def embedding_dimension(cls) -> int:
        return len(cls.embed_query("dimension test"))
    
if __name__ == "__main__":
    @time_it
    def test_embed():
        vector = EmbeddingModelWrapper.embed_query(
            "What are the certifications of Econocom?"
        )
        print(f"Vector dimension: {len(vector)}")
    test_embed()