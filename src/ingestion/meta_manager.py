import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import settings
from src.utils.logger import get_logger
logger = get_logger(__name__)

class MetadataManager:
    """
    Manages document-level metadata and state tracking during the ingestion phase.
    Maintains a central registry to prevent duplicate processing of identical files 
    by tracking SHA-256 file hashes.
    """
    def __init__(self):
        # We store the registry in the same data directory as our indexes
        self.registry_path = settings.DATA_DIR / "document_registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads the existing document registry from disk if it exists."""
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load document registry: {e}")
            return {}

    def _save_registry(self) -> None:
        """Saves it to the Disk.."""
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Unable to Save the Document: {e}")

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """
        Computes a SHA-256 hash of the file content.
        This is heavily utilized to detect if a file has already been ingested.
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to prevent memory spikes on massive PDFs
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to Compute hash for {file_path.name}: {e}")
            raise

    def is_file_ingested(self, file_hash: str) -> bool:
        """Checks if a file with the given hash already exists in the registry."""
        for doc_id, doc_metadata in self.registry.items():
            if doc_metadata.get("file_hash") == file_hash:
                return True
        return False

    def register_document(
        self,
        file_path: Path,
        source_type: str = "PDF",
    ) -> Optional[str]:
        """
        Calculates the file hash and registers the document if it's new.
        Returns a unique document ID (UUID) if registered, or None if already ingested.
        """
        file_hash = self.compute_file_hash(file_path)
        if self.is_file_ingested(file_hash):
            logger.info(f"Skipping '{file_path.name}'. Document already ingested.")
            return None

        # Generate a Unique ID for this Specific Document..
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        self.registry[doc_id] = {
            "doc_id": doc_id,
            "file_name": file_path.name,
            "file_hash": file_hash,
            "source_type": source_type,
            "ingestion_time": datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "chunk_count": 0
        }
        self._save_registry()
        logger.info(f"Registered New Document: '{file_path.name}' [{doc_id}]")
        return doc_id

    def update_document_status(
        self,
        doc_id: str,
        chunk_count: int,
        status: str = "completed"
    ) -> None:
        """Updates the registry with the final chunk count after the chunker processes it."""
        if doc_id in self.registry:
            self.registry[doc_id]["chunk_count"] = chunk_count
            self.registry[doc_id]["status"] = status
            self._save_registry()
            logger.debug(f"Updated document {doc_id} status to '{status}' with {chunk_count} chunks.")
        else:
            logger.warning(f"Attempted to Update a unknown Document ID: {doc_id}")


if __name__ == "__main__":
    # Quick isolated test
    manager = MetadataManager()
    
    # Create a dummy file to test hashing and registration
    test_file = Path("test_dummy.txt")
    test_file.write_text("This is a test document for Econocom.")
    
    # 1st Registration (Should Succeed)
    doc_id = manager.register_document(test_file, source_type="TXT")
    if doc_id:
        manager.update_document_status(doc_id, chunk_count=5)
        
    # 2nd Registration of same file (Should Skip)
    duplicate_check = manager.register_document(test_file, source_type="TXT")
    
    print("\nCurrent Registry State:")
    print(json.dumps(manager.registry, indent=2))
    
    # Clean up test file
    test_file.unlink()