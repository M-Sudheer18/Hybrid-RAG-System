import re
from typing import List
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.helpers import time_it, generate_file_hash
logger = get_logger(__name__)

class DocumentCleaner:
    # Cleans and Normalizes the Extracted Text Before Chunking and Embedding
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove Null bytes and Standard unprinted Chars
        text = text.replace("\x00", "")
        # Ensure Clean UTF-8 Encoding
        text = text.encode("utf-8", "ignore").decode("utf-8")
        # Replaces all consecutive whitespaces, tabs, and newlines with a single space.
        text = re.sub(r"\s+", " ", text)
        # Remove Start and Bottom Spaces 
        text = text.strip()
        return text

    @classmethod
    @time_it
    def clean_documents(cls, documents: List[Document]) -> List[Document]:
        """
        Go through the Langchain Documents objects, Cleaning their page content in place
        Filters out the Document thats Comes Empty after the Extration..
        """
        logger.info(f"Starting the Text Sanitation for {len(documents)} Document Pages..")
        cleaned_docs = []
        for doc in documents:
            cleaned_text = cls.clean_text(doc.page_content)
            # Only Retain the Documents Which Possess the Meaningfull text inside it
            if len(cleaned_text) > 10:
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)
        logger.info(f"Sanitation Completed. Retained {len(cleaned_docs)}")
        return cleaned_docs

# For testing
if __name__ == "__main__":
    sample_doc = Document(page_content="This is a test. \n\n   It has weird \x00 bytes and \t spaces.")
    cleaner = DocumentCleaner()
    cleaned = cleaner.clean_documents([sample_doc])
    print(cleaned[0].page_content)