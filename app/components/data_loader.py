import os
from langchain_components.vectorstores import FAISS
import app.config.config as config
from app.common.logger import get_logger
from app.common.custom_exception import Custom_Exception
from app.components.pdf_loader import load_pdf_files, create_text_chunk
from app.components.vector_store import save_vector_store

logger = get_logger(__name__)


def process_store_pdfs():
    try:
        logger.info("Starting PDF processing and vector store creation/loading")

        # Load PDF files
        documents = load_pdf_files()
        if not documents:
            raise Custom_Exception("No documents loaded from PDFs.")

        # Create text chunks
        text_chunks = create_text_chunk(documents)
        if not text_chunks:
            raise Custom_Exception("No text chunks created from documents.")

        # Save or load vector store
        vector_store = FAISS.load_local(config.DB_FAISS_PATH, None) if os.path.exists(config.DB_FAISS_PATH) else None
        if vector_store is None:
            logger.info("Vector store not found, creating a new one.")
            raise Custom_Exception("Failed to create and save vector store.")
            return None
        
        else:
            vector_store = save_vector_store(text_chunks)
            logger.info("PDF processing and vector store setup completed successfully.")
            return vector_store

    except Exception as e:
        error_message = Custom_Exception("Failed to process PDFs and setup vector store!", e)
        logger.error(str(error_message))
        return None



if __name__ == "__main__":
    process_store_pdfs()