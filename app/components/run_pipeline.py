import shutil
import os
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.pdf_loader import load_pdf_files, create_text_chunk
from app.components.vector_store import save_vector_store, load_vector_store
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def rebuild_vector_store(force_rebuild=True):
    try:
        if force_rebuild and os.path.exists(DB_FAISS_PATH):
            logger.info(f"Removing existing FAISS vector store at {DB_FAISS_PATH}...")
            shutil.rmtree(DB_FAISS_PATH)

        # Try to load existing vector store
        vector_store = load_vector_store()
        if vector_store:
            logger.info("Vector store loaded successfully.")
            return vector_store

        logger.info("Creating a new FAISS vector store from PDFs...")

        # Load PDFs
        documents = load_pdf_files()
        if not documents:
            raise CustomException("No PDF documents found.")

        logger.info(f"Loaded {len(documents)} PDF documents.")

        # Split into text chunks
        text_chunks = create_text_chunk(documents)
        if not text_chunks:
            raise CustomException("No text chunks generated from PDFs.")

        logger.info(f"Generated {len(text_chunks)} text chunks.")

        # Create and save FAISS vector store
        vector_store = save_vector_store(text_chunks)
        if vector_store:
            logger.info("FAISS vector store successfully created and saved.")
        else:
            logger.warning("FAISS vector store creation failed.")

        return vector_store

    except Exception as e:
        error_message = CustomException("Pipeline failed!", e)
        logger.error(str(error_message))
        return None


if __name__ == "__main__":
    rebuild_vector_store(force_rebuild=True)
