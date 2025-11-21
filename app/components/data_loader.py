from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.pdf_loader import load_pdf_files, create_text_chunk
from app.components.vector_store import save_vector_store, load_vector_store

logger = get_logger(__name__)

def process_store_pdfs():
    try:
        # Try to load existing vector store
        vector_store = load_vector_store()
        if vector_store:
            logger.info("Vector store loaded successfully.")
            return vector_store

        logger.info("No existing vector store found. Creating a new one...")

        # Load PDFs
        documents = load_pdf_files()
        if not documents:
            raise CustomException("No PDF documents loaded.")

        # Create text chunks
        text_chunks = create_text_chunk(documents)
        if not text_chunks:
            raise CustomException("No text chunks generated.")

        # Save vector store
        vector_store = save_vector_store(text_chunks)
        logger.info("Vector store created successfully.")
        return vector_store

    except Exception as e:
        error_message = CustomException("Failed to process PDFs and setup vector store!", e)
        logger.error(str(error_message))
        return None


if __name__ == "__main__":
    process_store_pdfs()
