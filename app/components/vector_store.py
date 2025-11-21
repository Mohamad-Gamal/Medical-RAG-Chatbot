import os
from langchain_community.vectorstores import FAISS
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.embedding import get_embedding_model

logger = get_logger(__name__)

def load_vector_store():    
    try:
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise CustomException("Embedding model could not be loaded.")

        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading FAISS vector store from {DB_FAISS_PATH}")
            vector_store = FAISS.load_local(
                DB_FAISS_PATH,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded FAISS vector store")
            return vector_store
        else:
            logger.info(f"FAISS vector store path does not exist: {DB_FAISS_PATH}")
            return None

    except Exception as e:
        error_message = CustomException("Failed to load FAISS vector store!", e)
        logger.error(str(error_message))
        return None


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("Text chunks are empty or None.")

        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise CustomException("Embedding model could not be loaded.")

        vector_store = FAISS.from_documents(
            documents=text_chunks,
            embedding=embedding_model
        )

        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        vector_store.save_local(DB_FAISS_PATH)

        logger.info(f"Successfully saved FAISS vector store at: {DB_FAISS_PATH}")
        return vector_store

    except Exception as e:
        error_message = CustomException("Failed to create/save FAISS vector store!", e)
        logger.error(str(error_message))
        return None
