from langchain_community.vectorstores import FAISS
import os
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.embedding import get_embedding_model


logger = get_logger(__name__)

def load_vector_store():    
    try:
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise Custom_Exception("Embedding model could not be loaded.")

        logger.info("Loading FAISS vector store from disk")
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"FAISS vector store path exists: {DB_FAISS_PATH}")
            vector_store = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dengerous_deserialization=True)
            logger.info("Successfully loaded FAISS vector store")
            return vector_store
        
        else:
            raise Custom_Exception(f"FAISS vector store path does not exist: {DB_FAISS_PATH}")

    except Exception as e:
        error_message = Custom_Exception("Failed to load FAISS vector store!", e)
        logger.error(str(error_message))
        return None

# create vector store DB
def save_vector_store(text_chunks):
    try:
        if text_chunks is None:
            raise Custom_Exception("Text chunks are None")
        logger.info("Creating FAISS vector store and saving to disk")
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise Custom_Exception("Embedding model could not be loaded.")
            return None
        vector_store = FAISS.from_documents(texts=text_chunks, embedding=embedding_model)
        vector_store.save_local(DB_FAISS_PATH)
        logger.info(f"Successfully saved FAISS vector store at: {DB_FAISS_PATH}")
        return vector_store

    except Exception as e:
        error_message = Custom_Exception("Failed to create/save FAISS vector store!", e)
        logger.error(str(error_message))
        return None