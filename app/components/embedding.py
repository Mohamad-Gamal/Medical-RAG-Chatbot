from langchain_huggingface import HuggingFaceEmbeddings
from app.config.config import HF_TOKEN, HUGGINGFACE_MODEL_NAME
import os
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        if not HF_TOKEN:
            raise CustomException("HF_TOKEN is not set in environment variables.")
    
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        logger.info(f"Loading HuggingFace Embeddings model: {HUGGINGFACE_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
        logger.info("Successfully loaded HuggingFace Embeddings model")
        return embeddings

    except Exception as e:
        error_message = CustomException("Failed to load HuggingFace Embeddings model!", e)
        logger.error(str(error_message))
        return None
