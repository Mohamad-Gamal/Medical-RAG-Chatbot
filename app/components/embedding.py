from langchain_huggingface import HuugingFaceEmbeddingsEmbeddings
from app.config.config import HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL_NAME

from app.common.logger import get_logger
from app.common.custom_exception import Custom_Exception

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Loading HuggingFace Embeddings model")
        embeddings = HuugingFaceEmbeddingsEmbeddings(
            model_name=HUGGINGFACE_MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )
        logger.info("Successfully loaded HuggingFace Embeddings model")
        return embeddings

    except Exception as e:
        error_message = Custom_Exception("Failed to load HuggingFace Embeddings model!", e)
        logger.error(str(error_message))
        return None