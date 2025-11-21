from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config.config import HF_TOKEN, HUGGINGFACE_MODEL_NAME, HUGGINGFACE_REPO_ID

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Loading HuggingFace Embeddings model")
        embeddings = HuugingFaceEmbeddings(
            model_name=HUGGINGFACE_MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACE_REPO_ID
        )
        logger.info("Successfully loaded HuggingFace Embeddings model")
        return embeddings

    except Exception as e:
        error_message = CustomException("Failed to load HuggingFace Embeddings model!", e)
        logger.error(str(error_message))
        return None