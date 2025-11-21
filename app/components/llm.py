# app/components/llm.py
from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading conversational LLM from HuggingFace...")

        if not hf_token:
            raise CustomException("HF_TOKEN is missing in environment variables.")

        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="conversational",        # Ensure the model is used in conversational mode
            temperature=0.4,
            huggingfacehub_api_token=hf_token
        )

        logger.info("LLM loaded successfully.")
        return llm

    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException(f"Error loading LLM: {e}")
