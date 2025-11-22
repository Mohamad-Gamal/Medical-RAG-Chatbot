from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM from HuggingFace...")

        if not hf_token:
            raise CustomException("HF_TOKEN is missing in environment variables.")

        # Fixed configuration for text generation models
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            task="text-generation",  # Changed from "conversational"
            temperature=0.1,  # Lower for medical accuracy
            #max_length=512,
            top_p=0.9,
            do_sample=True,
            return_full_text=False  # Don't repeat the input
        )

        logger.info("LLM loaded successfully.")
        return llm

    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException(f"Error loading LLM: {e}")