from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage  # For prompt formatting
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM using HuggingFaceEndpoint...")

        if not hf_token:
            raise CustomException("HF_TOKEN is missing in environment variables.")

        # Use HuggingFaceEndpoint with explicit task parameter
        endpoint = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            task="text-generation",  # This is REQUIRED
            temperature=0.1,
            max_new_tokens=512,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )
        llm = ChatHuggingFace(llm=endpoint)
        logger.info(f"LLM loaded successfully: {huggingface_repo_id}")
        return llm

    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException(f"Error loading LLM: {e}")